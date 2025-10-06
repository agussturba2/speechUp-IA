"""
Unit tests for Phase 2 components.

Tests AudioProcessor, VideoBufferManager, MetricsAnalyzer, and SessionCoordinator.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Import components
from api.websockets.audio_processor import AudioProcessor
from api.websockets.video_buffer_manager import VideoBufferManager
from api.websockets.metrics_analyzer import MetricsAnalyzer, AccumulatedState
from api.websockets.session_coordinator import SessionCoordinator, SessionFactory
from api.websockets.models import (
    AudioAnalysisResult, Word, FillerInstance, PauseSegment, PauseAnalysis,
    FrameAnalysisResult, IncrementalMetrics
)


# ============================================================================
# AudioProcessor Tests
# ============================================================================

class TestAudioProcessor:
    """Tests for AudioProcessor component."""
    
    @pytest.fixture
    def audio_processor(self):
        """Create AudioProcessor for testing."""
        processor = AudioProcessor(sample_rate=16000, vad_mode=2)
        yield processor
        processor.close()
    
    @pytest.fixture
    def sample_audio(self):
        """Create 1 second of sample audio data."""
        samples = np.zeros(16000, dtype=np.int16)
        return samples.tobytes()
    
    def test_initialization(self, audio_processor):
        """Test AudioProcessor initializes correctly."""
        assert audio_processor.sample_rate == 16000
        assert len(audio_processor.audio_buffer) == 0
        assert audio_processor.audio_buffer_duration == 0.0
    
    def test_add_audio(self, audio_processor, sample_audio):
        """Test adding audio to buffer."""
        audio_processor.add_audio(sample_audio)
        assert len(audio_processor.audio_buffer) > 0
        assert audio_processor.audio_buffer_duration > 0.9  # ~1 second
    
    def test_get_unprocessed_bytes(self, audio_processor, sample_audio):
        """Test tracking unprocessed audio."""
        audio_processor.add_audio(sample_audio)
        unprocessed = audio_processor.get_unprocessed_bytes()
        assert unprocessed == len(sample_audio)
    
    def test_reset(self, audio_processor, sample_audio):
        """Test resetting processor state."""
        audio_processor.add_audio(sample_audio)
        audio_processor.reset()
        
        assert len(audio_processor.audio_buffer) == 0
        assert audio_processor.audio_buffer_duration == 0.0
        assert audio_processor._last_processed_position == 0


# ============================================================================
# VideoBufferManager Tests
# ============================================================================

class TestVideoBufferManager:
    """Tests for VideoBufferManager component."""
    
    @pytest.fixture
    def video_manager(self):
        """Create VideoBufferManager for testing."""
        manager = VideoBufferManager(width=640, height=480, fps=30.0)
        yield manager
        manager.close()
    
    @pytest.fixture
    def sample_frame(self):
        """Create sample video frame."""
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_initialization(self, video_manager):
        """Test VideoBufferManager initializes correctly."""
        assert video_manager.width == 640
        assert video_manager.height == 480
        assert video_manager.fps == 30.0
        assert video_manager.frame_count == 0
    
    def test_add_frame_with_mock_decode(self, video_manager):
        """Test adding frame with mocked decoder."""
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with patch('api.websockets.video_buffer_manager.decode_frame_data') as mock_decode:
            mock_decode.return_value = mock_frame
            
            result = video_manager.add_frame(b'fake_frame_data')
            
            assert result is True
            assert video_manager.frame_count == 1
            assert len(video_manager.get_all_frames()) == 1
    
    def test_get_recent_frames(self, video_manager):
        """Test getting recent frames."""
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with patch('api.websockets.video_buffer_manager.decode_frame_data') as mock_decode:
            mock_decode.return_value = mock_frame
            
            # Add 5 frames
            for _ in range(5):
                video_manager.add_frame(b'frame')
            
            recent = video_manager.get_recent_frames(count=3)
            assert len(recent) == 3
    
    def test_reset(self, video_manager):
        """Test resetting manager state."""
        mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        with patch('api.websockets.video_buffer_manager.decode_frame_data') as mock_decode:
            mock_decode.return_value = mock_frame
            video_manager.add_frame(b'frame')
            
        video_manager.reset()
        
        assert video_manager.frame_count == 0
        assert len(video_manager.get_all_frames()) == 0


# ============================================================================
# MetricsAnalyzer Tests
# ============================================================================

class TestMetricsAnalyzer:
    """Tests for MetricsAnalyzer component."""
    
    @pytest.fixture
    def metrics_analyzer(self):
        """Create MetricsAnalyzer for testing."""
        analyzer = MetricsAnalyzer()
        yield analyzer
        analyzer.close()
    
    @pytest.fixture
    def sample_audio_result(self):
        """Create sample AudioAnalysisResult."""
        return AudioAnalysisResult(
            duration_sec=10.0,
            speech_detected=True,
            words=[
                Word(word="hello", index=0),
                Word(word="world", index=1)
            ],
            fillers=[
                FillerInstance(type="um", time=2.5, duration=0.3)
            ],
            pauses=[
                PauseSegment(start_time=5.0, duration=1.0)
            ],
            pause_analysis=PauseAnalysis(
                speech_percent=80.0,
                avg_speech_segment=2.0,
                avg_pause_length=1.0,
                pause_frequency=6.0
            )
        )
    
    def test_initialization(self, metrics_analyzer):
        """Test MetricsAnalyzer initializes correctly."""
        assert metrics_analyzer.state.word_count == 0
        assert metrics_analyzer.state.filler_count == 0
        assert metrics_analyzer._cache_dirty is True
    
    def test_update_from_audio(self, metrics_analyzer, sample_audio_result):
        """Test updating from audio result."""
        metrics_analyzer.update_from_audio(sample_audio_result)
        
        assert metrics_analyzer.state.word_count == 2
        assert metrics_analyzer.state.filler_count == 1
        assert metrics_analyzer.state.pause_count == 1
        assert metrics_analyzer.state.audio_processed_sec == 10.0
    
    def test_get_metrics(self, metrics_analyzer, sample_audio_result):
        """Test computing metrics."""
        metrics_analyzer.update_from_audio(sample_audio_result)
        
        metrics = metrics_analyzer.get_metrics()
        
        assert isinstance(metrics, IncrementalMetrics)
        assert metrics.wpm > 0  # Should have calculated WPM
        assert metrics.fillers_per_min > 0
    
    def test_get_confidence(self, metrics_analyzer, sample_audio_result):
        """Test confidence calculation."""
        # Initially low confidence
        confidence1 = metrics_analyzer.get_confidence()
        assert 0.0 <= confidence1 <= 1.0
        
        # After processing, confidence increases
        metrics_analyzer.update_from_audio(sample_audio_result)
        confidence2 = metrics_analyzer.get_confidence()
        
        assert confidence2 >= confidence1
    
    def test_reset(self, metrics_analyzer, sample_audio_result):
        """Test resetting analyzer."""
        metrics_analyzer.update_from_audio(sample_audio_result)
        metrics_analyzer.reset()
        
        assert metrics_analyzer.state.word_count == 0
        assert metrics_analyzer.state.filler_count == 0


# ============================================================================
# SessionCoordinator Tests
# ============================================================================

class TestSessionCoordinator:
    """Tests for SessionCoordinator component."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for testing."""
        audio_processor = Mock(spec=AudioProcessor)
        video_manager = Mock(spec=VideoBufferManager)
        metrics_analyzer = Mock(spec=MetricsAnalyzer)
        
        # Configure mocks
        video_manager.frame_count = 0
        video_manager.frame_buffer = []
        video_manager.get_all_frames.return_value = []
        
        audio_processor.process_audio = AsyncMock(return_value=None)
        
        metrics_analyzer.get_metrics.return_value = IncrementalMetrics(
            wpm=150.0,
            fillers_per_min=2.0,
            gesture_rate=1.0,
            expression_variability=0.5
        )
        metrics_analyzer.get_confidence.return_value = 0.5
        metrics_analyzer.get_recent_fillers.return_value = []
        metrics_analyzer.get_recent_gestures.return_value = []
        
        return audio_processor, video_manager, metrics_analyzer
    
    @pytest.fixture
    def session_coordinator(self, mock_components):
        """Create SessionCoordinator with mocked components."""
        audio_proc, video_mgr, metrics = mock_components
        
        coordinator = SessionCoordinator(
            width=640,
            height=480,
            audio_processor=audio_proc,
            video_manager=video_mgr,
            metrics_analyzer=metrics,
            enable_incremental=True
        )
        
        yield coordinator
        coordinator.close()
    
    def test_initialization(self, session_coordinator):
        """Test SessionCoordinator initializes correctly."""
        assert session_coordinator.width == 640
        assert session_coordinator.height == 480
        assert session_coordinator.enable_incremental is True
        assert session_coordinator.streaming_active is True
    
    def test_add_frame_delegates_to_video_manager(self, session_coordinator):
        """Test add_frame delegates to VideoBufferManager."""
        session_coordinator.video_manager.add_frame.return_value = True
        
        result = session_coordinator.add_frame(b'frame_data')
        
        assert result is True
        session_coordinator.video_manager.add_frame.assert_called_once_with(b'frame_data')
    
    def test_add_audio_delegates_to_audio_processor(self, session_coordinator):
        """Test add_audio delegates to AudioProcessor."""
        session_coordinator.add_audio(b'audio_data')
        
        session_coordinator.audio_processor.add_audio.assert_called_once_with(b'audio_data')
    
    @pytest.mark.asyncio
    async def test_process_incremental(self, session_coordinator):
        """Test incremental processing."""
        result = await session_coordinator.process_incremental()
        
        assert result.status in ["processing", "analyzing"]
        assert result.timestamp > 0
        
        # Verify components were called
        session_coordinator.audio_processor.process_audio.assert_called_once()
        session_coordinator.metrics_analyzer.get_metrics.assert_called_once()
    
    def test_end_stream(self, session_coordinator):
        """Test ending stream."""
        mock_path = Path("/fake/video.avi")
        session_coordinator.video_manager.finalize_video.return_value = mock_path
        
        video_path = session_coordinator.end_stream()
        
        assert video_path == mock_path
        assert session_coordinator.streaming_active is False
        session_coordinator.video_manager.finalize_video.assert_called_once()


# ============================================================================
# SessionFactory Tests
# ============================================================================

class TestSessionFactory:
    """Tests for SessionFactory."""
    
    def test_create_default(self):
        """Test creating default session."""
        session = SessionFactory.create_default(width=640, height=480)
        
        assert session.width == 640
        assert session.height == 480
        assert session.enable_incremental is True
        
        session.close()
    
    def test_create_testing(self):
        """Test creating testing session."""
        session = SessionFactory.create_testing(width=320, height=240)
        
        assert session.width == 320
        assert session.height == 240
        
        session.close()
    
    def test_create_production(self):
        """Test creating production session."""
        session = SessionFactory.create_production(width=1280, height=720)
        
        assert session.width == 1280
        assert session.height == 720
        
        session.close()


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for components working together."""
    
    @pytest.mark.asyncio
    async def test_full_session_workflow(self):
        """Test full session workflow with real components."""
        # Create session with real components
        session = SessionFactory.create_testing(width=640, height=480)
        
        try:
            # Add some mock data
            mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            with patch('api.websockets.video_buffer_manager.decode_frame_data') as mock_decode:
                mock_decode.return_value = mock_frame
                
                # Add frames
                for _ in range(10):
                    session.add_frame(b'frame')
                
                # Add audio (1 second of silence)
                audio_data = np.zeros(16000, dtype=np.int16).tobytes()
                session.add_audio(audio_data)
                
                # Process incrementally
                result = await session.process_incremental()
                
                assert result.frames_processed == 10
                assert result.status == "processing"
                
                # End stream
                # Mock the video creation
                with patch.object(session.video_manager, 'finalize_video') as mock_finalize:
                    mock_finalize.return_value = Path("/fake/video.avi")
                    video_path = session.end_stream()
                    
                    assert video_path is not None
                    assert session.streaming_active is False
        
        finally:
            session.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
