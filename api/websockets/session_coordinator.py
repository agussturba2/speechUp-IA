"""
Session coordinator for incremental oratory analysis.

Orchestrates audio processing, video buffering, and metrics analysis.
"""

import logging
import time
from typing import Dict, Any, Optional
from pathlib import Path

from .audio_processor import AudioProcessor
from .video_buffer_manager import VideoBufferManager
from .metrics_analyzer import MetricsAnalyzer
from .frame_analyzer import FrameAnalyzer, AnalyzerConfig
from .models import IncrementalUpdate, AnalysisStatus, FrameAnalysisResult
from .config import config as incremental_config

logger = logging.getLogger(__name__)


class SessionCoordinator:
    """
    Coordinates incremental oratory analysis session.
    
    Orchestrates:
    - AudioProcessor for audio analysis
    - VideoBufferManager for frame management
    - MetricsAnalyzer for metric computation
    
    Provides high-level interface for incremental processing.
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        audio_processor: Optional[AudioProcessor] = None,
        video_manager: Optional[VideoBufferManager] = None,
        metrics_analyzer: Optional[MetricsAnalyzer] = None,
        frame_analyzer: Optional[FrameAnalyzer] = None,
        enable_incremental: bool = True
    ):
        """
        Initialize session coordinator.
        
        Args:
            width: Video frame width
            height: Video frame height
            audio_processor: AudioProcessor instance (creates default if None)
            video_manager: VideoBufferManager instance (creates default if None)
            metrics_analyzer: MetricsAnalyzer instance (creates default if None)
            enable_incremental: Enable real-time incremental processing
        """
        self.width = width
        self.height = height
        self.enable_incremental = enable_incremental
        
        # Initialize components (dependency injection)
        self.audio_processor = audio_processor or AudioProcessor()
        self.video_manager = video_manager or VideoBufferManager(width, height)
        self.metrics_analyzer = metrics_analyzer or MetricsAnalyzer()
        self.frame_analyzer = frame_analyzer or FrameAnalyzer()
        
        # Session state
        self.streaming_active = True
        self.analysis_in_progress = False
        self.last_analysis_time = 0.0
        self.start_time = time.time()
        
        # Tracking for incremental processing
        self.last_processed_frame_index = 0
        self.partial_results = []
        
        logger.error(
            f"SessionCoordinator initialized: {width}x{height}, "
            f"incremental={enable_incremental}, frame_analysis=enabled"
        )
    
    def add_frame(self, frame_data: bytes) -> bool:
        """
        Add video frame to the session.
        
        Args:
            frame_data: Raw frame data
            
        Returns:
            True if successful
        """
        return self.video_manager.add_frame(frame_data)
    
    def add_audio(self, audio_data: bytes) -> None:
        """
        Add audio data to the session.
        
        Args:
            audio_data: Raw audio data
        """
        self.audio_processor.add_audio(audio_data)
    
    async def process_incremental(self) -> IncrementalUpdate:
        """
        Process accumulated data incrementally.
        
        Returns:
            IncrementalUpdate with current metrics and status
        """
        if self.analysis_in_progress:
            return IncrementalUpdate(
                status=AnalysisStatus.ANALYZING,
                frames_processed=self.video_manager.frame_count,
                timestamp=time.time(),
                message="Analysis already in progress"
            )
        
        self.analysis_in_progress = True
        t0 = time.perf_counter()
        
        try:
            if not self.enable_incremental:
                # Simple status update without processing
                result = IncrementalUpdate(
                    status=AnalysisStatus.PROCESSING,
                    frames_processed=self.video_manager.frame_count,
                    buffer_size=len(self.video_manager.frame_buffer),
                    processing_time_sec=time.perf_counter() - t0,
                    timestamp=time.time()
                )
                self.partial_results.append(result)
                return result
            
            # Process audio
            audio_result = await self.audio_processor.process_audio()
            if audio_result:
                self.metrics_analyzer.update_from_audio(audio_result)
            
            # Process frames (lightweight check for now)
            frame_result = self._process_new_frames()
            if frame_result:
                self.metrics_analyzer.update_from_frames(frame_result)
            
            # Get computed metrics
            metrics = self.metrics_analyzer.get_metrics()
            confidence = self.metrics_analyzer.get_confidence()
            
            # Build incremental update
            result = IncrementalUpdate(
                status=AnalysisStatus.PROCESSING,
                frames_processed=self.video_manager.frame_count,
                buffer_size=len(self.video_manager.frame_buffer),
                processing_time_sec=time.perf_counter() - t0,
                timestamp=time.time(),
                incremental_metrics=metrics,
                session_duration=time.time() - self.start_time,
                confidence=confidence,
                recent_fillers=self.metrics_analyzer.get_recent_fillers(),
                recent_gestures=self.metrics_analyzer.get_recent_gestures()
            )
            
            self.partial_results.append(result)
            self.last_analysis_time = time.perf_counter()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in incremental processing: {e}", exc_info=True)
            return IncrementalUpdate(
                status=AnalysisStatus.ERROR,
                frames_processed=self.video_manager.frame_count,
                timestamp=time.time(),
                error=str(e),
                error_type=type(e).__name__
            )
        finally:
            self.analysis_in_progress = False
    
    def _process_new_frames(self) -> Optional[FrameAnalysisResult]:
        """
        Process new frames with real gesture and expression analysis.
        
        Returns:
            FrameAnalysisResult or None
        """
        all_frames = self.video_manager.get_all_frames()
        new_frames = all_frames[self.last_processed_frame_index:]
        
        if not new_frames:
            return None
        
        frames_count = len(new_frames)
        logger.error(f"Processing {frames_count} new frames with FrameAnalyzer")
        
        # Filter valid frames
        valid_frames = [
            frame for frame in new_frames
            if frame is not None and frame.size > 0
        ]
        
        if not valid_frames:
            self.last_processed_frame_index = len(all_frames)
            return None
        
        # Analyze frames for gestures and expressions
        try:
            fps = self.video_manager.fps or 30.0
            result = self.frame_analyzer.analyze_frames(
                valid_frames,
                start_frame_index=self.last_processed_frame_index,
                fps=fps
            )
            
            self.last_processed_frame_index = len(all_frames)
            
            logger.error(
                f"Frame analysis: {result.frames_with_face} faces, "
                f"{len(result.expressions)} expressions, {len(result.gestures)} gestures"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Frame analysis failed: {e}", exc_info=True)
            self.last_processed_frame_index = len(all_frames)
            
            # Return basic result on error
            return FrameAnalysisResult(
                frames_analyzed=frames_count,
                frames_with_face=len(valid_frames),
                expressions=[],
                gestures=[],
                posture=[]
            )
    
    def end_stream(self) -> Optional[Path]:
        """
        End streaming and finalize video.
        
        Returns:
            Path to video file or None
        """
        logger.info("Ending stream")
        self.streaming_active = False
        
        # Finalize video
        video_path = self.video_manager.finalize_video()
        
        if video_path:
            logger.info(f"Video finalized: {video_path}")
        else:
            logger.warning("Failed to finalize video")
        
        return video_path
    
    def get_video_path(self) -> Optional[Path]:
        """Get path to video file."""
        return self.video_manager.get_video_path()
    
    def get_frame_count(self) -> int:
        """Get total frame count."""
        return self.video_manager.frame_count
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of session state.
        
        Returns:
            Dictionary with session summary
        """
        return {
            "session_duration": time.time() - self.start_time,
            "streaming_active": self.streaming_active,
            "frames_received": self.video_manager.frame_count,
            "audio_processed_sec": self.metrics_analyzer.state.audio_processed_sec,
            "incremental_steps": len(self.partial_results),
            "metrics": self.metrics_analyzer.get_state_summary()
        }
    
    def reset(self) -> None:
        """Reset session state."""
        self.audio_processor.reset()
        self.video_manager.reset()
        self.metrics_analyzer.reset()
        self.frame_analyzer.reset()
        
        self.streaming_active = True
        self.analysis_in_progress = False
        self.last_analysis_time = 0.0
        self.start_time = time.time()
        self.last_processed_frame_index = 0
        self.partial_results = []
        
        logger.debug("SessionCoordinator reset")
    
    def close(self) -> None:
        """Clean up all resources."""
        logger.info("Closing SessionCoordinator")
        
        self.audio_processor.close()
        self.video_manager.close()
        self.metrics_analyzer.close()
        self.frame_analyzer.close()
        
        self.partial_results.clear()
        
        logger.info("SessionCoordinator closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False


class SessionFactory:
    """
    Factory for creating SessionCoordinator instances with different configurations.
    """
    
    @staticmethod
    def create_default(width: int, height: int) -> SessionCoordinator:
        """
        Create SessionCoordinator with default configuration.
        
        Args:
            width: Video width
            height: Video height
            
        Returns:
            SessionCoordinator instance
        """
        return SessionCoordinator(
            width=width,
            height=height,
            enable_incremental=True
        )
    
    @staticmethod
    def create_testing(width: int = 640, height: int = 480) -> SessionCoordinator:
        """
        Create SessionCoordinator for testing with minimal resources.
        
        Args:
            width: Video width
            height: Video height
            
        Returns:
            SessionCoordinator instance configured for testing
        """
        from .config import TestingConfig
        
        config = TestingConfig()
        
        audio_processor = AudioProcessor(
            sample_rate=config.audio_sample_rate,
            vad_mode=config.vad_mode
        )
        
        video_manager = VideoBufferManager(
            width=width,
            height=height,
            fps=config.default_fps,
            buffer_size=config.frame_buffer_size,
            max_frames=config.max_all_frames_buffer
        )
        
        metrics_analyzer = MetricsAnalyzer()
        frame_analyzer = FrameAnalyzer()
        
        return SessionCoordinator(
            width=width,
            height=height,
            audio_processor=audio_processor,
            video_manager=video_manager,
            metrics_analyzer=metrics_analyzer,
            frame_analyzer=frame_analyzer,
            enable_incremental=True
        )
    
    @staticmethod
    def create_production(width: int, height: int) -> SessionCoordinator:
        """
        Create SessionCoordinator for production with optimized settings.
        
        Args:
            width: Video width
            height: Video height
            
        Returns:
            SessionCoordinator instance configured for production
        """
        from .config import ProductionConfig
        
        config = ProductionConfig()
        
        audio_processor = AudioProcessor(
            sample_rate=config.audio_sample_rate,
            vad_mode=config.vad_mode
        )
        
        video_manager = VideoBufferManager(
            width=width,
            height=height,
            fps=config.default_fps,
            buffer_size=config.frame_buffer_size,
            max_frames=config.max_all_frames_buffer
        )
        
        metrics_analyzer = MetricsAnalyzer()
        frame_analyzer = FrameAnalyzer()
        
        return SessionCoordinator(
            width=width,
            height=height,
            audio_processor=audio_processor,
            video_manager=video_manager,
            metrics_analyzer=metrics_analyzer,
            frame_analyzer=frame_analyzer,
            enable_incremental=config.enable_incremental_processing
        )
