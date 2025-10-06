"""
Metrics analysis component for incremental oratory feedback.

Computes and tracks verbal and non-verbal metrics incrementally.
"""

import logging
import time
from typing import Dict, Any, List
from dataclasses import dataclass, field

from .models import AudioAnalysisResult, FrameAnalysisResult, IncrementalMetrics
from .config import config as incremental_config

logger = logging.getLogger(__name__)


@dataclass
class AccumulatedState:
    """State accumulated during incremental analysis."""
    
    # Verbal metrics
    word_count: int = 0
    filler_count: int = 0
    pause_count: int = 0
    speaking_time_sec: float = 0.0
    transcript_segments: List[str] = field(default_factory=list)
    filler_instances: List[Dict[str, Any]] = field(default_factory=list)
    
    # Non-verbal metrics
    gestures: List[Dict[str, Any]] = field(default_factory=list)
    expressions: List[Dict[str, Any]] = field(default_factory=list)
    posture_changes: List[Dict[str, Any]] = field(default_factory=list)
    
    # Processing stats
    frames_processed: int = 0
    audio_processed_sec: float = 0.0
    start_time: float = field(default_factory=time.time)
    
    def get_session_duration(self) -> float:
        """Get session duration in seconds."""
        return time.time() - self.start_time


class MetricsAnalyzer:
    """
    Analyzes and computes metrics incrementally from audio and video analysis.
    
    Responsibilities:
    - Accumulate analysis results
    - Compute verbal metrics (WPM, fillers per minute, pause rate)
    - Compute non-verbal metrics (gesture rate, expression variability)
    - Track confidence scores
    """
    
    def __init__(self):
        """Initialize metrics analyzer."""
        self.state = AccumulatedState()
        
        # Computed metrics cache
        self._metrics_cache: Optional[IncrementalMetrics] = None
        self._cache_dirty = True
        
        logger.info("MetricsAnalyzer initialized")
    
    def update_from_audio(self, audio_result: AudioAnalysisResult) -> None:
        """
        Update state from audio analysis result.
        
        Args:
            audio_result: Result from audio processing
        """
        # Update audio processed time
        self.state.audio_processed_sec += audio_result.duration_sec
        
        # Update word count
        self.state.word_count += len(audio_result.words)
        
        # Update filler count and instances
        self.state.filler_count += len(audio_result.fillers)
        for filler in audio_result.fillers:
            self.state.filler_instances.append(filler.model_dump())
        
        # Update pause count
        self.state.pause_count += len(audio_result.pauses)
        
        # Update speaking time (total time - pauses)
        if audio_result.speech_detected:
            pause_duration = sum(p.duration for p in audio_result.pauses)
            self.state.speaking_time_sec += (audio_result.duration_sec - pause_duration)
        
        # Mark cache as dirty
        self._cache_dirty = True
        
        logger.debug(
            f"Updated from audio: words={len(audio_result.words)}, "
            f"fillers={len(audio_result.fillers)}, "
            f"pauses={len(audio_result.pauses)}"
        )
    
    def update_from_frames(self, frame_result: FrameAnalysisResult) -> None:
        """
        Update state from frame analysis result.
        
        Args:
            frame_result: Result from frame processing
        """
        # Update frames processed
        self.state.frames_processed += frame_result.frames_analyzed
        
        # Update non-verbal metrics
        for expression in frame_result.expressions:
            self.state.expressions.append(expression.model_dump())
        
        for gesture in frame_result.gestures:
            self.state.gestures.append(gesture.model_dump())
        
        # Mark cache as dirty
        self._cache_dirty = True
        
        logger.debug(
            f"Updated from frames: analyzed={frame_result.frames_analyzed}, "
            f"expressions={len(frame_result.expressions)}, "
            f"gestures={len(frame_result.gestures)}"
        )
    
    def get_metrics(self) -> IncrementalMetrics:
        """
        Get computed incremental metrics.
        
        Returns:
            IncrementalMetrics with current calculated values
        """
        if not self._cache_dirty and self._metrics_cache is not None:
            return self._metrics_cache
        
        # Compute metrics
        wpm = self._compute_wpm()
        fillers_per_min = self._compute_fillers_per_min()
        gesture_rate = self._compute_gesture_rate()
        expression_variability = self._compute_expression_variability()
        
        # Create metrics
        self._metrics_cache = IncrementalMetrics(
            wpm=wpm,
            fillers_per_min=fillers_per_min,
            gesture_rate=gesture_rate,
            expression_variability=expression_variability
        )
        self._cache_dirty = False
        
        return self._metrics_cache
    
    def _compute_wpm(self) -> float:
        """Compute words per minute."""
        if self.state.speaking_time_sec <= 0:
            return 0.0
        
        minutes = self.state.speaking_time_sec / 60.0
        if minutes <= 0:
            return 0.0
        
        return self.state.word_count / minutes
    
    def _compute_fillers_per_min(self) -> float:
        """Compute fillers per minute."""
        if self.state.speaking_time_sec <= 0:
            return 0.0
        
        minutes = self.state.speaking_time_sec / 60.0
        if minutes <= 0:
            return 0.0
        
        return self.state.filler_count / minutes
    
    def _compute_gesture_rate(self) -> float:
        """Compute gestures per minute."""
        audio_time = max(0.1, self.state.audio_processed_sec)
        minutes = audio_time / 60.0
        
        return len(self.state.gestures) / minutes
    
    def _compute_expression_variability(self) -> float:
        """Compute expression variability score."""
        if not self.state.expressions:
            return 0.0
        
        # Count unique expression types
        expression_types = set(
            expr.get("type", "") 
            for expr in self.state.expressions
        )
        
        # Normalize by number of expressions (per 10 expressions)
        num_expressions = len(self.state.expressions)
        variability = len(expression_types) / max(1, num_expressions / 10)
        
        # Clamp to [0, 1]
        return min(1.0, variability)
    
    def get_confidence(self) -> float:
        """
        Get confidence score based on amount of data processed.
        
        Returns:
            Confidence score from 0.0 to 1.0
        """
        target_seconds = incremental_config.confidence_growth_target_sec
        confidence = min(1.0, self.state.audio_processed_sec / target_seconds)
        return confidence
    
    def get_recent_fillers(self, count: int = None) -> List[Dict[str, Any]]:
        """
        Get recent filler instances.
        
        Args:
            count: Number of recent fillers to return
            
        Returns:
            List of recent filler dicts
        """
        count = count or incremental_config.recent_events_window
        return self.state.filler_instances[-count:]
    
    def get_recent_gestures(self, count: int = None) -> List[Dict[str, Any]]:
        """
        Get recent gesture instances.
        
        Args:
            count: Number of recent gestures to return
            
        Returns:
            List of recent gesture dicts
        """
        count = count or incremental_config.recent_events_window
        return self.state.gestures[-count:]
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get summary of accumulated state.
        
        Returns:
            Dictionary with state summary
        """
        return {
            "word_count": self.state.word_count,
            "filler_count": self.state.filler_count,
            "pause_count": self.state.pause_count,
            "speaking_time_sec": self.state.speaking_time_sec,
            "frames_processed": self.state.frames_processed,
            "audio_processed_sec": self.state.audio_processed_sec,
            "session_duration": self.state.get_session_duration(),
            "gestures": len(self.state.gestures),
            "expressions": len(self.state.expressions)
        }
    
    def reset(self) -> None:
        """Reset analyzer state."""
        self.state = AccumulatedState()
        self._metrics_cache = None
        self._cache_dirty = True
        logger.debug("MetricsAnalyzer reset")
    
    def close(self) -> None:
        """Clean up resources."""
        self.reset()
        logger.info("MetricsAnalyzer closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
