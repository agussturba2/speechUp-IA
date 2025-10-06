"""
Pydantic models for WebSocket incremental analysis results.

Provides type-safe, validated data structures for analysis results.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


class AnalysisStatus(str, Enum):
    """Status of analysis operation."""
    CONNECTED = "connected"
    BUFFERING = "buffering"
    PROCESSING = "processing"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    ERROR = "error"
    STREAMING = "streaming"


class Word(BaseModel):
    """Transcribed word with metadata."""
    word: str = Field(..., min_length=1)
    index: int = Field(..., ge=0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    model_config = ConfigDict(frozen=True)


class FillerInstance(BaseModel):
    """Detected filler word instance."""
    type: str = Field(..., description="Filler word type (e.g., 'eh', 'um')")
    time: float = Field(..., ge=0.0, description="Time in seconds")
    duration: float = Field(default=0.3, ge=0.0, description="Duration in seconds")
    
    model_config = ConfigDict(frozen=True)


class PauseSegment(BaseModel):
    """Speech pause segment."""
    start_time: float = Field(..., ge=0.0)
    duration: float = Field(..., gt=0.0)
    
    model_config = ConfigDict(frozen=True)


class PauseAnalysis(BaseModel):
    """Pause analysis metrics."""
    speech_percent: float = Field(..., ge=0.0, le=100.0)
    avg_speech_segment: float = Field(..., ge=0.0)
    avg_pause_length: float = Field(..., ge=0.0)
    pause_frequency: float = Field(..., ge=0.0)
    
    model_config = ConfigDict(frozen=True)


class AudioAnalysisResult(BaseModel):
    """Result from audio buffer processing."""
    duration_sec: float = Field(..., ge=0.0)
    speech_detected: bool
    words: List[Word] = Field(default_factory=list)
    fillers: List[FillerInstance] = Field(default_factory=list)
    pauses: List[PauseSegment] = Field(default_factory=list)
    pause_analysis: Optional[PauseAnalysis] = None
    
    @field_validator('duration_sec')
    @classmethod
    def validate_duration(cls, v: float) -> float:
        if v > 3600:  # 1 hour max
            raise ValueError("Audio duration exceeds maximum (3600s)")
        return v
    
    model_config = ConfigDict(frozen=True)


class Expression(BaseModel):
    """Facial expression detection."""
    type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    frame_index: int = Field(..., ge=0)
    timestamp: float = Field(..., ge=0.0)
    
    model_config = ConfigDict(frozen=True)


class Gesture(BaseModel):
    """Body gesture detection."""
    type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    frame_index: int = Field(..., ge=0)
    timestamp: float = Field(..., ge=0.0)
    
    model_config = ConfigDict(frozen=True)


class FrameAnalysisResult(BaseModel):
    """Result from frame processing."""
    frames_analyzed: int = Field(..., ge=0)
    frames_with_face: int = Field(..., ge=0)
    expressions: List[Expression] = Field(default_factory=list)
    gestures: List[Gesture] = Field(default_factory=list)
    posture: List[Dict[str, Any]] = Field(default_factory=list)
    
    @field_validator('frames_with_face')
    @classmethod
    def validate_face_frames(cls, v: int, info) -> int:
        frames_analyzed = info.data.get('frames_analyzed', 0)
        if v > frames_analyzed:
            raise ValueError(f"frames_with_face ({v}) cannot exceed frames_analyzed ({frames_analyzed})")
        return v
    
    model_config = ConfigDict(frozen=True)


class IncrementalMetrics(BaseModel):
    """Metrics computed from incremental analysis."""
    wpm: float = Field(..., ge=0.0, le=500.0, description="Words per minute")
    fillers_per_min: float = Field(..., ge=0.0, description="Fillers per minute")
    gesture_rate: float = Field(..., ge=0.0, description="Gestures per minute")
    expression_variability: float = Field(..., ge=0.0, le=1.0, description="Expression variety score")
    
    @field_validator('wpm')
    @classmethod
    def validate_wpm(cls, v: float) -> float:
        if v > 0 and v < 50:
            # Warn about unusually slow speech
            pass
        elif v > 300:
            # Warn about unusually fast speech
            pass
        return v
    
    model_config = ConfigDict(frozen=True)


class IncrementalUpdate(BaseModel):
    """Incremental analysis update message."""
    status: AnalysisStatus
    frames_processed: int = Field(..., ge=0)
    buffer_size: Optional[int] = Field(None, ge=0)
    processing_time_sec: Optional[float] = Field(None, ge=0.0)
    timestamp: float
    incremental_metrics: Optional[IncrementalMetrics] = None
    session_duration: Optional[float] = Field(None, ge=0.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    recent_fillers: List[FillerInstance] = Field(default_factory=list)
    recent_gestures: List[Gesture] = Field(default_factory=list)
    message: Optional[str] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


class QualityMetrics(BaseModel):
    """Video/audio quality metrics."""
    frames_analyzed: int = Field(..., ge=0)
    audio_analyzed_sec: float = Field(..., ge=0.0)
    analysis_ms: int = Field(..., ge=0)
    audio_available: bool
    debug: Dict[str, Any] = Field(default_factory=dict)


class MediaInfo(BaseModel):
    """Media file information."""
    frames_total: int = Field(..., ge=0)
    frames_with_face: int = Field(..., ge=0)
    fps: float = Field(..., gt=0.0)
    duration_sec: float = Field(..., ge=0.0)


class Scores(BaseModel):
    """Analysis scores (0-100)."""
    fluency: int = Field(..., ge=0, le=100)
    clarity: int = Field(..., ge=0, le=100)
    pace: int = Field(..., ge=0, le=100)
    engagement: int = Field(..., ge=0, le=100)


class VerbalMetrics(BaseModel):
    """Verbal communication metrics."""
    wpm: float = Field(..., ge=0.0)
    fillers_per_min: float = Field(..., ge=0.0)
    filler_counts: Dict[str, int] = Field(default_factory=dict)
    transcript_len: Optional[int] = Field(None, ge=0)
    transcript_short: Optional[str] = None
    transcript_full: Optional[str] = None
    stt_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    articulation_rate_sps: Optional[float] = Field(None, ge=0.0)
    pronunciation_score: Optional[float] = Field(None, ge=0.0, le=1.0)


class Event(BaseModel):
    """Analysis event (filler, gesture, etc.)."""
    time: float = Field(..., ge=0.0)
    kind: str = Field(..., description="Event type: filler, gesture, pause, etc.")
    label: str = Field(..., description="Event label/name")
    duration: Optional[float] = Field(None, ge=0.0)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    source: Optional[str] = Field(None, description="incremental, pipeline, etc.")
    
    model_config = ConfigDict(frozen=True)


class AnalysisResult(BaseModel):
    """Complete analysis result structure."""
    status: AnalysisStatus
    quality: Optional[QualityMetrics] = None
    media: Optional[MediaInfo] = None
    scores: Optional[Scores] = None
    verbal: Optional[VerbalMetrics] = None
    events: List[Event] = Field(default_factory=list)
    timestamp: float
    error: Optional[str] = None
    error_type: Optional[str] = None
    message: Optional[str] = None
    
    @field_validator('events')
    @classmethod
    def sort_events(cls, v: List[Event]) -> List[Event]:
        """Sort events by time."""
        return sorted(v, key=lambda e: e.time)


class SessionConfig(BaseModel):
    """Configuration for an incremental session."""
    buffer_size: int = Field(default=30, ge=10, le=120)
    width: int = Field(default=640, ge=320, le=1920)
    height: int = Field(default=480, ge=240, le=1080)
    processing_interval: int = Field(default=60, ge=10, le=300, description="Process every N frames")
    max_frames_buffer: int = Field(default=3000, ge=300, le=10000)
    max_audio_buffer_sec: int = Field(default=60, ge=10, le=300)
    enable_incremental_processing: bool = Field(default=True)
    
    @field_validator('width', 'height')
    @classmethod
    def validate_dimensions(cls, v: int, info) -> int:
        if v % 2 != 0:
            raise ValueError(f"Dimension must be even number, got {v}")
        return v
    
    @property
    def max_audio_buffer_bytes(self) -> int:
        """Calculate max audio buffer in bytes."""
        # 16kHz, 16-bit PCM, mono
        return 16000 * 2 * self.max_audio_buffer_sec


# Legacy type aliases for backward compatibility
AudioAnalysisResultDict = Dict[str, Any]
FrameAnalysisResultDict = Dict[str, Any]
IncrementalMetricsDict = Dict[str, Any]
AnalysisResultDict = Dict[str, Any]
