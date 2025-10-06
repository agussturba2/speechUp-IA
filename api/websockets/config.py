"""
Configuration for incremental WebSocket analysis.

Provides validated, environment-aware configuration for incremental processing.
"""

import os
from typing import Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class IncrementalConfig(BaseSettings):
    """Configuration for incremental oratory analysis."""
    
    # --- Buffer Configuration ---
    frame_buffer_size: int = Field(
        default=30,
        ge=10,
        le=120,
        env="INCREMENTAL_FRAME_BUFFER_SIZE",
        description="Size of the rolling frame buffer"
    )
    
    max_all_frames_buffer: int = Field(
        default=3000,
        ge=300,
        le=10000,
        env="INCREMENTAL_MAX_FRAMES",
        description="Maximum frames to store (prevents OOM)"
    )
    
    max_audio_buffer_sec: int = Field(
        default=60,
        ge=10,
        le=300,
        env="INCREMENTAL_MAX_AUDIO_SEC",
        description="Maximum audio buffer in seconds"
    )
    
    # --- Processing Configuration ---
    processing_interval: int = Field(
        default=60,
        ge=10,
        le=300,
        env="INCREMENTAL_PROCESSING_INTERVAL",
        description="Process incrementally every N frames"
    )
    
    enable_incremental_processing: bool = Field(
        default=True,
        env="INCREMENTAL_ENABLE_PROCESSING",
        description="Enable real-time incremental analysis"
    )
    
    # --- Audio Processing ---
    audio_sample_rate: int = Field(
        default=16000,
        ge=8000,
        le=48000,
        env="AUDIO_SAMPLE_RATE"
    )
    
    audio_bytes_per_sample: int = Field(
        default=2,
        ge=1,
        le=4,
        env="AUDIO_BYTES_PER_SAMPLE"
    )
    
    audio_channels: int = Field(
        default=1,
        ge=1,
        le=2,
        env="AUDIO_CHANNELS"
    )
    
    audio_overlap_seconds: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        env="AUDIO_OVERLAP_SEC",
        description="Audio overlap between processing windows"
    )
    
    min_audio_for_transcription_sec: float = Field(
        default=1.0,
        ge=0.5,
        le=5.0,
        env="MIN_AUDIO_TRANSCRIPTION_SEC",
        description="Minimum audio duration for transcription"
    )
    
    # --- Video Configuration ---
    default_fps: float = Field(
        default=30.0,
        ge=10.0,
        le=120.0,
        env="DEFAULT_FPS"
    )
    
    default_width: int = Field(
        default=640,
        ge=320,
        le=1920,
        env="DEFAULT_WIDTH"
    )
    
    default_height: int = Field(
        default=480,
        ge=240,
        le=1080,
        env="DEFAULT_HEIGHT"
    )
    
    min_video_file_size: int = Field(
        default=100,
        ge=10,
        le=10000,
        env="MIN_VIDEO_FILE_SIZE",
        description="Minimum valid video file size in bytes"
    )
    
    # --- WebSocket Configuration ---
    inactivity_timeout_sec: float = Field(
        default=10.0,
        ge=5.0,
        le=300.0,
        env="WEBSOCKET_INACTIVITY_TIMEOUT",
        description="WebSocket inactivity timeout"
    )
    
    # --- VAD Configuration ---
    vad_mode: int = Field(
        default=2,
        ge=0,
        le=3,
        env="VAD_MODE",
        description="WebRTC VAD aggressiveness (0-3)"
    )
    
    speech_detection_threshold: float = Field(
        default=10.0,
        ge=0.0,
        le=100.0,
        env="SPEECH_DETECTION_THRESHOLD",
        description="Minimum speech percentage to consider speech detected"
    )
    
    # --- Thread Pool Configuration ---
    max_workers: int = Field(
        default=1,
        ge=1,
        le=4,
        env="INCREMENTAL_MAX_WORKERS",
        description="ThreadPoolExecutor max workers"
    )
    
    # --- Filler Detection ---
    filler_time_distribution_offset: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Offset for distributing fillers across audio"
    )
    
    default_filler_duration: float = Field(
        default=0.3,
        ge=0.1,
        le=1.0,
        description="Default duration for filler words in seconds"
    )
    
    # --- Metrics ---
    confidence_growth_target_sec: float = Field(
        default=30.0,
        ge=10.0,
        le=120.0,
        description="Target duration for confidence to reach 1.0"
    )
    
    recent_events_window: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of recent events to include in updates"
    )
    
    # Computed properties
    @property
    def audio_bytes_per_ms(self) -> int:
        """Calculate bytes per millisecond for audio."""
        return (self.audio_sample_rate * self.audio_bytes_per_sample) // 1000
    
    @property
    def audio_normalization_factor(self) -> float:
        """Normalization factor for 16-bit PCM audio."""
        return 32768.0
    
    @property
    def max_audio_buffer_bytes(self) -> int:
        """Calculate maximum audio buffer in bytes."""
        return self.audio_sample_rate * self.audio_bytes_per_sample * self.max_audio_buffer_sec
    
    @property
    def min_audio_for_transcription_bytes(self) -> int:
        """Calculate minimum audio bytes for transcription."""
        return int(
            self.audio_sample_rate * 
            self.audio_bytes_per_sample * 
            self.min_audio_for_transcription_sec
        )
    
    @property
    def audio_overlap_bytes(self) -> int:
        """Calculate audio overlap in bytes."""
        return int(
            self.audio_overlap_seconds * 
            self.audio_sample_rate * 
            self.audio_bytes_per_sample
        )
    
    # Validators
    @field_validator('default_width', 'default_height')
    @classmethod
    def validate_even_dimensions(cls, v: int) -> int:
        """Ensure video dimensions are even (required by most codecs)."""
        if v % 2 != 0:
            raise ValueError(f"Dimension must be even, got {v}")
        return v
    
    @field_validator('max_all_frames_buffer')
    @classmethod
    def validate_frame_buffer_size(cls, v: int, info) -> int:
        """Validate frame buffer size to prevent OOM."""
        # Assuming 640x480x3 bytes per frame
        estimated_mb = (v * 640 * 480 * 3) / (1024 * 1024)
        if estimated_mb > 2000:  # 2GB limit
            raise ValueError(
                f"Frame buffer too large ({estimated_mb:.0f}MB). "
                f"Risk of OOM. Max recommended: 3000 frames"
            )
        return v
    
    @field_validator('processing_interval')
    @classmethod
    def validate_processing_interval(cls, v: int, info) -> int:
        """Ensure processing interval is reasonable."""
        max_frames = info.data.get('max_all_frames_buffer', 3000)
        if v > max_frames / 2:
            raise ValueError(
                f"Processing interval ({v}) should be < max_frames/2 ({max_frames/2})"
            )
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra env vars


class ProductionConfig(IncrementalConfig):
    """Production-optimized configuration."""
    max_all_frames_buffer: int = 6000  # 200 seconds @ 30fps
    max_audio_buffer_sec: int = 120
    enable_incremental_processing: bool = True
    max_workers: int = 2


class DevelopmentConfig(IncrementalConfig):
    """Development configuration with lower limits."""
    max_all_frames_buffer: int = 900  # 30 seconds @ 30fps
    max_audio_buffer_sec: int = 30
    inactivity_timeout_sec: float = 5.0


class TestingConfig(IncrementalConfig):
    """Testing configuration with minimal resources."""
    max_all_frames_buffer: int = 300  # 10 seconds @ 30fps
    max_audio_buffer_sec: int = 10
    processing_interval: int = 30
    inactivity_timeout_sec: float = 2.0
    enable_incremental_processing: bool = True


def get_config(env: Optional[str] = None) -> IncrementalConfig:
    """
    Get configuration based on environment.
    
    Args:
        env: Environment name ('production', 'development', 'testing')
             If None, uses ENVIRONMENT env var or defaults to IncrementalConfig
    
    Returns:
        Configuration instance
    """
    if env is None:
        env = os.getenv("ENVIRONMENT", "development").lower()
    
    config_map = {
        "production": ProductionConfig,
        "prod": ProductionConfig,
        "development": DevelopmentConfig,
        "dev": DevelopmentConfig,
        "testing": TestingConfig,
        "test": TestingConfig,
    }
    
    config_class = config_map.get(env, IncrementalConfig)
    return config_class()


# Default configuration instance
config = get_config()
