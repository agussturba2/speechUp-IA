# config.py
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    # --- Metadata ---
    APP_NAME: str = "Oratory Feedback API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "API for video analysis and oratory feedback generation"

    # --- Logging ---
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field(None, env="LOG_FILE")
    JSON_LOGS: bool = Field(False, env="JSON_LOGS")

    # --- CORS ---
    ALLOWED_ORIGINS: List[str] = Field(["*"], env="ALLOWED_ORIGINS")

    # --- SpeechUp ASR Config ---
    SPEECHUP_ASR_MAX_WINDOW_SEC: int = Field(60, env="SPEECHUP_ASR_MAX_WINDOW_SEC")
    SPEECHUP_DEBUG_ASR: bool = Field(False, env="SPEECHUP_DEBUG_ASR")
    SPEECHUP_INCLUDE_TRANSCRIPT: bool = Field(True, env="SPEECHUP_INCLUDE_TRANSCRIPT")
    SPEECHUP_TRANSCRIPT_PREVIEW_MAX: int = Field(1200, env="SPEECHUP_TRANSCRIPT_PREVIEW_MAX")
    SPEECHUP_ADVICE_TARGET_COUNT: int = Field(5, env="SPEECHUP_ADVICE_TARGET_COUNT")

    # --- Pipeline Config ---
    SPEECHUP_USE_AUDIO: bool = Field(True, env="SPEECHUP_USE_AUDIO")
    SPEECHUP_USE_ASR: bool = Field(True, env="SPEECHUP_USE_ASR")
    SPEECHUP_USE_PROSODY: bool = Field(True, env="SPEECHUP_USE_PROSODY")

    # --- Whisper Config ---
    WHISPER_DEVICE: str = Field("cpu", env="WHISPER_DEVICE")
    SPEECHUP_ASR_MODEL: str = Field("base", env="SPEECHUP_ASR_MODEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = AppSettings()
