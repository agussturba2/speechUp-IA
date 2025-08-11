# api/video_config.py
import os
from typing import List, Optional
from pydantic import  Field
from pydantic_settings import BaseSettings

class AppSettings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Provides validation and type-hinting for configuration.
    """
    # Application Metadata
    APP_NAME: str = "Oratory Feedback API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "API for video analysis and oratory feedback generation"

    # Logging Configuration
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    LOG_FILE: Optional[str] = Field(None, env="LOG_FILE")
    JSON_LOGS: bool = Field(False, env="JSON_LOGS")

    # CORS Configuration
    # Use a comma-separated string in your .env file:
    # ALLOWED_ORIGINS="http://localhost,http://localhost:3000"
    ALLOWED_ORIGINS: List[str] = Field(["*"], env="ALLOWED_ORIGINS")

    class Config:
        # This allows pydantic to read comma-separated strings into a list
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True


# Create a single, importable instance of the settings
settings = AppSettings()
