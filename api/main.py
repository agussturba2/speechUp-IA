# api/main.py
"""FastAPI application entrypoint.

This file uses the application factory pattern to create and configure
the FastAPI application. Run with:

    uvicorn api.main:app --reload
"""

import os
import logging
from fastapi import FastAPI

logger = logging.getLogger(__name__)

def _flag(name: str, default_on: bool = True) -> bool:
    """
    Read feature flag from env with safe defaults and cast to boolean.
    default_on=True -> if env var missing, treat as enabled.
    Truthy values accepted: 1, true, yes, on (case-insensitive).
    """
    raw = os.getenv(name)
    if raw is None:
        return default_on
    return raw.strip().lower() in ("1", "true", "yes", "on")

def _ensure_default_flags_on() -> None:
    """
    If flags are missing, set them to '1' so downstream modules that read os.environ
    will also see them enabled by default.
    """
    for key in ("SPEECHUP_USE_AUDIO", "SPEECHUP_USE_ASR", "SPEECHUP_USE_PROSODY"):
        if os.getenv(key) is None:
            os.environ[key] = "1"

from config import settings
from api.core.logging import setup_logging
from api.core.middleware import setup_middleware
from api.core.exception_handlers import setup_exception_handlers
from api.routers import router


def create_app() -> FastAPI:
    """
    Creates, configures, and returns a FastAPI application instance.

    This factory encapsulates the application's setup logic, making it
    reusable for testing and other deployment scenarios.

    Returns:
        The configured FastAPI application instance.
    """
    # Ensure default flags are ON before any other setup
    _ensure_default_flags_on()
    
    setup_logging(settings)

    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
    )

    setup_middleware(app, settings)

    setup_exception_handlers(app)

    app.include_router(router)

    # Log the effective feature flags (post-default/parse)
    use_audio   = _flag("SPEECHUP_USE_AUDIO", default_on=True)
    use_asr     = _flag("SPEECHUP_USE_ASR", default_on=True)
    use_prosody = _flag("SPEECHUP_USE_PROSODY", default_on=True)

    logger.info(
        "FEATURES -> USE_AUDIO=%s USE_ASR=%s USE_PROSODY=%s",
        use_audio, use_asr, use_prosody
    )

    @app.on_event("startup")
    async def startup_event():
        logger = logging.getLogger("api.main")
        logger.info("Application startup complete.")

    @app.on_event("shutdown")
    async def shutdown_event():
        logger = logging.getLogger("api.main")
        logger.info("Application shutting down.")

    return app


# Create the application instance for the Uvicorn server
app = create_app()
