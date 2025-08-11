# api/main.py
"""FastAPI application entrypoint.

This file uses the application factory pattern to create and configure
the FastAPI application. Run with:

    uvicorn api.main:app --reload
"""
import logging
from fastapi import FastAPI

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
    setup_logging(settings)

    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
    )

    setup_middleware(app, settings)

    setup_exception_handlers(app)

    app.include_router(router)

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
