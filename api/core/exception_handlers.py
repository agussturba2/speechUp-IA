# api/core/exception_handlers.py
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("api.errors")


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler to catch unhandled exceptions.

    This ensures that any unexpected error returns a standardized 500
    response and is logged with a traceback.
    """
    logger.error(
        f"Unhandled exception for request: {request.method} {request.url.path}",
        exc_info=True,  # This adds the full traceback to the log
        extra={
            "path": str(request.url.path),
            "method": request.method,
            "error": str(exc)
        }
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred on the server."
        }
    )


def setup_exception_handlers(app: FastAPI) -> None:
    """
    Adds custom exception handlers to the FastAPI application.

    Args:
        app: The FastAPI application instance.
    """
    app.add_exception_handler(Exception, global_exception_handler)
