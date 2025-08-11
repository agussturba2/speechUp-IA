# api/core/middleware.py
import logging
import time
import os
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from config import AppSettings

logger = logging.getLogger("api.middleware")


async def log_requests_middleware(request: Request, call_next: Callable) -> Response:
    """
    Middleware to log incoming requests and their responses.

    It generates a unique request ID, logs the start and completion of each
    request, and adds a process time header.
    """
    request_id = os.urandom(8).hex()
    start_time = time.time()

    logger.info(
        f"Request started: {request.method} {request.url.path}",
        extra={"request_id": request_id, "method": request.method, "path": str(request.url.path)}
    )

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time-ms"] = f"{process_time:.2f}"

    logger.info(
        f"Request completed: {request.method} {request.url.path} {response.status_code}",
        extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "process_time_ms": round(process_time, 2)
        }
    )
    return response


def setup_middleware(app: FastAPI, settings: AppSettings) -> None:
    """
    Configures and adds all middleware to the FastAPI application.

    Args:
        app: The FastAPI application instance.
        settings: The application settings object.
    """
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request logging middleware
    app.middleware("http")(log_requests_middleware)
