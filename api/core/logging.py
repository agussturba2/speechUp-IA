# api/core/logging.py
import logging
import sys
from typing import Dict

from config import AppSettings


def setup_logging(settings: AppSettings) -> None:
    """
    Configures the application's logging system based on settings.

    This function sets the log level, format (standard or JSON), and handlers
    (console and optional file).

    Args:
        settings: The application settings object.
    """
    log_level_map: Dict[str, int] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    log_level = log_level_map.get(settings.LOG_LEVEL.upper(), logging.INFO)

    # In a real project, you would use a library like 'python-json-logger'
    log_format = (
        '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}'
        if settings.JSON_LOGS
        else "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    handlers = [logging.StreamHandler(sys.stdout)]
    if settings.LOG_FILE:
        handlers.append(logging.FileHandler(settings.LOG_FILE))

    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
    )

    # Set log levels for key modules
    for module in ["api", "video", "audio", "utils"]:
        logging.getLogger(module).setLevel(log_level)

    logging.getLogger(__name__).info(
        f"Logging configured with level: {settings.LOG_LEVEL}, JSON: {settings.JSON_LOGS}"
    )
