"""Sub-package for video analysis algorithms (face, posture, etc.)."""

from .faces import process_frame  # noqa: F401

__all__ = [
    "process_frame",
]
