"""Video processing package.

This package consolidates functionality that previously lived in
`video_processor.*`.  External code can now simply do ``from video import ...``.
"""

from .metrics import build_metrics_response  # noqa: F401
from .realtime import decode_frame_data  # noqa: F401

__all__ = [
    "build_metrics_response",
    "decode_frame_data",
]
