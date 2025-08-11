"""Video processing package.

This package consolidates functionality that previously lived in
`video_processor.*`.  External code can now simply do ``from video import ...``.
"""

from .extract_frames import extract_frames  # noqa: F401
from .metrics import build_metrics_response  # noqa: F401
from .realtime import decode_frame_data, process_realtime_buffer  # noqa: F401

__all__ = [
    "extract_frames",
    "build_metrics_response",
    "decode_frame_data",
    "process_realtime_buffer",
]
