"""
WebSocket handlers for real-time processing and feedback.

This module provides WebSocket handlers for real-time video and audio analysis.
"""

from .video import handle_realtime_feedback

__all__ = [
    "handle_realtime_feedback",
]
