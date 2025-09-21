"""
WebSocket handlers for real-time processing and feedback.

This module provides WebSocket handlers for real-time video and audio analysis.
"""

from .oratory import handle_oratory_feedback
from .incremental import handle_incremental_oratory_feedback

__all__ = [
    "handle_oratory_feedback",
    "handle_incremental_oratory_feedback",
]
