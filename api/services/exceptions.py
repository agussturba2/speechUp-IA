# api/services/exceptions.py
"""Custom exceptions for the video processing service."""


class VideoProcessingError(Exception):
    """Base exception for video processing failures."""

    def __init__(self, message: str, video_id: str = "N/A"):
        self.message = message
        self.video_id = video_id
        super().__init__(f"[VideoID: {video_id}] {message}")


class InvalidVideoFileError(VideoProcessingError):
    """Raised when the video file is invalid, corrupted, or has no frames."""
    pass


class AudioAnalysisError(VideoProcessingError):
    """Raised when audio analysis fails for a specific reason."""
    pass
