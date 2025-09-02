# api/utils/file_handling.py
"""
Utility functions for handling temporary files, validation, and system checks.
"""
import logging
import shutil
from pathlib import Path
from typing import Tuple

import aiofiles
import cv2
from fastapi import UploadFile

from api.services.exceptions import InvalidVideoFileError

logger = logging.getLogger(__name__)


async def save_temp_video(destination_path: Path, video_file: UploadFile) -> None:
    """
    Asynchronously saves an uploaded video file to a temporary location.

    Args:
        destination_path: The Path object for the destination file.
        video_file: The uploaded file from FastAPI.
    """
    logger.info(f"Saving video to: {destination_path}")
    try:
        async with aiofiles.open(destination_path, "wb") as f:
            while chunk := await video_file.read(1024 * 1024):  # Read in 1MB chunks
                await f.write(chunk)
    except Exception as e:
        raise IOError(f"Failed to write video file to {destination_path}: {e}")


def validate_video_file(video_path: Path) -> None:
    """
    Validates a video file using OpenCV to ensure it's openable and has content.

    Args:
        video_path: The path to the video file.

    Raises:
        InvalidVideoFileError: If the file is invalid or empty.
    """
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise InvalidVideoFileError("Video file is empty or does not exist.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise InvalidVideoFileError("Invalid video format or corrupted file.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    if frame_count <= 0 or fps <= 0:
        raise InvalidVideoFileError("Video contains no frames or has zero duration.")
    logger.info(f"Video validated: {frame_count} frames at {fps:.2f} FPS.")


def check_ffmpeg_tools() -> Tuple[bool, bool]:
    """
    Checks if 'ffmpeg' and 'ffprobe' are available in the system's PATH.

    Returns:
        A tuple of booleans: (ffmpeg_available, ffprobe_available).
    """
    ffmpeg_available = shutil.which("ffmpeg") is not None
    ffprobe_available = shutil.which("ffprobe") is not None
    if not ffmpeg_available or not ffprobe_available:
        logger.warning("FFmpeg/FFprobe not found in PATH. Audio analysis will be skipped.")
    return ffmpeg_available, ffprobe_available


def cleanup_temp_files(*files: Path) -> None:
    """
    Safely removes one or more temporary files.

    Args:
        *files: A variable number of Path objects to remove.
    """
    for file_path in files:
        if file_path and file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"Removed temp file: {file_path}")
            except OSError as e:
                logger.warning(f"Could not remove temp file {file_path}: {e}")
