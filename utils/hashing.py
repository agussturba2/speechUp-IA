# /utils/hashing.py

"""
Utility functions for generating consistent hashes for caching.
"""
import hashlib
import cv2
from typing import List, Optional, Tuple


def compute_buffer_id(
        frames: List,
        hash_dim: Tuple[int, int]
) -> Optional[str]:
    """
    Generates a unique ID for a buffer of frames.

    Uses the first and last frames, resized to a small dimension,
    to create a fast and reasonably unique hash.

    Args:
        frames: A list of frames (as numpy arrays).
        hash_dim: The (width, height) to resize frames to before hashing.

    Returns:
        A unique MD5 hash string, or None if the buffer is empty.
    """
    if not frames:
        return None
    try:
        first_frame_resized = cv2.resize(frames[0], hash_dim)
        last_frame_resized = cv2.resize(frames[-1], hash_dim)

        hasher = hashlib.md5()
        hasher.update(first_frame_resized.tobytes())
        hasher.update(last_frame_resized.tobytes())
        hasher.update(str(len(frames)).encode())
        return hasher.hexdigest()
    except Exception:
        # Could fail if frames are corrupted
        return None


def compute_task_id(
        buffers: List[List],
        confidence: float,
        fps_sample: Optional[int],
        hash_dim: Tuple[int, int]
) -> Optional[str]:
    """
    Generates a unique ID for an entire analysis task.

    This hash considers the video content (first/last frames) and the
    key analysis parameters to ensure that a cached result is only
    reused if the task is identical.

    Args:
        buffers: A list of all frame buffers.
        confidence: The detection confidence used for the analysis.
        fps_sample: The sampling rate used.
        hash_dim: The (width, height) for resizing frames before hashing.

    Returns:
        A unique MD5 hash string, or None if buffers are empty.
    """
    if not buffers or not buffers[0]:
        return None
    try:
        first_frame = cv2.resize(buffers[0][0], hash_dim)
        last_frame = cv2.resize(buffers[-1][-1], hash_dim)

        meta_info = (
            f"{confidence}_{fps_sample}_{len(buffers)}_"
            f"{sum(len(b) for b in buffers)}"
        )

        hasher = hashlib.md5()
        hasher.update(first_frame.tobytes())
        hasher.update(last_frame.tobytes())
        hasher.update(meta_info.encode())
        return hasher.hexdigest()
    except Exception:
        return None
