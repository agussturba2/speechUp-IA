# /analysis/frame_analyzer.py

"""
Processes a buffer of frames to detect faces and poses.
"""
import os
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Tuple

from utils.logging import get_logger
from utils.cache import get_cache
from utils.hashing import compute_buffer_id
from video.video_config import HASH_RESIZE_DIM, CACHE_EXPIRATION_SECONDS, WORKER_CONFIG

# Assuming this function exists and analyzes a single frame
from video.analysis.faces import process_frame as analyze_single_frame

logger = get_logger(__name__)
results_cache = get_cache("results")


def _determine_worker_count(frame_count: int) -> Tuple[int, int]:
    """Calculates optimal worker count and batch size."""
    cpu_count = os.cpu_count() or 4
    mem_gb = psutil.virtual_memory().available / (1024 ** 3)

    # Limit workers by available memory to prevent thrashing
    max_workers = int(min(cpu_count, mem_gb / WORKER_CONFIG['ram_per_thread_gb']))
    batch_size = max(1, frame_count // max_workers if max_workers > 0 else frame_count)

    return max_workers, batch_size


def process_buffer(
        frames: List,
        indices: List[int],
        face_detector,
        pose_detector
) -> Tuple[Dict[str, Any], int, int]:
    """
    Analyzes a buffer of frames for faces and poses, using a thread pool.

    This function checks a cache for pre-computed results for the buffer.
    If not found, it distributes frame analysis across multiple threads.

    Args:
        frames: A list of frames to analyze.
        indices: The original frame indices corresponding to the frames.
        face_detector: An initialized MediaPipe FaceDetection model.
        pose_detector: An initialized MediaPipe Pose model.

    Returns:
        A tuple containing:
        - A dictionary with detailed feedback.
        - The total number of faces detected in the buffer.
        - The total number of good postures detected in the buffer.
    """
    buffer_id = compute_buffer_id(frames, HASH_RESIZE_DIM)
    if buffer_id and (cached := results_cache.get(f"buffer_{buffer_id}")):
        logger.debug(f"Cache hit for buffer ID: {buffer_id}")
        return cached

    logger.debug(f"Cache miss for buffer ID: {buffer_id}. Processing {len(frames)} frames.")
    max_workers, batch_size = _determine_worker_count(len(frames))

    faces_detected, posture_good = 0, 0
    feedback_texts = [None] * len(frames)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(analyze_single_frame, frame, face_detector, pose_detector): i
            for i, frame in enumerate(frames)
        }
        for future in futures:
            idx = futures[future]
            try:
                face, posture, feedback = future.result()
                if face: faces_detected += 1
                if posture: posture_good += 1
                feedback_texts[idx] = feedback
            except Exception as e:
                logger.warning(f"Error processing frame at index {indices[idx]}: {e}")
                feedback_texts[idx] = f"Error: {e}"

    summary = {
        "frames": indices,
        "faces_detected": faces_detected,
        "good_postures": posture_good,
        "feedback_text": feedback_texts
    }

    if buffer_id:
        results_cache.set(
            f"buffer_{buffer_id}",
            (summary, faces_detected, posture_good),
            expire=CACHE_EXPIRATION_SECONDS
        )

    gc.collect()
    return summary, faces_detected, posture_good
