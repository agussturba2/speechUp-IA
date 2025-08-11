# /pipeline.py

"""
Main video analysis pipeline orchestrator.
"""
from typing import Dict, Any
import cv2

from utils.logging import get_logger, log_execution_time
from utils.cache import get_cache
from utils.gpu import setup_gpu
from utils.hashing import compute_task_id
from video.video_config import QUALITY_PROFILES, CACHE_EXPIRATION_SECONDS, HASH_RESIZE_DIM
from video import extract_frames
from video.advice_generator import AdviceGenerator
from video.analysis.model_manager import ModelManager
from video.analysis.frame_analyzer import process_buffer

# --- Initial Setup ---
logger = get_logger(__name__)
setup_gpu()
results_cache = get_cache("results")


def get_adaptive_fps(duration_sec: float, base_fps_sample: int) -> int:
    """Adjusts FPS sampling rate for longer videos to speed up processing."""
    if duration_sec > 900:  # 15 minutes
        return max(base_fps_sample, 3)
    if duration_sec > 300:  # 5 minutes
        return max(base_fps_sample, 2)
    return base_fps_sample


@log_execution_time(logger)
def run_analysis_pipeline(
        video_path: str,
        quality_mode: str = "balanced"
) -> Dict[str, Any]:
    """
    Executes the full video analysis pipeline.

    This function orchestrates the major steps:
    1.  Configuration loading.
    2.  Video metadata extraction.
    3.  Task-level caching check.
    4.  Frame extraction.
    5.  Model initialization.
    6.  Concurrent buffer processing.
    7.  Result aggregation and final summary generation.

    Args:
        video_path: The path to the video file.
        quality_mode: The desired quality profile ("speed", "balanced", "quality").

    Returns:
        A dictionary containing a summary and detailed feedback.
    """
    props = QUALITY_PROFILES.get(quality_mode, QUALITY_PROFILES["balanced"])

    # 1. Get video metadata
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = frame_count / fps if fps > 0 else 0
    cap.release()

    effective_fps = get_adaptive_fps(duration_sec, 3)

    # 2. Extract frames into buffers
    # This now uses the assumed refactored `extract_frames` function
    all_buffers, all_indices = extract_frames(
        video_path, props["buffer_seconds"], props["width"],
        props["height"], effective_fps, True, quality_mode == "quality"
    )

    if not all_buffers or not any(all_buffers):
        logger.warning("No frames were extracted from the video.")
        return {"summary": "Video could not be processed.", "detailed_feedback": []}

    # 3. Check task-level cache
    task_id = compute_task_id(all_buffers, props["confidence"], effective_fps, HASH_RESIZE_DIM)
    if task_id and (cached := results_cache.get(f"task_{task_id}")):
        logger.info(f"Full task cache hit for ID: {task_id}. Returning cached results.")
        return cached

    # 4. Process all buffers
    feedbacks, total_faces, total_posture = [], 0, 0
    with ModelManager(props["confidence"], props["complexity"], props["smooth"]) as models:
        for buffer, idx in zip(all_buffers, all_indices):
            result, faces, posture = process_buffer(buffer, idx, models.face_detector, models.pose_detector)
            feedbacks.append(result)
            total_faces += faces
            total_posture += posture

    # 5. Generate final summary
    total_frames = sum(len(buf) for buf in all_buffers)
    advice_gen = AdviceGenerator()
    final_result = {
        "summary": {
            "total_frames_analyzed": total_frames,
            "total_faces_detected": total_faces,
            "total_good_posture": total_posture,
            "face_detection_percentage": round(100 * total_faces / total_frames, 2) if total_frames else 0,
            "good_posture_percentage": round(100 * total_posture / total_frames, 2) if total_frames else 0,
            "global_advice": advice_gen.generate_global_advice(total_faces, total_posture, total_frames)
        },
        "detailed_feedback": feedbacks
    }

    # 6. Cache the final result
    if task_id:
        results_cache.set(f"task_{task_id}", final_result, expire=CACHE_EXPIRATION_SECONDS)

    return final_result
