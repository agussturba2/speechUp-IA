"""
Frame extraction utilities with adaptive and static sampling modes.
Refactored for clarity, testability and performance.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def get_video_metadata(video_path: str) -> Tuple[int, float, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"No se pudo abrir el video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    cap.release()
    return frame_count, fps, duration


def merge_overlapping_segments(segments: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not segments:
        return []
    segments.sort()
    merged = [segments[0]]
    for current in segments[1:]:
        last = merged[-1]
        if current[0] <= last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged


def detect_motion_segments(video_path: str, fps: float, frame_count: int, threshold=0.05) -> List[Tuple[int, int]]:
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    segments = []
    in_motion = False
    frame_idx = 0
    step = max(1, int(fps / 2))

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (80, 80))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            score = float(np.mean(diff) / 255.0)

            if score > threshold and not in_motion:
                start = max(0, frame_idx - int(fps * 0.5))
                in_motion = True
            elif score <= threshold * 0.7 and in_motion:
                end = min(frame_count - 1, frame_idx + int(fps * 0.5))
                segments.append((start, end))
                in_motion = False

        prev_frame = gray
        frame_idx += step

    if in_motion:
        segments.append((start, frame_count - 1))

    cap.release()
    return merge_overlapping_segments(segments)


def buffered_frame_generator(
    video_path: str,
    width: int,
    height: int,
    buffer_size: int,
    fps_sample: int = 1
):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25

    sampling_rate = max(1, int(fps / fps_sample))

    buffer, indices = [], []
    frame_count = 0
    success, frame = cap.read()

    while success:
        if frame_count % sampling_rate == 0:
            resized = cv2.resize(frame, (width, height))
            buffer.append(resized)
            indices.append(frame_count)

            if len(buffer) >= buffer_size:
                yield buffer, indices
                buffer, indices = [], []

        frame_count += 1
        success, frame = cap.read()

    if buffer:
        yield buffer, indices
    cap.release()


def extract_frames_adaptive(
    video_path: str,
    buffer_seconds: int,
    width: int,
    height: int,
    fps_sample: int = 2,
    motion_threshold: float = 0.05,
    quality_boost: bool = True
) -> Tuple[List[List], List[List[int]]]:
    frame_count, fps, _ = get_video_metadata(video_path)
    segments = detect_motion_segments(video_path, fps, frame_count, motion_threshold) if quality_boost else [(0, frame_count - 1)]

    cap = cv2.VideoCapture(video_path)

    high_sample = min(fps, fps_sample * 2)
    low_sample = max(1, fps_sample // 2)
    avg_sample = (high_sample + low_sample) / 2
    buffer_size = int((fps / max(1, int(fps / avg_sample))) * buffer_seconds)

    all_buffers, all_indices = [], []
    buffer, indices = [], []
    frame_idx = 0
    ret, frame = cap.read()

    while ret:
        in_segment = any(start <= frame_idx <= end for start, end in segments)
        sample_rate = max(1, int(fps / (high_sample if in_segment else low_sample)))

        if frame_idx % sample_rate == 0:
            resized = cv2.resize(frame, (width, height))
            buffer.append(resized)
            indices.append(frame_idx)

            if len(buffer) >= buffer_size:
                all_buffers.append(buffer)
                all_indices.append(indices)
                buffer, indices = [], []

        frame_idx += 1
        ret, frame = cap.read()

    if buffer:
        all_buffers.append(buffer)
        all_indices.append(indices)

    cap.release()
    return all_buffers, all_indices


def extract_frames(
    video_path: str,
    buffer_seconds: int,
    width: int,
    height: int,
    fps_sample: int = 2,
    adaptive: bool = False,
    quality_boost: bool = False
) -> Tuple[List[List], List[List[int]]]:
    if adaptive:
        return extract_frames_adaptive(
            video_path,
            buffer_seconds,
            width,
            height,
            fps_sample,
            quality_boost=quality_boost
        )

    frame_count, fps, _ = get_video_metadata(video_path)
    buffer_size = int((fps / max(1, int(fps / fps_sample))) * buffer_seconds)

    all_buffers, all_indices = [], []
    for buffer, indices in buffered_frame_generator(
        video_path, width, height, buffer_size, fps_sample
    ):
        all_buffers.append(buffer)
        all_indices.append(indices)

    return all_buffers, all_indices
