"""Image compression helpers focused on ROI-aware downscaling."""

from typing import List, Tuple, Optional

import cv2
import numpy as np


def apply_smart_compression(
    frame: np.ndarray,
    face_locations: Optional[List[Tuple[int, int, int, int]]] = None,
    downscale_full: float = 0.5,
    downscale_bg: float = 0.25,
) -> np.ndarray:
    """Compress frame giving priority to facial regions.

    Args:
        frame: BGR image array.
        face_locations: List of (x1, y1, x2, y2) bounding boxes. If None or
            empty, the whole frame is uniformly downscaled by *downscale_full*.
        downscale_full: Factor used when no faces are detected.
        downscale_bg: Factor applied to non-face regions when faces are present.

    Returns:
        Compressed BGR image with higher quality around faces.
    """

    if frame is None:
        return frame

    if not face_locations:
        # Uniform downscale when no faces
        return cv2.resize(frame, (0, 0), fx=downscale_full, fy=downscale_full)

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for x1, y1, x2, y2 in face_locations:
        padding = int(max(x2 - x1, y2 - y1) * 0.2)
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        mask[y1:y2, x1:x2] = 255

    high_quality = cv2.bitwise_and(frame, frame, mask=mask)
    bg_mask = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(frame, frame, mask=bg_mask)

    bg_small = cv2.resize(background, (0, 0), fx=downscale_bg, fy=downscale_bg)
    bg_resized = cv2.resize(bg_small, (frame.shape[1], frame.shape[0]))

    return cv2.add(high_quality, cv2.bitwise_and(bg_resized, bg_resized, mask=bg_mask))
