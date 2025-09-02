"""Face & posture analysis per frame.

This module was extracted from the former ``video.inference`` to keep the
pipeline orchestrator slim and allow re-use by realtime and offline paths.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np
import mediapipe as mp

from utils.image_compress import apply_smart_compression
from utils.math import calculate_posture_score

mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose

# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def process_frame(
    frame: np.ndarray,
    face_detector: mp_face_detection.FaceDetection,
    pose_detector: mp_pose.Pose,
) -> Tuple[bool, bool, str]:
    """Analyse a single frame.

    Args:
        frame: BGR image.
        face_detector: Shared MediaPipe face detector.
        pose_detector: Shared MediaPipe pose detector.

    Returns:
        face_detected: Whether a face is present.
        posture_good: Whether posture is centred.
        feedback_text: Quick textual feedback in Spanish.
    """

    if frame is None:
        return False, False, "Error: Frame vacío o nulo."

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_result = face_detector.process(rgb)
    face_locations: List[Tuple[int, int, int, int]] = []

    if face_result.detections:
        face_detected = True
        feedback = "Rostro detectado, buena presencia."

        h, w = rgb.shape[:2]
        for detection in face_result.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * w)
            y1 = int(bbox.ymin * h)
            x2 = int((bbox.xmin + bbox.width) * w)
            y2 = int((bbox.ymin + bbox.height) * h)
            face_locations.append((x1, y1, x2, y2))

        compressed_rgb = apply_smart_compression(rgb, face_locations)
        pose_result = pose_detector.process(compressed_rgb)
        if pose_result.pose_landmarks:
            ls = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            rs = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            posture_good = calculate_posture_score(ls.x, rs.x, nose.x)
            feedback += (
                " Postura erguida y centrada." if posture_good else " Mejora tu postura: centra tu cabeza."
            )
        else:
            posture_good = False
            feedback += " No se pudo analizar la postura."
    else:
        face_detected = False
        posture_good = False
        feedback = "No se detecta rostro. Acércate a la cámara o mejora la iluminación."

    return face_detected, posture_good, feedback
