# /video/advice_generator.py

"""
Generates qualitative advice based on quantitative analysis results.
"""
from typing import List
from video.video_config import ADVICE_THRESHOLDS


class AdviceGenerator:
    """Encapsulates logic for creating feedback from analysis metrics."""

    def __init__(self, thresholds: dict = ADVICE_THRESHOLDS):
        self.thresholds = thresholds

    def _get_face_advice(self, ratio: float) -> str:
        if ratio < self.thresholds["face"]["poor"]:
            return "Your face was not visible for a significant portion of the video."
        if ratio < self.thresholds["face"]["good"]:
            return "Your face was visible most of the time. Good job."
        return "Excellent visual presence! Your face was consistently visible."

    def _get_posture_advice(self, ratio: float) -> str:
        if ratio < self.thresholds["posture"]["poor"]:
            return "Your posture could be improved significantly."
        if ratio < self.thresholds["posture"]["good"]:
            return "Your posture was generally good, with some room for improvement."
        return "Excellent posture throughout the video!"

    def generate_global_advice(
            self, total_faces: int, total_posture: int, total_frames: int
    ) -> str:
        """
        Creates a summary advice string from overall detection statistics.

        Args:
            total_faces: Total frames where a face was detected.
            total_posture: Total frames where good posture was detected.
            total_frames: Total number of frames analyzed.

        Returns:
            A consolidated, human-readable advice string.
        """
        if total_frames == 0:
            return "No data available to generate advice."

        face_ratio = total_faces / total_frames
        # Posture is only relevant if a person (face) is visible
        posture_ratio = total_posture / total_frames if total_frames > 0 else 0

        advice_parts = [
            self._get_face_advice(face_ratio),
            self._get_posture_advice(posture_ratio)
        ]

        if (face_ratio < self.thresholds["face"]["good"] and
                posture_ratio < self.thresholds["posture"]["good"]):
            advice_parts.append(
                "Consider practicing in front of a mirror to self-assess "
                "your visual presence and posture."
            )

        return " ".join(advice_parts)
