# /analysis/model_manager.py

"""
Manages the lifecycle of MediaPipe models for face and pose detection.
"""
import mediapipe as mp
from utils.gpu import get_optimal_mediapipe_config


class ModelManager:
    """
    A context manager to handle the setup and teardown of MediaPipe models.

    This isolates MediaPipe dependencies and ensures resources are properly
    released.
    """

    def __init__(self, confidence: float, complexity: int, smooth: bool):
        """
        Initializes the ModelManager with desired model configurations.

        Args:
            confidence: Minimum detection confidence for both models.
            complexity: Model complexity for the pose model.
            smooth: Whether to smooth pose landmarks across frames.
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_pose = mp.solutions.pose

        # Get optimal settings (e.g., for GPU) and override with specifics
        mp_config = get_optimal_mediapipe_config(complexity)

        self.face_config = {
            "model_selection": 0,
            "min_detection_confidence": confidence
        }
        self.pose_config = {
            "static_image_mode": not smooth,
            "model_complexity": mp_config["model_complexity"],
            "smooth_landmarks": smooth,
            "enable_segmentation": mp_config["enable_segmentation"],
            "min_detection_confidence": mp_config["min_detection_confidence"]
        }

        self.face_detector = None
        self.pose_detector = None

    def __enter__(self):
        """Initializes and returns the face and pose detector models."""
        self.face_detector = self.mp_face_detection.FaceDetection(**self.face_config)
        self.pose_detector = self.mp_pose.Pose(**self.pose_config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes the models and releases resources."""
        if self.face_detector:
            self.face_detector.close()
        if self.pose_detector:
            self.pose_detector.close()
