"""
Frame analysis component for incremental gesture and expression detection.

Provides scalable, maintainable video analysis using MediaPipe for:
- Facial expression detection
- Hand gesture tracking
- Body posture analysis
"""

import logging
import time
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import numpy as np
import cv2

from .models import Expression, Gesture, FrameAnalysisResult
from .config import config as incremental_config

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerConfig:
    """Configuration for frame analyzer."""
    enable_face_detection: bool = True
    enable_gesture_detection: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    max_faces: int = 1
    gesture_threshold: float = 0.3  # Movement threshold for gesture detection
    expression_smoothing: int = 3  # Frames to smooth expression detection


class FrameAnalyzer:
    """
    Analyzes video frames for gestures and expressions.
    
    Uses MediaPipe for efficient real-time analysis:
    - Face mesh for expression detection
    - Hand landmarks for gesture tracking
    - Pose landmarks for body language
    
    Designed for incremental processing with minimal state.
    """
    
    def __init__(self, config: Optional[AnalyzerConfig] = None):
        """
        Initialize frame analyzer.
        
        Args:
            config: Analyzer configuration (uses defaults if None)
        """
        self.config = config or AnalyzerConfig()
        
        # MediaPipe components (lazy initialization)
        self._face_mesh = None
        self._hands = None
        self._pose = None
        
        # State tracking for incremental processing
        self._last_hand_position = None
        self._frame_index = 0
        self._expression_history = []
        
        logger.error(
            f"FrameAnalyzer initialized: face={self.config.enable_face_detection}, "
            f"gestures={self.config.enable_gesture_detection}"
        )
    
    def _init_mediapipe_components(self):
        """Lazy initialization of MediaPipe components."""
        try:
            import mediapipe as mp
            
            if self.config.enable_face_detection and self._face_mesh is None:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    max_num_faces=self.config.max_faces,
                    refine_landmarks=True,
                    min_detection_confidence=self.config.min_detection_confidence,
                    min_tracking_confidence=self.config.min_tracking_confidence
                )
                logger.error("MediaPipe FaceMesh initialized successfully")
            
            if self.config.enable_gesture_detection and self._hands is None:
                self._hands = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=self.config.min_detection_confidence,
                    min_tracking_confidence=self.config.min_tracking_confidence
                )
                logger.error("MediaPipe Hands initialized successfully")
                
        except ImportError as e:
            logger.warning(f"MediaPipe not available: {e}")
            self.config.enable_face_detection = False
            self.config.enable_gesture_detection = False
    
    def analyze_frames(
        self,
        frames: List[np.ndarray],
        start_frame_index: int,
        fps: float = 30.0
    ) -> FrameAnalysisResult:
        """
        Analyze a batch of frames for gestures and expressions.
        
        Args:
            frames: List of frame arrays (BGR format)
            start_frame_index: Starting frame index for this batch
            fps: Frames per second for timestamp calculation
            
        Returns:
            FrameAnalysisResult with detected gestures and expressions
        """
        if not frames:
            return FrameAnalysisResult(
                frames_analyzed=0,
                frames_with_face=0,
                expressions=[],
                gestures=[],
                posture=[]
            )
        
        # Lazy init MediaPipe
        if self._face_mesh is None and self._hands is None:
            logger.error("Initializing MediaPipe components for first time...")
            self._init_mediapipe_components()
            logger.error(f"MediaPipe initialized: face_mesh={self._face_mesh is not None}, hands={self._hands is not None}")
        
        expressions: List[Expression] = []
        gestures: List[Gesture] = []
        frames_with_face = 0
        
        for i, frame in enumerate(frames):
            if frame is None or frame.size == 0:
                continue
            
            frame_index = start_frame_index + i
            timestamp = frame_index / fps
            
            # Detect face and expressions
            if self.config.enable_face_detection and self._face_mesh is not None:
                face_detected, frame_expressions = self._analyze_face(
                    frame, frame_index, timestamp
                )
                if face_detected:
                    frames_with_face += 1
                    expressions.extend(frame_expressions)
            
            # Detect hand gestures
            if self.config.enable_gesture_detection and self._hands is not None:
                frame_gestures = self._analyze_hands(frame, frame_index, timestamp)
                gestures.extend(frame_gestures)
            
            self._frame_index = frame_index
        
        logger.error(
            f"Analyzed {len(frames)} frames: {frames_with_face} with face, "
            f"{len(expressions)} expressions, {len(gestures)} gestures"
        )
        
        return FrameAnalysisResult(
            frames_analyzed=len(frames),
            frames_with_face=frames_with_face,
            expressions=expressions,
            gestures=gestures,
            posture=[]
        )
    
    def _analyze_face(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float
    ) -> tuple[bool, List[Expression]]:
        """
        Analyze face for expressions.
        
        Args:
            frame: Frame array (BGR)
            frame_index: Current frame index
            timestamp: Time in seconds
            
        Returns:
            Tuple of (face_detected, expressions)
        """
        expressions = []
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self._face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return False, expressions
            
            # Analyze facial landmarks for expressions
            for face_landmarks in results.multi_face_landmarks:
                detected_expressions = self._detect_expressions_from_landmarks(
                    face_landmarks,
                    frame_index,
                    timestamp
                )
                expressions.extend(detected_expressions)
            
            return True, expressions
            
        except Exception as e:
            logger.debug(f"Face analysis error: {e}")
            return False, expressions
    
    def _detect_expressions_from_landmarks(
        self,
        landmarks,
        frame_index: int,
        timestamp: float
    ) -> List[Expression]:
        """
        Detect expressions from facial landmarks.
        
        Uses simple heuristics based on landmark positions.
        Can be extended with more sophisticated models.
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_index: Current frame index
            timestamp: Time in seconds
            
        Returns:
            List of detected expressions
        """
        expressions = []
        
        try:
            # Extract key landmarks
            landmarks_list = landmarks.landmark
            
            # Mouth landmarks (upper lip: 13, lower lip: 14)
            mouth_top = landmarks_list[13].y
            mouth_bottom = landmarks_list[14].y
            mouth_open = abs(mouth_bottom - mouth_top)
            
            # Eye landmarks (approximation for blinks/expressions)
            left_eye_top = landmarks_list[159].y
            left_eye_bottom = landmarks_list[145].y
            eye_open = abs(left_eye_bottom - left_eye_top)
            
            # Mouth corners (for smile detection)
            left_corner = landmarks_list[61].y
            right_corner = landmarks_list[291].y
            mouth_center = landmarks_list[13].y
            smile_indicator = (left_corner + right_corner) / 2 - mouth_center
            
            # Detect smile
            if smile_indicator < -0.01 and mouth_open < 0.05:
                expressions.append(Expression(
                    type="smile",
                    confidence=min(0.9, abs(smile_indicator) * 50),
                    frame_index=frame_index,
                    timestamp=timestamp
                ))
            
            # Detect open mouth (speaking or surprise)
            elif mouth_open > 0.04:
                expressions.append(Expression(
                    type="speaking",
                    confidence=min(0.9, mouth_open * 15),
                    frame_index=frame_index,
                    timestamp=timestamp
                ))
            
            # Detect neutral (no strong expression)
            else:
                # Only report neutral occasionally to reduce noise
                if frame_index % 30 == 0:
                    expressions.append(Expression(
                        type="neutral",
                        confidence=0.7,
                        frame_index=frame_index,
                        timestamp=timestamp
                    ))
            
        except Exception as e:
            logger.debug(f"Expression detection error: {e}")
        
        return expressions
    
    def _analyze_hands(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp: float
    ) -> List[Gesture]:
        """
        Analyze hands for gestures.
        
        Args:
            frame: Frame array (BGR)
            frame_index: Current frame index
            timestamp: Time in seconds
            
        Returns:
            List of detected gestures
        """
        gestures = []
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self._hands.process(rgb_frame)
            
            if not results.multi_hand_landmarks:
                self._last_hand_position = None
                return gestures
            
            # Track hand movement
            for hand_landmarks in results.multi_hand_landmarks:
                gesture = self._detect_gesture_from_movement(
                    hand_landmarks,
                    frame_index,
                    timestamp
                )
                if gesture:
                    gestures.append(gesture)
            
        except Exception as e:
            logger.debug(f"Hand analysis error: {e}")
        
        return gestures
    
    def _detect_gesture_from_movement(
        self,
        landmarks,
        frame_index: int,
        timestamp: float
    ) -> Optional[Gesture]:
        """
        Detect gestures based on hand movement.
        
        Args:
            landmarks: MediaPipe hand landmarks
            frame_index: Current frame index
            timestamp: Time in seconds
            
        Returns:
            Detected gesture or None
        """
        try:
            # Get wrist position (landmark 0)
            wrist = landmarks.landmark[0]
            current_position = np.array([wrist.x, wrist.y, wrist.z])
            
            if self._last_hand_position is not None:
                # Calculate movement
                movement = np.linalg.norm(current_position - self._last_hand_position)
                
                # Detect significant movement as gesture
                if movement > self.config.gesture_threshold:
                    self._last_hand_position = current_position
                    
                    # Classify movement direction
                    delta = current_position - self._last_hand_position
                    
                    if abs(delta[0]) > abs(delta[1]):
                        gesture_type = "hand_move_horizontal"
                    else:
                        gesture_type = "hand_move_vertical"
                    
                    return Gesture(
                        type=gesture_type,
                        confidence=min(0.9, movement * 3),
                        frame_index=frame_index,
                        timestamp=timestamp
                    )
            
            self._last_hand_position = current_position
            
        except Exception as e:
            logger.debug(f"Gesture detection error: {e}")
        
        return None
    
    def reset(self) -> None:
        """Reset analyzer state."""
        self._last_hand_position = None
        self._frame_index = 0
        self._expression_history.clear()
        logger.debug("FrameAnalyzer reset")
    
    def close(self) -> None:
        """Clean up resources."""
        if self._face_mesh:
            self._face_mesh.close()
            self._face_mesh = None
        
        if self._hands:
            self._hands.close()
            self._hands = None
        
        if self._pose:
            self._pose.close()
            self._pose = None
        
        self.reset()
        logger.info("FrameAnalyzer closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
