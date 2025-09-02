"""
Real-time video processing utilities for feedback de oratoria
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Dict, Any, Tuple

def decode_frame_data(frame_data: bytes, target_width: int = 160, target_height: int = 160) -> np.ndarray:
    """
    Decode binary frame data received from WebSocket and resize
    
    Args:
        frame_data: Binary frame data
        target_width: Width to resize to
        target_height: Height to resize to
        
    Returns:
        np.ndarray: Decoded and resized frame
    """
    # Convert binary data to numpy array
    nparr = np.frombuffer(frame_data, np.uint8)
    
    # Decode the image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Resize the frame
    if frame is not None:
        frame = cv2.resize(frame, (target_width, target_height))
    
    return frame

def process_realtime_frame(frame: np.ndarray, face_detector, pose_detector, mp_pose) -> Tuple[bool, bool, str]:
    """
    Process a single frame with pre-loaded models in real-time with GPU acceleration
    
    Args:
        frame: Frame to process
        face_detector: MediaPipe face detection model
        pose_detector: MediaPipe pose detection model
        mp_pose: MediaPipe pose module
    
    Returns:
        Tuple: (Face detected, Good posture, Feedback text)
    """
    # Convert to RGB for MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face
    face_result = face_detector.process(rgb)
    face_detected = bool(face_result.detections)
    posture_good = False
    
    if face_detected:
        feedback = "Rostro detectado, buena presencia."
        
        # Analyze posture
        pose_result = pose_detector.process(rgb)
        if pose_result.pose_landmarks:
            l_shoulder = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            r_shoulder = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            nose = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            
            # Check if head is centered
            if abs((l_shoulder.x + r_shoulder.x)/2 - nose.x) < 0.08:
                posture_good = True
                feedback += " Postura erguida y centrada."
            else:
                feedback += " Mejora tu postura: centra tu cabeza."
        else:
            feedback += " No se pudo analizar la postura."
    else:
        feedback = "No se detecta rostro. Acércate a la cámara o mejora la iluminación."
    
    return face_detected, posture_good, feedback

def process_realtime_buffer(buffer: List[np.ndarray], face_detector, pose_detector, mp_pose) -> Dict[str, Any]:
    """
    Process a buffer of frames in real-time
    
    Args:
        buffer: List of frames
        face_detector: MediaPipe face detection model
        pose_detector: MediaPipe pose detection model
        mp_pose: MediaPipe pose module
    
    Returns:
        Dict: Real-time feedback and statistics
    """
    faces_detected = 0
    postures_good = 0
    feedback_texts = []
    
    for frame in buffer:
        face_detected, posture_good, feedback = process_realtime_frame(
            frame, face_detector, pose_detector, mp_pose)
        
        if face_detected:
            faces_detected += 1
        if posture_good:
            postures_good += 1
        
        feedback_texts.append(feedback)
    
    # Calculate statistics
    buffer_size = len(buffer)
    face_percentage = round(100 * faces_detected / buffer_size, 1) if buffer_size > 0 else 0
    posture_percentage = round(100 * postures_good / buffer_size, 1) if buffer_size > 0 else 0
    
    # Generate realtime feedback
    status = "Excelente" if face_percentage > 80 and posture_percentage > 80 else \
             "Bueno" if face_percentage > 60 and posture_percentage > 50 else \
             "Mejorable" if face_percentage > 30 else "Necesita ajustes"
    
    main_feedback = "Tu presencia es adecuada." if face_percentage > 70 else \
                   "Acércate más a la cámara para mejor análisis."
                   
    posture_feedback = "Mantén esa buena postura." if posture_percentage > 70 else \
                      "Mejora tu postura corporal, mantén la cabeza centrada."
    
    # Build response
    response = {
        "timestamp": np.datetime64('now').item().timestamp(),
        "frames_procesados": buffer_size,
        "rostros_detectados": faces_detected,
        "posturas_buenas": postures_good,
        "porcentaje_frames_con_rostro": face_percentage,
        "porcentaje_posturas_buenas": posture_percentage,
        "estado": status,
        "feedback_general": main_feedback + " " + posture_feedback,
        "feedback_detalle": feedback_texts[-3:] if len(feedback_texts) > 3 else feedback_texts
    }
    
    return response
