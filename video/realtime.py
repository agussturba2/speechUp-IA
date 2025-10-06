"""
Real-time video processing utilities for feedback de oratoria
"""

import cv2
import numpy as np

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

   