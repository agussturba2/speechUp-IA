"""
WebSocket handlers for real-time video processing and feedback.
"""

import time
from collections import deque
import tempfile
import os
import numpy as np
import soundfile as sf
import mediapipe as mp
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Any

# Use new OratoryAnalyzer instead of legacy AudioAnalyzer
from audio import OratoryAnalyzer
from video.realtime import decode_frame_data, process_realtime_buffer

class RealtimeSession:
    """
    Session handler for real-time video analysis.
    
    Manages frame buffers, models, and analysis state for a WebSocket connection.
    """
    
    def __init__(
        self,
        buffer_size: int = 10,
        width: int = 160,
        height: int = 160,
        min_detection_confidence: float = 0.6,
    ):
        """
        Initialize a new realtime analysis session.
        
        Args:
            buffer_size: Size of the frame buffer
            width: Width to resize frames to
            height: Height to resize frames to
            min_detection_confidence: Confidence threshold for face detection
        """
        self.buffer_size = buffer_size
        self.width = width
        self.height = height
        self.min_detection_confidence = min_detection_confidence
        
        # Initialize frame buffer
        self.frame_buffer = deque(maxlen=buffer_size)
        
        # Initialize audio components
        self.audio_analyzer = OratoryAnalyzer()
        self.audio_buffer = bytearray()
        
        # Initialize MediaPipe models with GPU acceleration
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_pose = mp.solutions.pose
        
        # Get optimized configuration for GPU acceleration
        from utils.gpu import get_optimal_mediapipe_config, GPU_AVAILABLE
        
        # Use lightweight model (0) for real-time processing
        mp_config = get_optimal_mediapipe_config(model_complexity=0)
        
        # Create model instances with optimized settings for realtime
        self.face_detector = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0=short-range, 1=full-range 
            min_detection_confidence=min_detection_confidence
        )
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=False,  # Always false for real-time video
            model_complexity=mp_config["model_complexity"],
            smooth_landmarks=True,    # Always true for continuous video
            enable_segmentation=mp_config["enable_segmentation"],
            min_detection_confidence=mp_config["min_detection_confidence"]
        )
        
        # Log GPU acceleration status
        print(f"[RealtimeSession] Using GPU acceleration: {GPU_AVAILABLE}")
        
    
    def add_frame(self, frame_data: bytes) -> bool:
        """
        Add a video frame to the buffer.
        
        Args:
            frame_data: Encoded video frame data
            
        Returns:
            bool: True if frame was successfully added
        """
        frame = decode_frame_data(frame_data, self.width, self.height)
        if frame is None:
            return False
        
        self.frame_buffer.append(frame)
        return True
    
    def add_audio(self, audio_data: bytes) -> None:
        """
        Add audio data to the buffer.
        
        Args:
            audio_data: Raw audio data (PCM)
        """
        self.audio_buffer.extend(audio_data)
    
    def generate_feedback(self) -> Dict[str, Any]:
        """
        Generate feedback from current buffers.
        
        Returns:
            Dict with feedback data
        """
        # Skip if buffer is empty
        if not self.frame_buffer:
            return {"error": "Empty frame buffer"}
        
        start_time = time.time()
        
        # Process video frames
        feedback = process_realtime_buffer(
            list(self.frame_buffer),
            self.face_detector,
            self.pose_detector,
            self.mp_pose
        )
        
        # Process audio if we have enough data (1 second)
        audio_metrics = self._analyze_audio_if_available()
        if audio_metrics:
            feedback["audio_metrics"] = audio_metrics
        
        # Add performance metrics
        processing_time = time.time() - start_time
        feedback["response_time_sec"] = round(processing_time, 3)
        feedback["buffer_size"] = len(self.frame_buffer)
        
        return feedback
    
    def _analyze_audio_if_available(self) -> Dict[str, Any]:
        """
        Analyze audio buffer if enough data is available.
        
        Returns:
            Dict of audio metrics or None if not enough data
        """
        # Check if we have at least 1 second of audio (16kHz, 16-bit = 32000 bytes)
        if len(self.audio_buffer) < 32000:
            return None
        
        try:
            # Convert to float array
            pcm_int16 = np.frombuffer(self.audio_buffer, dtype=np.int16)
            audio_float = pcm_int16.astype(np.float32) / 32768.0
            
            # Save to temporary file for analysis
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpwav:
                tmpfile_path = tmpwav.name
                sf.write(tmpfile_path, audio_float, 16000)
                
                try:
                    # Analyze with new OratoryAnalyzer
                    return self.audio_analyzer.analyze(tmpfile_path)
                finally:
                    # Clean up temp file
                    tmpwav.close()
                    os.unlink(tmpfile_path)
            
        except Exception as e:
            return {"error": f"Audio analysis error: {str(e)}"}
        finally:
            # Clear buffer after analysis
            self.audio_buffer.clear()
    
    def close(self) -> None:
        """Clean up resources."""
        self.face_detector.close()
        self.pose_detector.close()


async def handle_realtime_feedback(
    websocket: WebSocket,
    buffer_size: int = 10,
    width: int = 160,
    height: int = 160,
    min_detection_confidence: float = 0.6,
    feedback_interval: float = 0.5
):
    """
    Handle WebSocket connection for real-time video analysis.
    
    Args:
        websocket: WebSocket connection
        buffer_size: Size of the frame buffer
        width: Width to resize frames to
        height: Height to resize frames to
        min_detection_confidence: Confidence threshold for face detection
        feedback_interval: Time between feedback messages in seconds
    """
    await websocket.accept()
    
    # Initialize session
    session = None
    
    try:
        session = RealtimeSession(
            buffer_size=buffer_size,
            width=width,
            height=height,
            min_detection_confidence=min_detection_confidence,
        )
        
        # Timing control
        last_feedback_time = 0
        
        while True:
            # Receive binary data (video frame or audio chunk)
            data = await websocket.receive_bytes()
            
            try:
                # Check prefix for audio chunks (client should send b'AUD' + pcm16le)
                if data.startswith(b'AUD'):
                    session.add_audio(data[3:])
                    continue  # Skip further processing for audio
                
                # Process video frame
                session.add_frame(data)
                
                # Send feedback at regular intervals
                current_time = time.time()
                if current_time - last_feedback_time >= feedback_interval:
                    feedback = session.generate_feedback()
                    await websocket.send_json(feedback)
                    last_feedback_time = current_time
                    
            except Exception as e:
                # Handle frame processing errors
                await websocket.send_json({
                    "error": f"Error processing frame: {str(e)}",
                    "timestamp": time.time()
                })
    
    except WebSocketDisconnect:
        # Client disconnected, clean exit
        pass
    
    except Exception as e:
        # Handle connection errors
        try:
            if websocket.client_state.name == "CONNECTED":
                await websocket.send_json({
                    "error": f"WebSocket connection error: {str(e)}",
                    "timestamp": time.time()
                })
                await websocket.close()
        except:
            pass
    
    finally:
        # Clean up resources
        if session is not None:
            session.close()
