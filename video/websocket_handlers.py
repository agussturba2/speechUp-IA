"""
WebSocket handlers for real-time video processing
"""

import time
from collections import deque
from fastapi import WebSocket, WebSocketDisconnect
import mediapipe as mp
import numpy as np
import tempfile
import soundfile as sf
from audio import AudioAnalyzer

from .realtime import decode_frame_data, process_realtime_buffer

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
    
    try:
        # Initialize MediaPipe models once
        mp_face_detection = mp.solutions.face_detection
        mp_pose = mp.solutions.pose
        
        # Configure models for real-time processing
        with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=min_detection_confidence
        ) as face_detector, mp_pose.Pose(
            static_image_mode=False,  # Optimized for video
            model_complexity=0,       # Lightweight model for real-time
            smooth_landmarks=True,    # Smoothing for continuous video
            enable_segmentation=False,
            min_detection_confidence=0.5
        ) as pose_detector:
            
            # Initialize audio analyzer and buffer
            audio_analyzer = AudioAnalyzer()
            audio_buffer = bytearray()
            # Circular buffer for recent frames
            frame_buffer = deque(maxlen=buffer_size)
            
            # Timing control
            last_feedback_time = 0
            
            while True:
                # Receive binary data (video frame or audio chunk)
                frame_data = await websocket.receive_bytes()

                # Check prefix for audio chunks (client should send b'AUD' + pcm16le)
                if frame_data.startswith(b'AUD'):
                    audio_buffer.extend(frame_data[3:])
                    # Skip further processing for audio-only packet
                    continue
                try:
                    # Process the received frame (video)
                    frame = decode_frame_data(frame_data, width, height)
                    if frame is None:
                        continue
                        
                    # Add to buffer
                    frame_buffer.append(frame)
                    
                    # Check if it's time to send feedback
                    current_time = time.time()
                    if (current_time - last_feedback_time >= feedback_interval and 
                            len(frame_buffer) > 0):
                        # Generate feedback from current buffer
                        feedback = process_realtime_buffer(
                            list(frame_buffer), face_detector, pose_detector, mp_pose
                        )
                        # If we have >=1s audio (32000 bytes), analyze and attach metrics
                        if len(audio_buffer) >= 32000:  # 16000 samples * 2 bytes
                            pcm_int16 = np.frombuffer(audio_buffer, dtype=np.int16)
                            audio_float = pcm_int16.astype(np.float32) / 32768.0
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpwav:
                                sf.write(tmpwav.name, audio_float, 16000)
                                try:
                                    audio_metrics = audio_analyzer.analyze(tmpwav.name)
                                    feedback["audio_metrics"] = audio_metrics
                                finally:
                                    tmpwav.close()
                                    import os
                                    os.unlink(tmpwav.name)
                            audio_buffer.clear()
                        
                        # Add performance metrics
                        processing_time = time.time() - current_time
                        feedback["tiempo_respuesta"] = round(processing_time, 3)
                        feedback["buffer_size"] = len(frame_buffer)
                        
                        # Send feedback to client
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
            # If we can't even send an error message, just exit
            pass
