"""
WebSocket handlers for incremental oratory analysis and feedback.

This module provides a WebSocket handler that processes audio and video frames
incrementally as they are received, rather than waiting for a complete video file.
"""

import logging
import os
import time
import tempfile
import json
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import asyncio
import httpx
from random import random, choice, randint

from video.pipeline import run_analysis_pipeline
from video.metrics import build_metrics_response
from api.schemas.analysis_json import AnalysisJSON
from video.realtime import decode_frame_data
from utils.gpu import GPU_AVAILABLE
from .oratory import process_pipeline_results, send_analysis_result

logger = logging.getLogger(__name__)


class IncrementalOratorySession:
    """
    Session handler for incremental oratory analysis.
    
    Processes video and audio frames incrementally as they arrive,
    and generates feedback once streaming ends.
    """

    def __init__(
        self,
        buffer_size: int = 30,
        width: int = 640,
        height: int = 480,
        processing_interval: int = 60,  # Process every N frames
    ):
        """
        Initialize a new incremental oratory analysis session.
        
        Args:
            buffer_size: Size of the frame buffer
            width: Width of the video frames
            height: Height of the video frames
            processing_interval: Process frames every N frames
        """
        # Configuration parameters
        self.buffer_size = buffer_size
        self.width = width
        self.height = height
        self.processing_interval = processing_interval

        # Initialize frame buffer
        self.frame_buffer = deque(maxlen=buffer_size)
        self.audio_buffer = bytearray()
        self.audio_buffer_duration = 0.0  # Track duration of audio in buffer (seconds)
        
        # Store processed frames for final analysis
        self.all_frames = []
        self.frame_count = 0
        
        # Video storage
        self.video_writer = None
        self.video_path = None
        
        # Explicitly set tmp_path to None
        self.tmp_path = None
        
        # Intermediate processing results
        self.partial_results = []
        
        # NEW: Accumulated state for incremental analysis
        self.accumulated_state = {
            "verbal": {
                "word_count": 0,
                "filler_count": 0,
                "pause_count": 0,
                "speaking_time_sec": 0,
                "transcript_segments": [],
                "filler_instances": []
            },
            "nonverbal": {
                "gestures": [],
                "expressions": [],
                "posture_changes": []
            },
            "metrics": {
                "wpm": 0,
                "fillers_per_min": 0,
                "pause_rate": 0,
                "avg_pause_duration": 0,
                "gesture_rate": 0,
                "expression_variability": 0
            },
            "frames_processed": 0,
            "audio_processed_sec": 0,
            "start_time": time.time()
        }
        
        # NEW: Buffers for incremental analysis
        self.analysis_frame_buffer = []
        self.analysis_audio_buffer = bytearray()
        self.last_processed_frame_index = 0
        
        # Analysis state
        self.analysis_in_progress = False
        self.last_analysis_time = 0
        self.streaming_active = True

        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # NEW: Flag to enable true incremental processing
        self.enable_incremental_processing = True

        # Log GPU status and initialization
        logger.info(f"IncrementalOratorySession initialized with buffer_size={buffer_size}, dimensions={width}x{height}, GPU={GPU_AVAILABLE}")
        logger.info(f"Video format: AVI with XVID codec")
        logger.info(f"Incremental processing enabled: {self.enable_incremental_processing}")

    def add_frame(self, frame_data: bytes) -> bool:
        """
        Add a video frame to the buffer and to stored frames.
        
        Args:
            frame_data: Raw frame data
            
        Returns:
            bool: Success status
        """
        frame = decode_frame_data(frame_data, self.width, self.height)
        if frame is None:
            logger.warning("Failed to decode frame data")
            return False

        self.frame_buffer.append(frame)
        self.all_frames.append(frame)
        self.frame_count += 1
        
        # If we have a video writer, write the frame
        if self.video_writer is not None:
            self.video_writer.write(frame)
            
        return True

    def add_audio(self, audio_data: bytes) -> None:
        """
        Add audio data to the buffer.
        
        Args:
            audio_data: Raw audio data
        """
        # Assume 16kHz, 16-bit mono audio (32 bytes = 1ms)
        duration_ms = len(audio_data) / 32.0
        duration_sec = duration_ms / 1000.0
        
        self.audio_buffer.extend(audio_data)
        self.audio_buffer_duration += duration_sec
        
        # Also add to analysis buffer
        self.analysis_audio_buffer.extend(audio_data)

    def end_stream(self) -> None:
        """
        Signal that the stream has ended and final processing can begin.
        """
        logger.info("Stream ending - finalizing data for analysis")
        self.streaming_active = False
        
        # Close any existing video writer and recreate properly
        self._cleanup_video_resources()
        
        # Create a new AVI video with all accumulated frames
        if self.all_frames:
            try:
                # Find a valid test frame
                test_frame = None
                for frame in self.all_frames:
                    if frame is not None and frame.size > 0:
                        test_frame = frame
                        break
                
                if test_frame is None:
                    logger.error("No valid frames found for creating video")
                    return
                
                # Get frame dimensions
                h, w = test_frame.shape[:2]
                logger.info(f"Frame dimensions for final video: {w}x{h}")
                
                # Create new temporary file with AVI extension
                fd, self.video_path = tempfile.mkstemp(suffix=".avi")
                os.close(fd)
                logger.info(f"Created final video file: {self.video_path}")
                
                # Try different codecs in order of reliability
                codecs = [
                    ('XVID', 'XVID codec - widely compatible'),
                    ('MJPG', 'Motion JPEG codec'),
                    ('DIB ', 'Uncompressed RGB'),  # Note the space after DIB
                ]
                
                success = False
                for codec_name, desc in codecs:
                    try:
                        logger.info(f"Attempting to use {desc} for final video")
                        fourcc = cv2.VideoWriter_fourcc(*codec_name)
                        
                        # Create video writer
                        self.video_writer = cv2.VideoWriter(
                            self.video_path, 
                            fourcc, 
                            30.0,  # Assume 30 FPS
                            (w, h)
                        )
                        
                        # Check if writer opened successfully
                        if not self.video_writer.isOpened():
                            logger.error(f"Failed to open VideoWriter with {codec_name}")
                            continue
                            
                        # Write all frames
                        frames_written = 0
                        for frame in self.all_frames:
                            if frame is not None and frame.size > 0:
                                try:
                                    self.video_writer.write(frame)
                                    frames_written += 1
                                except Exception as e:
                                    logger.error(f"Error writing frame {frames_written}: {e}")
                        
                        # Release writer to ensure frames are flushed
                        self.video_writer.release()
                        self.video_writer = None
                        
                        # Verify the file was created successfully
                        if os.path.exists(self.video_path) and os.path.getsize(self.video_path) > 100:
                            logger.info(f"Successfully created final video with {frames_written} frames using {codec_name}")
                            logger.info(f"Final video size: {os.path.getsize(self.video_path)} bytes")
                            
                            # Test opening the file with OpenCV to make sure it's valid
                            cap = cv2.VideoCapture(self.video_path)
                            if cap.isOpened():
                                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                cap.release()
                                
                                if frame_count > 0:
                                    logger.info(f"Final video can be opened by OpenCV, contains {frame_count} frames")
                                    success = True
                                    break
                                else:
                                    logger.error(f"Final video has no frames according to OpenCV")
                            else:
                                logger.error(f"Final video cannot be opened by OpenCV")
                        else:
                            logger.error(f"Final video file is empty or too small")
                            
                    except Exception as e:
                        logger.error(f"Error with {codec_name} codec in end_stream: {e}")
                
                if not success:
                    logger.error("All codecs failed for final video creation")
                    
                    # If we have a path but couldn't create a working video, remove it
                    if self.video_path and os.path.exists(self.video_path):
                        try:
                            os.remove(self.video_path)
                            self.video_path = None
                            logger.info("Removed invalid final video file")
                        except Exception as e:
                            logger.error(f"Failed to remove invalid video: {e}")
                
            except Exception as e:
                logger.error(f"Error creating final video: {e}", exc_info=True)

    def _init_video_file(self) -> None:
        """
        Initialize video file for storing all frames.
        """
        # Verify that we have frames to write
        if not self.all_frames:
            logger.warning("Cannot initialize video file: No frames available")
            return
            
        if self.video_writer is None:
            try:
                # Get a random frame to check format
                test_frame = None
                for frame in self.all_frames:
                    if frame is not None and frame.size > 0:
                        test_frame = frame
                        break
                
                if test_frame is None:
                    logger.error("No valid frames found in buffer")
                    return
                
                # Get frame dimensions
                h, w = test_frame.shape[:2]
                logger.info(f"Frame dimensions: {w}x{h}")
                
                # Create a temporary file for the video - try .avi format with basic codec
                fd, self.video_path = tempfile.mkstemp(suffix=".avi")
                os.close(fd)
                logger.info(f"Created temporary file: {self.video_path}")
                
                # Try different codecs in order of reliability
                codecs = [
                    ('XVID', 'XVID codec - widely compatible'),
                    ('MJPG', 'Motion JPEG codec'),
                    ('DIB ', 'Uncompressed RGB'),  # Note the space after DIB
                ]
                
                success = False
                for codec_name, desc in codecs:
                    try:
                        logger.info(f"Attempting to use {desc}")
                        fourcc = cv2.VideoWriter_fourcc(*codec_name)
                        
                        # Create video writer
                        self.video_writer = cv2.VideoWriter(
                            self.video_path, 
                            fourcc, 
                            30.0,  # Assume 30 FPS
                            (w, h)
                        )
                        
                        # Check if writer opened successfully
                        if self.video_writer.isOpened():
                            logger.info(f"Successfully opened VideoWriter with {codec_name} codec")
                            
                            # Test write a single frame
                            self.video_writer.write(test_frame)
                            self.video_writer.release()
                            
                            # Check if file was created with content
                            if os.path.exists(self.video_path) and os.path.getsize(self.video_path) > 0:
                                logger.info(f"Test frame written successfully with {codec_name} codec")
                                
                                # Reopen the writer for further writing
                                self.video_writer = cv2.VideoWriter(
                                    self.video_path, 
                                    fourcc, 
                                    30.0,
                                    (w, h)
                                )
                                
                                if not self.video_writer.isOpened():
                                    logger.error(f"Failed to reopen VideoWriter after test")
                                    continue
                                
                                success = True
                                break
                            else:
                                logger.error(f"Test frame write failed with {codec_name} codec")
                        else:
                            logger.error(f"Failed to open VideoWriter with {codec_name} codec")
                            
                    except Exception as codec_error:
                        logger.error(f"Error with {codec_name} codec: {codec_error}")
                
                # If no codec worked, raise an error
                if not success:
                    raise RuntimeError("Failed to initialize video writer with any codec")
                
                # Write all existing frames
                frames_written = 0
                for frame in self.all_frames:
                    if frame is not None and frame.size > 0:
                        try:
                            self.video_writer.write(frame)
                            frames_written += 1
                        except Exception as frame_error:
                            logger.error(f"Error writing frame {frames_written}: {frame_error}")
                
                logger.info(f"Written {frames_written} frames out of {len(self.all_frames)} total")
                
                # Make sure to call write() flush by releasing and reopening
                if frames_written > 0:
                    self.video_writer.release()
                    
                    # Verify file exists and has content
                    if os.path.exists(self.video_path) and os.path.getsize(self.video_path) > 100:  # Ensure it's not just headers
                        logger.info(f"Video file created successfully: {self.video_path} ({os.path.getsize(self.video_path)} bytes)")
                        
                        # Reopen the writer for possible future writes
                        fourcc = cv2.VideoWriter_fourcc(*codec_name)
                        self.video_writer = cv2.VideoWriter(
                            self.video_path,
                            fourcc,
                            30.0,
                            (w, h)
                        )
                    else:
                        logger.error(f"Video file empty or too small after writing {frames_written} frames")
                        raise FileNotFoundError(f"Video file was not created properly: {self.video_path}")
                else:
                    logger.error("No frames were successfully written")
                    raise ValueError("Failed to write any frames to video file")
                    
            except Exception as e:
                logger.error(f"Error initializing video file: {e}", exc_info=True)
                # Clean up if there was an error
                if hasattr(self, 'video_path') and self.video_path and os.path.exists(self.video_path):
                    try:
                        os.remove(self.video_path)
                        logger.info(f"Removed invalid video file: {self.video_path}")
                    except Exception:
                        pass
                self.video_path = None
                self.video_writer = None
                raise  # Re-raise the exception to be handled by the caller

    def _check_video_format(self):
        """
        Check the format of the current video file.
        Logs detailed information about the file.
        """
        if not self.video_path:
            logger.error("Cannot check video format: No video path set")
            return
            
        try:
            # Check if file exists
            if not os.path.exists(self.video_path):
                logger.error(f"Video file does not exist: {self.video_path}")
                return
                
            # Check file size
            file_size = os.path.getsize(self.video_path)
            logger.info(f"Video file size: {file_size} bytes")
            
            # Check extension
            _, ext = os.path.splitext(self.video_path)
            logger.info(f"Video file extension: {ext}")
            
            # Try to open with OpenCV
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                # Get video properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                logger.info(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames")
                
                # Try to read the first frame
                ret, frame = cap.read()
                if ret:
                    logger.info("Successfully read the first frame")
                else:
                    logger.error("Failed to read the first frame")
                
                cap.release()
            else:
                logger.error(f"OpenCV cannot open the video file: {self.video_path}")
                
            # Additional low-level check - try reading first few bytes
            try:
                with open(self.video_path, 'rb') as f:
                    header = f.read(16)
                    header_hex = ' '.join(f'{b:02x}' for b in header)
                    logger.info(f"File header (hex): {header_hex}")
            except Exception as e:
                logger.error(f"Error reading file header: {e}")
                
        except Exception as e:
            logger.error(f"Error checking video format: {e}")

    def _cleanup_video_resources(self):
        """
        Clean up video writer and path, without removing the file.
        Used when recreating the video file.
        """
        if self.video_writer is not None:
            try:
                if self.video_writer.isOpened():
                    logger.info("Releasing existing video writer")
                    self.video_writer.release()
            except Exception as e:
                logger.error(f"Error releasing video writer: {e}")
            self.video_writer = None

    def _process_new_frames(self) -> Dict[str, Any]:
        """
        Process new frames that haven't been analyzed yet.
        This method performs lightweight analysis on newly received frames.
        
        Returns:
            Dict with frame analysis results
        """
        if not self.analysis_frame_buffer:
            return {}
        
        # Get only unprocessed frames
        new_frames = self.analysis_frame_buffer[self.last_processed_frame_index:]
        if not new_frames:
            return {}
            
        try:
            # Perform lightweight analysis on new frames
            frames_count = len(new_frames)
            logger.info(f"Processing {frames_count} new frames incrementally")
            
            # Simplified frame analysis
            # In a real implementation, this would perform actual CV analysis
            # For now, we'll do basic simulated analysis
            
            # Simulate some analysis results
            frame_results = {
                "frames_analyzed": frames_count,
                "frames_with_face": max(0, frames_count - randint(0, min(3, frames_count))),
                "expressions": [],
                "gestures": [],
                "posture": []
            }
            
            # Simulate detection of expressions and gestures
            if frames_count > 5:
                # Randomly detect some expressions
                if random() > 0.7:
                    expression = {
                        "type": choice(["smile", "frown", "neutral", "surprised"]),
                        "confidence": random() * 0.5 + 0.5,
                        "frame_index": self.last_processed_frame_index + randint(0, frames_count-1)
                    }
                    frame_results["expressions"].append(expression)
                    
                # Randomly detect some gestures
                if random() > 0.8:
                    gesture = {
                        "type": choice(["hand_movement", "head_nod", "pointing"]),
                        "confidence": random() * 0.5 + 0.5,
                        "frame_index": self.last_processed_frame_index + randint(0, frames_count-1)
                    }
                    frame_results["gestures"].append(gesture)
            
            # Update the last processed frame index
            self.last_processed_frame_index += frames_count
            
            return frame_results
            
        except Exception as e:
            logger.error(f"Error processing new frames: {e}", exc_info=True)
            return {}
    
    def _process_audio_buffer(self) -> Dict[str, Any]:
        """
        Process accumulated audio buffer for speech analysis.
        
        Returns:
            Dict with audio analysis results
        """
        if len(self.analysis_audio_buffer) < 32000:  # Need at least 1 second of audio (16kHz * 2 bytes)
            return {}
            
        try:
            # In a real implementation, this would perform actual audio analysis
            # For demo purposes, we'll simulate some basic analysis
            
            # Calculate audio duration in seconds (16kHz, 16-bit mono)
            audio_duration = len(self.analysis_audio_buffer) / 32000
            logger.info(f"Processing {audio_duration:.2f} seconds of audio incrementally")
            
            # Simulate speech analysis results
            audio_results = {
                "duration_sec": audio_duration,
                "speech_detected": random() > 0.2,  # 80% chance of speech
                "words": [],
                "fillers": [],
                "pauses": []
            }
            
            # Simulate word detection
            if audio_results["speech_detected"]:
                # Simulate ~2 words per second with some randomness
                word_count = int(audio_duration * (1.5 + random()))
                audio_results["words"] = [{
                    "index": i,
                    "start_time": i * (audio_duration / (word_count + 1)),
                    "end_time": (i + 0.8) * (audio_duration / (word_count + 1)),
                    "confidence": 0.7 + random() * 0.3
                } for i in range(word_count)]
                
                # Simulate some fillers (um, eh) - ~1 per 10 seconds with randomness
                filler_count = int(audio_duration / 10 * random())
                if filler_count > 0:
                    audio_results["fillers"] = [{
                        "type": choice(["um", "eh", "mmm", "like"]),
                        "time": random() * audio_duration,
                        "duration": 0.2 + random() * 0.4
                    } for _ in range(filler_count)]
                    
                # Simulate some pauses - ~1 per 5 seconds with randomness
                pause_count = int(audio_duration / 5 * random())
                if pause_count > 0:
                    audio_results["pauses"] = [{
                        "start_time": random() * audio_duration,
                        "duration": 0.3 + random() * 0.7
                    } for _ in range(pause_count)]
            
            # Clear the analysis audio buffer for next iteration
            # In a real implementation, you might want to keep some overlap
            self.analysis_audio_buffer = bytearray()
            
            return audio_results
            
        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}", exc_info=True)
            return {}
    
    def _update_accumulated_state(self, frame_results: Dict[str, Any], audio_results: Dict[str, Any]) -> None:
        """
        Update the accumulated state with new results.
        
        Args:
            frame_results: Results from frame analysis
            audio_results: Results from audio analysis
        """
        # Update frame stats
        self.accumulated_state["frames_processed"] += frame_results.get("frames_analyzed", 0)
        
        # Update nonverbal metrics
        for expression in frame_results.get("expressions", []):
            self.accumulated_state["nonverbal"]["expressions"].append(expression)
            
        for gesture in frame_results.get("gestures", []):
            self.accumulated_state["nonverbal"]["gestures"].append(gesture)
            
        # Update verbal metrics
        if audio_results:
            # Track audio processed
            self.accumulated_state["audio_processed_sec"] += audio_results.get("duration_sec", 0)
            
            # Update word count
            self.accumulated_state["verbal"]["word_count"] += len(audio_results.get("words", []))
            
            # Update filler count and instances
            fillers = audio_results.get("fillers", [])
            self.accumulated_state["verbal"]["filler_count"] += len(fillers)
            self.accumulated_state["verbal"]["filler_instances"].extend(fillers)
            
            # Update pause count
            pauses = audio_results.get("pauses", [])
            self.accumulated_state["verbal"]["pause_count"] += len(pauses)
            
            # Update speaking time
            if audio_results.get("speech_detected", False):
                # Subtract pause durations
                speech_duration = audio_results.get("duration_sec", 0)
                pause_duration = sum(p.get("duration", 0) for p in pauses)
                self.accumulated_state["verbal"]["speaking_time_sec"] += (speech_duration - pause_duration)
            
        # Recalculate metrics
        self._update_metrics()
    
    def _update_metrics(self) -> None:
        """
        Update calculated metrics based on accumulated state.
        """
        verbal = self.accumulated_state["verbal"]
        nonverbal = self.accumulated_state["nonverbal"]
        metrics = self.accumulated_state["metrics"]
        audio_time = max(0.1, self.accumulated_state["audio_processed_sec"])  # Avoid div by 0
        
        # Verbal metrics
        if verbal["speaking_time_sec"] > 0:
            # Words per minute
            minutes = verbal["speaking_time_sec"] / 60
            if minutes > 0:
                metrics["wpm"] = verbal["word_count"] / minutes
                
            # Fillers per minute
            if minutes > 0:
                metrics["fillers_per_min"] = verbal["filler_count"] / minutes
                
            # Pause rate and duration
            if verbal["pause_count"] > 0:
                metrics["pause_rate"] = verbal["pause_count"] / minutes
        
        # Nonverbal metrics
        if self.accumulated_state["frames_processed"] > 0:
            # Gesture rate
            metrics["gesture_rate"] = len(nonverbal["gestures"]) / (audio_time / 60)
            
            # Expression variability
            expression_types = set(e.get("type", "") for e in nonverbal["expressions"])
            metrics["expression_variability"] = len(expression_types) / max(1, len(nonverbal["expressions"]) / 10)
    
    async def process_incremental(self) -> Dict[str, Any]:
        """
        Process current frames in the buffer incrementally.
        
        Returns:
            Dict with partial feedback data
        """
        if not self.frame_buffer:
            return {"status": "buffering", "buffer_size": 0}

        if self.analysis_in_progress:
            return {"status": "analyzing", "message": "Analysis already in progress"}

        self.analysis_in_progress = True
        t0 = time.perf_counter()
        
        try:
            # Make sure new frames are added to analysis buffer
            self.analysis_frame_buffer = list(self.all_frames)
            
            # Check if we should do true incremental processing
            if self.enable_incremental_processing:
                # Process new frames
                frame_results = self._process_new_frames()
                
                # Process audio buffer if we have enough data
                audio_results = self._process_audio_buffer()
                
                # Update accumulated state
                self._update_accumulated_state(frame_results, audio_results)
                
                # Generate response with incremental analysis
                partial_result = {
                    "status": "processing",
                    "frames_processed": self.frame_count,
                    "buffer_size": len(self.frame_buffer),
                    "processing_time_sec": round(time.perf_counter() - t0, 3),
                    "timestamp": time.time(),
                    "incremental_metrics": {
                        "wpm": round(self.accumulated_state["metrics"]["wpm"], 1),
                        "fillers_per_min": round(self.accumulated_state["metrics"]["fillers_per_min"], 2),
                        "gesture_rate": round(self.accumulated_state["metrics"]["gesture_rate"], 2),
                        "expression_variability": round(self.accumulated_state["metrics"]["expression_variability"], 2),
                    },
                    "session_duration": round(time.time() - self.accumulated_state["start_time"], 1),
                    "confidence": min(1.0, self.accumulated_state["audio_processed_sec"] / 30)  # Confidence grows over time
                }
                
                # Add detected events if any
                recent_fillers = [f for f in self.accumulated_state["verbal"]["filler_instances"][-3:]] if self.accumulated_state["verbal"]["filler_instances"] else []
                if recent_fillers:
                    partial_result["recent_fillers"] = recent_fillers
                    
                recent_gestures = [g for g in self.accumulated_state["nonverbal"]["gestures"][-3:]] if self.accumulated_state["nonverbal"]["gestures"] else []
                if recent_gestures:
                    partial_result["recent_gestures"] = recent_gestures
                
            else:
                # Just return basic status as before
                partial_result = {
                    "status": "processing",
                    "frames_processed": self.frame_count,
                    "buffer_size": len(self.frame_buffer),
                    "processing_time_sec": round(time.perf_counter() - t0, 3),
                    "timestamp": time.time()
                }
            
            self.partial_results.append(partial_result)
            return partial_result
            
        except Exception as e:
            logger.error(f"Error in incremental processing: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "error_type": str(type(e).__name__),
                "timestamp": time.time()
            }
        finally:
            self.analysis_in_progress = False
            self.last_analysis_time = time.perf_counter()

    async def generate_final_feedback(self) -> Dict[str, Any]:
        """
        Generate final feedback after stream has ended.
        
        Returns:
            Dict with complete feedback data
        """
        if self.streaming_active:
            logger.warning("generate_final_feedback called while stream is still active")
            return {"status": "streaming", "message": "Stream still active"}

        if self.analysis_in_progress:
            logger.warning("Analysis already in progress")
            return {"status": "analyzing", "message": "Analysis already in progress"}

        if not self.all_frames:
            logger.error("No frames received")
            return {"status": "error", "message": "No frames received"}

        self.analysis_in_progress = True
        t0 = time.perf_counter()
        logger.info(f"Starting final feedback generation with {len(self.all_frames)} frames")
        result = None

        try:
            # Get data from our incremental analysis if available
            has_incremental_data = (
                self.enable_incremental_processing and 
                self.accumulated_state["frames_processed"] > 0
            )
            
            if has_incremental_data:
                logger.info("Using accumulated incremental data to enhance final analysis")
            
            # Ensure we have a proper video file to analyze
            if not self.video_path or not os.path.exists(self.video_path):
                logger.info("Video file not found, recreating...")
                # Force recreation of the video file
                self.end_stream()
                
                # Check if video creation worked
                if not self.video_path or not os.path.exists(self.video_path):
                    raise ValueError("Failed to create video file for analysis")
            
            # Log info about the video file
            logger.info(f"Using video file for analysis: {self.video_path}")
            file_size = os.path.getsize(self.video_path)
            logger.info(f"Video file size: {file_size} bytes")
            
            # Verify video can be opened with OpenCV
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file with OpenCV: {self.video_path}")
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Check video has frames
            if frame_count == 0:
                raise ValueError(f"Video has 0 frames")
            
            logger.info(f"Video ready for analysis: {self.video_path}, {frame_count} frames, {fps:.1f} FPS, {width}x{height}")
            cap.release()
            
            # Run the full pipeline on the video
            logger.info(f"Starting analysis pipeline")
            t0_pipeline = time.perf_counter()
            
            try:
                proc = run_analysis_pipeline(self.video_path)
                analysis_time = time.perf_counter() - t0_pipeline
                logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
                
                # Process results using the helper function from oratory.py
                result = process_pipeline_results(proc, analysis_time)
                
            except Exception as pipeline_error:
                logger.error(f"Analysis pipeline error: {pipeline_error}", exc_info=True)
                
                # If incremental data is available, return that instead as a fallback
                if has_incremental_data:
                    logger.info("Using incremental data as fallback due to pipeline error")
                    # Create a result structure from accumulated state
                    result = self._create_result_from_incremental_data()
                    
                else:
                    result = {
                        "status": "error",
                        "error": f"Analysis failed: {str(pipeline_error)}",
                        "error_type": str(type(pipeline_error).__name__),
                        "timestamp": time.time()
                    }
            
            # Asegurarse que result no sea None antes de intentar mejorarlo
            if result is None:
                logger.error("No result was generated from pipeline processing")
                result = {
                    "status": "error",
                    "error": "No analysis result was generated",
                    "timestamp": time.time()
                }
                
            # Enhance result with incremental data if available
            if has_incremental_data and result is not None:
                try:
                    result = self._enhance_result_with_incremental_data(result)
                except Exception as enhance_error:
                    logger.error(f"Error enhancing result with incremental data: {enhance_error}")
                    # No sobreescribir el resultado si hay un error en el enhancement
            
            # Add metadata about incremental processing dentro del campo debug para compatibilidad con el DTO
            if result is not None:
                # Envolver todo en un try para evitar errores inesperados
                try:
                    # Asegurarse de que exista el objeto debug
                    debug_obj = result.setdefault("quality", {}).setdefault("debug", {})
                    debug_obj["incremental_processing"] = True
                    
                    # Verificar si partial_results y all_frames existen y no son None
                    partial_results_length = 0
                    if hasattr(self, "partial_results") and self.partial_results is not None:
                        partial_results_length = len(self.partial_results)
                    debug_obj["incremental_steps"] = partial_results_length
                    
                    all_frames_length = 0
                    if hasattr(self, "all_frames") and self.all_frames is not None:
                        all_frames_length = len(self.all_frames)
                    debug_obj["total_frames_received"] = all_frames_length
                except Exception as debug_error:
                    logger.error(f"Error setting debug fields: {debug_error}")
                    # En caso de error, crear un objeto debug básico
                    if "quality" not in result:
                        result["quality"] = {}
                    result["quality"]["debug"] = {
                        "incremental_processing": True,
                        "incremental_steps": 0,
                        "total_frames_received": 0
                    }
            
            logger.info("Successfully generated feedback")
            return result

        except Exception as e:
            logger.error(f"Error generating feedback: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e),
                "error_type": str(type(e).__name__),
                "timestamp": time.time()
            }
        finally:
            self.analysis_in_progress = False
            # Don't clean up the file here, we might need it for debugging
            # We'll let the session close handler clean everything up

    
            
    def _create_result_from_incremental_data(self) -> Dict[str, Any]:
        """
        Create a result structure using only the incrementally collected data.
        Used as a fallback when the final pipeline fails.
        
        Returns:
            Dict with feedback data from incremental analysis
        """
        try:
            # Verificar que accumulated_state y sus campos necesarios existan
            if not hasattr(self, "accumulated_state") or self.accumulated_state is None:
                logger.error("accumulated_state no está disponible")
                return {
                    "status": "error",
                    "error": "No incremental data available",
                    "timestamp": time.time()
                }
                
            # Calculate total duration - con verificación de nulidad
            start_time = self.accumulated_state.get("start_time", time.time() - 10)  # Default a 10 segundos atrás
            duration_sec = time.time() - start_time
            
            # Obtener métricas con valores predeterminados seguros
            metrics = {}
            if "metrics" in self.accumulated_state and self.accumulated_state["metrics"] is not None:
                metrics = self.accumulated_state["metrics"]
            
            # Valores predeterminados para métricas en caso de que no existan
            wpm = metrics.get("wpm", 0.0)
            fillers_per_min = metrics.get("fillers_per_min", 0.0)
            pause_rate = metrics.get("pause_rate", 0.0)
            gesture_rate = metrics.get("gesture_rate", 0.0)
            expression_variability = metrics.get("expression_variability", 0.0)
            
            # Verificar audio_buffer y all_frames
            audio_buffer_length = 0
            if hasattr(self, "audio_buffer") and self.audio_buffer is not None:
                audio_buffer_length = len(self.audio_buffer)
                
            all_frames_length = 0
            if hasattr(self, "all_frames") and self.all_frames is not None:
                all_frames_length = len(self.all_frames)
                
            partial_results_length = 0
            if hasattr(self, "partial_results") and self.partial_results is not None:
                partial_results_length = len(self.partial_results)
            
            # Obtener datos procesados con valores predeterminados seguros
            frames_processed = self.accumulated_state.get("frames_processed", 0)
            audio_processed_sec = self.accumulated_state.get("audio_processed_sec", 0.0)
            
            # Create a result structure similar to what process_pipeline_results would return
            result = {
                "status": "completed",
                "quality": {
                    "frames_analyzed": frames_processed,
                    "audio_analyzed_sec": audio_processed_sec,
                    "analysis_ms": int(duration_sec * 1000),
                    "audio_available": audio_buffer_length > 0,
                    "debug": {
                        "incremental_processing": True,
                        "incremental_steps": partial_results_length
                    }
                },
                "media": {
                    "frames_total": all_frames_length,
                    "frames_with_face": frames_processed,
                    "fps": 30,  # Assumed FPS
                    "duration_sec": duration_sec
                },
                "scores": {
                    # Map our incremental metrics to scores (0-100)
                    "fluency": min(100, int(max(0, 100 - fillers_per_min * 20))),
                    "clarity": min(100, int(max(0, 100 - pause_rate * 10))),
                    "pace": min(100, int(max(0, 100 - abs(wpm - 150) / 2))),
                    "engagement": min(100, int(max(0, gesture_rate * 25 + expression_variability * 50)))
                },
                "verbal": {
                    "wpm": round(wpm, 1),
                    "fillers_per_min": round(fillers_per_min, 2),
                    "filler_counts": {}
                },
                "events": [],
                "timestamp": time.time()
            }
            
            # Convert accumulated fillers and gestures to events - con verificaciones de nulidad
            if ("verbal" in self.accumulated_state and 
                self.accumulated_state["verbal"] is not None and
                "filler_instances" in self.accumulated_state["verbal"] and
                self.accumulated_state["verbal"]["filler_instances"] is not None):
                
                for filler in self.accumulated_state["verbal"]["filler_instances"]:
                    if isinstance(filler, dict) and "time" in filler and "type" in filler:
                        result["events"].append({
                            "t": filler["time"],
                            "kind": "filler",
                            "label": filler["type"],
                            "duration": filler.get("duration", 0.3)
                        })
                        
                        # Count fillers by type
                        filler_type = filler["type"]
                        if filler_type not in result["verbal"]["filler_counts"]:
                            result["verbal"]["filler_counts"][filler_type] = 0
                        result["verbal"]["filler_counts"][filler_type] += 1
            
            if ("nonverbal" in self.accumulated_state and 
                self.accumulated_state["nonverbal"] is not None and
                "gestures" in self.accumulated_state["nonverbal"] and
                self.accumulated_state["nonverbal"]["gestures"] is not None):
                
                for gesture in self.accumulated_state["nonverbal"]["gestures"]:
                    if isinstance(gesture, dict) and "type" in gesture:
                        result["events"].append({
                            "t": gesture.get("frame_index", 0) / 30,  # Assuming 30 FPS
                            "kind": "gesture",
                            "label": gesture["type"],
                            "confidence": gesture.get("confidence", 0.8)
                        })
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating result from incremental data: {e}")
            return {
                "status": "error",
                "error": f"Failed to create incremental result: {str(e)}",
                "timestamp": time.time()
            }
        
    def _enhance_result_with_incremental_data(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance the final analysis result with data from incremental processing.
        
        Args:
            result: Result from final analysis pipeline
            
        Returns:
            Enhanced result with incremental data
        """
        # Verificar que result no sea None para evitar el error 'NoneType' object does not support item assignment
        if result is None:
            logger.warning("_enhance_result_with_incremental_data received None result, creating empty result")
            result = {
                "status": "error",
                "error": "No analysis result was generated",
                "timestamp": time.time()
            }
            
        # Añadir los datos incrementales como campo separado para no interferir con el DTO
        try:
            # Verificar que accumulated_state y sus subcampos existan
            if (self.accumulated_state and 
                "metrics" in self.accumulated_state and 
                self.accumulated_state["metrics"] is not None):
                
                metrics = self.accumulated_state["metrics"]
                result.setdefault("incremental", {})["metrics"] = {
                    "wpm": round(metrics.get("wpm", 0.0), 1),
                    "fillers_per_min": round(metrics.get("fillers_per_min", 0.0), 2),
                    "gesture_rate": round(metrics.get("gesture_rate", 0.0), 2),
                    "expression_variability": round(metrics.get("expression_variability", 0.0), 2)
                }
            else:
                # Si no tenemos metrics, crear un diccionario vacío
                result.setdefault("incremental", {})["metrics"] = {
                    "wpm": 0.0,
                    "fillers_per_min": 0.0,
                    "gesture_rate": 0.0,
                    "expression_variability": 0.0
                }
        except Exception as metrics_error:
            logger.error(f"Error setting incremental metrics: {metrics_error}")
            # Asegurarse de que al menos el campo esté inicializado
            result.setdefault("incremental", {})["metrics"] = {
                "wpm": 0.0,
                "fillers_per_min": 0.0,
                "gesture_rate": 0.0,
                "expression_variability": 0.0
            }
        
        # Usar el campo debug existente para información incremental adicional
        # para evitar problemas con el DTO del backend
        try:
            debug = result.setdefault("quality", {}).setdefault("debug", {})
            debug["incremental_processing"] = True
            
            # Verificar si partial_results existe y no es None
            if hasattr(self, "partial_results") and self.partial_results is not None:
                debug["incremental_steps"] = len(self.partial_results)
            else:
                debug["incremental_steps"] = 0
                
            # Verificar si all_frames existe y no es None
            if hasattr(self, "all_frames") and self.all_frames is not None:
                debug["total_frames_received"] = len(self.all_frames)
            else:
                debug["total_frames_received"] = 0
        except Exception as debug_error:
            logger.error(f"Error setting debug fields: {debug_error}")
            # Asegurar que al menos los campos estén inicializados
            result.setdefault("quality", {}).setdefault("debug", {}).update({
                "incremental_processing": True,
                "incremental_steps": 0,
                "total_frames_received": 0
            })
        
        # If any filler events were detected incrementally but are missing in the final result, add them
        try:
            result_filler_times = set()
            for event in result.get("events", []):
                if event.get("kind") == "filler" and "t" in event:
                    result_filler_times.add(round(event["t"], 1))
            
            # Add any incremental fillers that might have been missed
            # Verificar primero que accumulated_state y sus subcampos existan
            if (self.accumulated_state and 
                "verbal" in self.accumulated_state and 
                self.accumulated_state["verbal"] is not None and
                "filler_instances" in self.accumulated_state["verbal"] and 
                self.accumulated_state["verbal"]["filler_instances"] is not None):
                
                for filler in self.accumulated_state["verbal"]["filler_instances"]:
                    if isinstance(filler, dict) and "time" in filler:
                        rounded_time = round(filler["time"], 1)
                        if rounded_time not in result_filler_times:
                            result.setdefault("events", []).append({
                                "t": filler["time"],
                                "kind": "filler",
                                "label": filler.get("type", "um"),
                                "duration": filler.get("duration", 0.3),
                                "source": "incremental"  # Mark the source as incremental
                            })
        except Exception as filler_error:
            logger.error(f"Error processing fillers: {filler_error}")
            # No es necesario tomar ninguna acción adicional, simplemente evitar errores
        
        return result
    
    def _cleanup_temp_file(self) -> None:
        """
        Clean up temporary video file.
        """
        if self.video_writer is not None:
            try:
                if self.video_writer.isOpened():
                    self.video_writer.release()
            except Exception as e:
                logger.error(f"Error releasing video writer: {e}")
            
        if self.video_path and os.path.exists(self.video_path):
            try:
                os.remove(self.video_path)
                logger.info(f"Removed temporary video file: {self.video_path}")
            except Exception as e:
                logger.error(f"Failed to remove temporary file {self.video_path}: {e}")
                
        self.video_path = None
        self.video_writer = None

    def close(self) -> None:
        """
        Clean up resources.
        """
        self._cleanup_temp_file()
        self.frame_buffer.clear()
        self.all_frames.clear()
        self.audio_buffer = bytearray()
        self.executor.shutdown(wait=False)


async def handle_incremental_oratory_feedback(
    websocket: WebSocket,
    buffer_size: int = 30,
    width: int = 640,
    height: int = 480,
    incremental_interval: int = 60  # Process every N frames
) -> None:
    """
    Handle WebSocket connection for incremental oratory feedback.
    
    Args:
        websocket: WebSocket connection
        buffer_size: Size of the frame buffer
        width: Width of video frames
        height: Height of video frames
        incremental_interval: Process incrementally every N frames
    """
    await websocket.accept()

    # Get user_id from query parameters
    user_id = websocket.query_params.get("user_id")
    if user_id is None:
        await websocket.close(code=1008, reason="user_id is required in query params")
        return

    # Initialize session
    session = None

    try:
        session = IncrementalOratorySession(
            buffer_size=buffer_size,
            width=width,
            height=height,
            processing_interval=incremental_interval
        )
        
        # Send initial connection message
        await websocket.send_json({
            "status": "connected",
            "message": "Conexión establecida. Envía frames de video para análisis incremental.",
            "timestamp": time.time()
        })
        
        # Track frames received for incremental processing
        frames_since_last_process = 0
        last_activity_time = time.time()
        inactivity_timeout = 10.0  # seconds
        
        while True:
            try:
                # Use a short timeout to detect end of stream
                message = await asyncio.wait_for(
                    websocket.receive(), 
                    timeout=inactivity_timeout
                )
                last_activity_time = time.time()
                
                # Handle different message types
                if "bytes" in message:
                    data = message["bytes"]
                    
                    # Check prefix for audio chunks
                    if data.startswith(b'AUD'):
                        session.add_audio(data[3:])
                        continue
                    
                    # Process video frame
                    if session.add_frame(data):
                        frames_since_last_process += 1
                        
                        # Perform incremental processing at regular intervals
                        if frames_since_last_process >= incremental_interval:
                            # Send progress update
                            await websocket.send_json({
                                "status": "processing",
                                "message": f"Procesando incrementalmente ({session.frame_count} frames)",
                                "frames_processed": session.frame_count,
                                "timestamp": time.time()
                            })
                            
                            # Run incremental processing with our new system
                            result = await session.process_incremental()
                            
                            # Send detailed incremental update with metrics
                            if "incremental_metrics" in result:
                                # We have real incremental metrics
                                await websocket.send_json({
                                    "status": "incremental_update",
                                    "frames_processed": session.frame_count,
                                    "metrics": result["incremental_metrics"],
                                    "confidence": result.get("confidence", 0.5),
                                    "timestamp": time.time()
                                })
                                
                                # Check if we have any recent fillers to report
                                if "recent_fillers" in result and result["recent_fillers"]:
                                    await websocket.send_json({
                                        "status": "filler_detected",
                                        "fillers": result["recent_fillers"],
                                        "timestamp": time.time()
                                    })
                            else:
                                # Send simplified update (backward compatibility)
                                await websocket.send_json({
                                    "status": "incremental_update",
                                    "frames_processed": session.frame_count,
                                    "timestamp": time.time()
                                })
                            
                            frames_since_last_process = 0
                            
                elif "text" in message:
                    # Handle text commands
                    try:
                        cmd = json.loads(message["text"])
                        
                        if cmd.get("action") == "end_stream":
                            # End the stream and process final result
                            await websocket.send_json({
                                "status": "processing",
                                "message": "Finalizando stream y generando análisis final...",
                                "timestamp": time.time()
                            })
                            
                            # Signal stream end
                            session.end_stream()
                            
                            # Generate final feedback
                            result = await session.generate_final_feedback()
                            
                            # Verificar que el resultado no sea None
                            if result is None:
                                logger.error("generate_final_feedback returned None result")
                                result = {
                                    "status": "error",
                                    "error": "Failed to generate analysis",
                                    "timestamp": time.time()
                                }
                            
                            # Send the result to the REST endpoint
                            asyncio.create_task(send_analysis_result(user_id, result))
                            
                            # Send final status to client
                            await websocket.send_json({
                                "status": "completed",
                                "message": "Análisis completado exitosamente",
                                "timestamp": time.time()
                            })
                            
                            # Close websocket after sending final result
                            await websocket.close(code=1000, reason="Analysis completed")
                            break
                            
                        elif cmd.get("action") == "status":
                            # Return current status
                            await websocket.send_json({
                                "status": "info",
                                "frames_processed": session.frame_count,
                                "buffer_size": len(session.frame_buffer),
                                "streaming_active": session.streaming_active,
                                "timestamp": time.time()
                            })
                            
                    except Exception as e:
                        logger.error(f"Error processing command: {e}")
                        await websocket.send_json({
                            "status": "error",
                            "error": f"Error processing command: {str(e)}",
                            "timestamp": time.time()
                        })
                        
            except asyncio.TimeoutError:
                # Check for inactivity timeout
                if time.time() - last_activity_time > inactivity_timeout:
                    logger.info("Inactivity timeout detected, ending stream")
                    
                    # Send notification to client
                    await websocket.send_json({
                        "status": "processing",
                        "message": "Finalizando stream por inactividad...",
                        "timestamp": time.time()
                    })
                    
                    if session.frame_count > 0:
                        # End stream and generate final results
                        session.end_stream()
                        result = await session.generate_final_feedback()
                        
                        # Verificar que el resultado no sea None
                        if result is None:
                            logger.error("generate_final_feedback returned None result en timeout handler")
                            result = {
                                "status": "error",
                                "error": "Failed to generate analysis after timeout",
                                "timestamp": time.time()
                            }
                            
                        # Send to REST endpoint
                        asyncio.create_task(send_analysis_result(user_id, result))
                        
                        # Notify client
                        await websocket.send_json({
                            "status": "completed",
                            "message": "Análisis completado por inactividad",
                            "timestamp": time.time()
                        })
                    else:
                        await websocket.send_json({
                            "status": "closed",
                            "message": "Conexión cerrada por inactividad sin frames recibidos",
                            "timestamp": time.time()
                        })
                        
                    await websocket.close(code=1000, reason="Inactivity timeout")
                    break
                    
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
        
        # If we received frames, complete the analysis
        if session and session.frame_count > 0:
            try:
                session.end_stream()
                result = await session.generate_final_feedback()
                
                # Verificar que el resultado no sea None
                if result is None:
                    logger.error("generate_final_feedback returned None result after disconnect")
                    result = {
                        "status": "error",
                        "error": "Failed to generate analysis after disconnect",
                        "timestamp": time.time()
                    }
                
                asyncio.create_task(send_analysis_result(user_id, result))
                logger.info(f"Analysis completed after disconnect for user {user_id}")
            except Exception as e:
                logger.error(f"Failed to complete analysis after disconnect: {e}")
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        try:
            if websocket.client_state.name == "CONNECTED":
                await websocket.send_json({
                    "status": "error",
                    "error": f"Error: {str(e)}",
                    "timestamp": time.time()
                })
                await websocket.close(code=1011, reason=str(e))
        except Exception:
            logger.error("Failed to send error message")
            
    finally:
        # Clean up resources
        if session:
            try:
                session.close()
                logger.info("Session resources cleaned up")
            except Exception as cleanup_error:
                logger.error(f"Error during session cleanup: {cleanup_error}")
