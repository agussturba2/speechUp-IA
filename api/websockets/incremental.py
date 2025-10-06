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
from typing import Dict, Any, List, Optional, Tuple, TypedDict
import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import asyncio
import httpx
from video.pipeline import run_analysis_pipeline
from video.realtime import decode_frame_data
from .oratory import process_pipeline_results, send_analysis_result

# Importaciones para análisis de audio real
from audio.asr import transcribe_wav
from audio.speech import SpeechSegmenter
from audio.text_metrics import detect_spanish_fillers
import scipy.io.wavfile as wav

logger = logging.getLogger(__name__)


# Type definitions for better type safety
class AudioAnalysisResult(TypedDict, total=False):
    """Result from audio buffer processing."""
    duration_sec: float
    speech_detected: bool
    words: List[Dict[str, Any]]
    fillers: List[Dict[str, Any]]
    pauses: List[Dict[str, Any]]
    pause_analysis: Dict[str, Any]


class FrameAnalysisResult(TypedDict, total=False):
    """Result from frame processing."""
    frames_analyzed: int
    frames_with_face: int
    expressions: List[Dict[str, Any]]
    gestures: List[Dict[str, Any]]
    posture: List[Dict[str, Any]]


class IncrementalMetrics(TypedDict):
    """Metrics computed from incremental analysis."""
    wpm: float
    fillers_per_min: float
    gesture_rate: float
    expression_variability: float


class AnalysisResult(TypedDict, total=False):
    """Complete analysis result structure."""
    status: str
    ok: bool
    error: str
    quality: Dict[str, Any]
    media: Dict[str, Any]
    scores: Dict[str, int]
    verbal: Dict[str, Any]
    events: List[Dict[str, Any]]
    timestamp: float


# Audio processing constants
AUDIO_SAMPLE_RATE = 16000
AUDIO_BYTES_PER_SAMPLE = 2
AUDIO_CHANNELS = 1
AUDIO_BYTES_PER_MS = (AUDIO_SAMPLE_RATE * AUDIO_BYTES_PER_SAMPLE) // 1000  # 32 bytes/ms
AUDIO_NORMALIZATION_FACTOR = 32768.0

# Video constants
DEFAULT_FPS = 30.0
DEFAULT_VIDEO_CODEC = 'XVID'
VIDEO_FALLBACK_CODECS = [('XVID', 'XVID codec - widely compatible'), ('MJPG', 'Motion JPEG codec'), ('DIB ', 'Uncompressed RGB')]

# Buffer limits to prevent excessive memory usage
MAX_ALL_FRAMES_BUFFER = 3000  # Max frames to store (100 seconds @ 30fps)
MAX_AUDIO_BUFFER_BYTES = 16000 * 2 * 60  # Max 60 seconds of audio

# Processing constants
MIN_AUDIO_FOR_TRANSCRIPTION = AUDIO_SAMPLE_RATE * AUDIO_BYTES_PER_SAMPLE  # 1 second
AUDIO_OVERLAP_SECONDS = 0.5
MIN_VIDEO_FILE_SIZE = 100  # bytes


def _create_video_writer(video_path: str, width: int, height: int, fps: float = DEFAULT_FPS) -> Tuple[Optional[cv2.VideoWriter], Optional[str]]:
    """
    Create a video writer with fallback codec support.
    
    Args:
        video_path: Path to save the video file
        width: Video width
        height: Video height
        fps: Frames per second
        
    Returns:
        Tuple of (VideoWriter, codec_name) or (None, None) if all codecs failed
    """
    for codec_name, desc in VIDEO_FALLBACK_CODECS:
        try:
            logger.debug(f"Attempting to create VideoWriter with {desc}")
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            
            writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            
            if writer.isOpened():
                logger.info(f"Successfully created VideoWriter with {codec_name} codec")
                return writer, codec_name
            else:
                logger.debug(f"Failed to open VideoWriter with {codec_name}")
                
        except Exception as e:
            logger.debug(f"Error with {codec_name} codec: {e}")
    
    logger.error("All video codecs failed")
    return None, None


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

        # Initialize frame buffer with size limit
        self.frame_buffer = deque(maxlen=buffer_size)
        self.audio_buffer = bytearray()
        self.audio_buffer_duration = 0.0  # Track duration of audio in buffer (seconds)
        
        # Store processed frames for final analysis (with limit to prevent OOM)
        self.all_frames = deque(maxlen=MAX_ALL_FRAMES_BUFFER)
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
        
        # NEW: Buffers for incremental analysis (with limits)
        self.analysis_frame_buffer = []
        self.analysis_audio_buffer = bytearray()
        self.analysis_audio_buffer_max = MAX_AUDIO_BUFFER_BYTES
        self.last_processed_frame_index = 0
        
        # Analysis state
        self.analysis_in_progress = False
        self.last_analysis_time = 0
        self.streaming_active = True

        # Thread pool for async processing
        self.executor = ThreadPoolExecutor(max_workers=1)
        
        # NEW: Flag to enable true incremental processing
        self.enable_incremental_processing = True
        
        # Speech segmenter for VAD
        self.speech_segmenter = SpeechSegmenter(vad_mode=2)

        # Log GPU status and initialization
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
        Add audio data to the buffer with size limiting.
        
        Args:
            audio_data: Raw audio data
        """
        # Calculate duration based on sample rate and format
        duration_ms = len(audio_data) / float(AUDIO_BYTES_PER_MS)
        duration_sec = duration_ms / 1000.0
        
        self.audio_buffer.extend(audio_data)
        self.audio_buffer_duration += duration_sec
        
        # Limit main audio buffer to prevent excessive memory usage
        if len(self.audio_buffer) > MAX_AUDIO_BUFFER_BYTES:
            bytes_to_remove = len(self.audio_buffer) - MAX_AUDIO_BUFFER_BYTES
            self.audio_buffer = bytearray(self.audio_buffer[bytes_to_remove:])
            # Adjust duration accordingly
            removed_duration = bytes_to_remove / float(AUDIO_BYTES_PER_MS) / 1000.0
            self.audio_buffer_duration = max(0, self.audio_buffer_duration - removed_duration)
        
        # Also add to analysis buffer with limit
        self.analysis_audio_buffer.extend(audio_data)
        if len(self.analysis_audio_buffer) > self.analysis_audio_buffer_max:
            bytes_to_remove = len(self.analysis_audio_buffer) - self.analysis_audio_buffer_max
            self.analysis_audio_buffer = bytearray(self.analysis_audio_buffer[bytes_to_remove:])

    def end_stream(self) -> None:
        """
        Signal that the stream has ended and final processing can begin.
        Optimized to avoid rewriting all frames if video already exists.
        """
        logger.info("Stream ending - finalizing data for analysis")
        self.streaming_active = False
        
        # If we already have a video writer and path, just finalize it
        if self.video_writer is not None and self.video_path:
            logger.info("Finalizing existing video file")
            self._cleanup_video_resources()
            
            # Verify the video file is valid
            if os.path.exists(self.video_path) and os.path.getsize(self.video_path) > MIN_VIDEO_FILE_SIZE:
                logger.info(f"Video file ready: {self.video_path} ({os.path.getsize(self.video_path)} bytes)")
                return
            else:
                logger.warning("Existing video file is invalid, will recreate")
                self.video_path = None
        
        # Only recreate video if we don't have a valid one
        if self.all_frames and not self.video_path:
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
                
                # Create video writer using helper function
                self.video_writer, codec_name = _create_video_writer(self.video_path, w, h, DEFAULT_FPS)
                
                success = False
                if self.video_writer is not None:
                    try:
                        logger.info(f"Writing {len(self.all_frames)} frames to final video")
                        
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
                        if os.path.exists(self.video_path) and os.path.getsize(self.video_path) > MIN_VIDEO_FILE_SIZE:
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
                                else:
                                    logger.error(f"Final video has no frames according to OpenCV")
                            else:
                                logger.error(f"Final video cannot be opened by OpenCV")
                        else:
                            logger.error(f"Final video file is empty or too small")
                            
                    except Exception as e:
                        logger.error(f"Error writing final video: {e}")
                
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
                
                # Create a temporary file for the video
                fd, self.video_path = tempfile.mkstemp(suffix=".avi")
                os.close(fd)
                logger.info(f"Created temporary file: {self.video_path}")
                
                # Create video writer using helper function
                self.video_writer, codec_name = _create_video_writer(self.video_path, w, h, DEFAULT_FPS)
                
                if self.video_writer is None:
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
                    if os.path.exists(self.video_path) and os.path.getsize(self.video_path) > MIN_VIDEO_FILE_SIZE:
                        logger.info(f"Video file created successfully: {self.video_path} ({os.path.getsize(self.video_path)} bytes)")
                        
                        # Reopen the writer for possible future writes
                        fourcc = cv2.VideoWriter_fourcc(*codec_name)
                        self.video_writer = cv2.VideoWriter(
                            self.video_path,
                            fourcc,
                            DEFAULT_FPS,
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
            
            # Initialize results
            frame_results = {
                "frames_analyzed": frames_count,
                "frames_with_face": 0,
                "expressions": [],
                "gestures": [],
                "posture": []
            }
            
            # For incremental processing, we do lightweight checks
            # Full analysis happens in the final pipeline
            # Here we can do basic face detection or skip for performance
            
            # Simple frame quality check - count non-empty frames
            valid_frames = 0
            for frame in new_frames:
                if frame is not None and frame.size > 0:
                    valid_frames += 1
                    # Could add basic face detection here if needed
                    # For now, assume frames with data are valid
            
            frame_results["frames_with_face"] = valid_frames
            
            # Update the last processed frame index
            self.last_processed_frame_index += frames_count
            
            logger.debug(f"Lightweight frame analysis: {valid_frames}/{frames_count} valid frames")
            
            return frame_results
            
        except Exception as e:
            logger.error(f"Error processing new frames: {e}", exc_info=True)
            return {}
    
    async def _process_audio_buffer(self) -> Dict[str, Any]:
        """
        Process accumulated audio buffer with REAL speech analysis using Whisper and VAD.
        This method is async to avoid blocking the event loop during transcription.
        
        Returns:
            Dict with audio analysis results
        """
        if len(self.analysis_audio_buffer) < MIN_AUDIO_FOR_TRANSCRIPTION:
            return {}
            
        try:
            # Convert buffer to numpy array (16-bit PCM to float32)
            audio_np = np.frombuffer(
                self.analysis_audio_buffer, 
                dtype=np.int16
            ).astype(np.float32) / AUDIO_NORMALIZATION_FACTOR
            
            # Calculate audio duration
            audio_duration = len(audio_np) / float(AUDIO_SAMPLE_RATE)
            logger.info(f"Processing {audio_duration:.2f} seconds of audio with REAL analysis")
            
            # 1. VAD - Voice Activity Detection and pause analysis
            segments = self.speech_segmenter.get_speech_segments(audio_np, AUDIO_SAMPLE_RATE)
            pause_analysis = self.speech_segmenter.analyze_pauses(segments)
            
            speech_detected = pause_analysis.get("speech_percent", 0) > 10
            
            # Extract pauses from segments
            pauses = []
            for is_speech, start, end in segments:
                if not is_speech:  # Silence segment
                    pauses.append({
                        "start_time": start,
                        "duration": end - start
                    })
            
            # Initialize results
            audio_results = {
                "duration_sec": audio_duration,
                "speech_detected": speech_detected,
                "words": [],
                "fillers": [],
                "pauses": pauses,
                "pause_analysis": pause_analysis
            }
            
            # 2. Transcription with Whisper (only if speech detected)
            if speech_detected and len(audio_np) > 0:
                try:
                    # Save audio to temporary WAV file
                    temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
                    os.close(temp_wav_fd)
                    
                    # Write WAV file
                    wav.write(temp_wav_path, AUDIO_SAMPLE_RATE, (audio_np * AUDIO_NORMALIZATION_FACTOR).astype(np.int16))
                    
                    # Transcribe using Whisper in executor to avoid blocking event loop
                    logger.info(f"Transcribing audio segment with Whisper (async)")
                    loop = asyncio.get_event_loop()
                    transcript_result = await loop.run_in_executor(
                        self.executor, 
                        transcribe_wav, 
                        temp_wav_path, 
                        "es"
                    )
                    
                    # Clean up temp file
                    try:
                        os.unlink(temp_wav_path)
                    except Exception:
                        pass
                    
                    # Check if transcription was successful
                    if transcript_result.get("status") == "success":
                        text = transcript_result.get("text", "").strip()
                        logger.info(f"Transcription: '{text[:100]}...'")
                        
                        # 3. Count words
                        if text:
                            words = text.split()
                            audio_results["words"] = [
                                {"word": w, "index": i} 
                                for i, w in enumerate(words)
                            ]
                            
                            # 4. Detect filler words using real analysis
                            try:
                                filler_analysis = detect_spanish_fillers(text)
                                filler_counts = filler_analysis.get("filler_counts", {})
                                
                                # Extract filler instances
                                fillers = []
                                for filler_type, count in filler_counts.items():
                                    # Distribute fillers approximately across the audio duration
                                    for i in range(count):
                                        fillers.append({
                                            "type": filler_type,
                                            "time": (i + 0.5) * (audio_duration / max(1, count)),
                                            "duration": 0.3
                                        })
                                
                                audio_results["fillers"] = fillers
                                logger.info(f"Detected {len(fillers)} filler words: {filler_counts}")
                                
                            except Exception as filler_error:
                                logger.error(f"Error analyzing fillers: {filler_error}")
                                audio_results["fillers"] = []
                    else:
                        logger.warning(f"Transcription failed: {transcript_result.get('error', 'Unknown error')}")
                        
                except Exception as transcription_error:
                    logger.error(f"Error in transcription: {transcription_error}", exc_info=True)
            
            # Clear the analysis audio buffer for next iteration
            # Keep some overlap for continuity
            overlap_bytes = int(AUDIO_OVERLAP_SECONDS * AUDIO_SAMPLE_RATE * AUDIO_BYTES_PER_SAMPLE)
            if len(self.analysis_audio_buffer) > overlap_bytes:
                self.analysis_audio_buffer = bytearray(self.analysis_audio_buffer[-overlap_bytes:])
            else:
                self.analysis_audio_buffer = bytearray()
            
            return audio_results
            
        except Exception as e:
            logger.error(f"Error processing audio buffer: {e}", exc_info=True)
            # Clear buffer on error
            self.analysis_audio_buffer = bytearray()
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
                
                # Process audio buffer if we have enough data (async to avoid blocking)
                audio_results = await self._process_audio_buffer()
                
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

    
            
    def _create_result_from_incremental_data(self) -> AnalysisResult:
        """
        Create a result structure using only the incrementally collected data.
        Used as a fallback when the final pipeline fails.
        
        Returns:
            Dict with feedback data from incremental analysis
        """
        try:
            # Get accumulated state with defaults
            acc_state = self.accumulated_state or {}
            metrics = acc_state.get("metrics", {})
            verbal = acc_state.get("verbal", {})
            nonverbal = acc_state.get("nonverbal", {})
            
            # Calculate total duration
            start_time = acc_state.get("start_time", time.time() - 10)
            duration_sec = time.time() - start_time
            
            # Extract metrics with defaults
            wpm = metrics.get("wpm", 0.0)
            fillers_per_min = metrics.get("fillers_per_min", 0.0)
            pause_rate = metrics.get("pause_rate", 0.0)
            gesture_rate = metrics.get("gesture_rate", 0.0)
            expression_variability = metrics.get("expression_variability", 0.0)
            
            # Get buffer/frame counts safely
            audio_buffer_length = len(self.audio_buffer) if hasattr(self, "audio_buffer") else 0
            all_frames_length = len(self.all_frames) if hasattr(self, "all_frames") else 0
            partial_results_length = len(self.partial_results) if hasattr(self, "partial_results") else 0
            frames_processed = acc_state.get("frames_processed", 0)
            audio_processed_sec = acc_state.get("audio_processed_sec", 0.0)
            
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
            
            # Convert accumulated fillers to events
            filler_instances = verbal.get("filler_instances", [])
            for filler in filler_instances:
                if isinstance(filler, dict) and "time" in filler and "type" in filler:
                    result["events"].append({
                        "time": filler["time"],
                        "kind": "filler",
                        "label": filler["type"],
                        "duration": filler.get("duration", 0.3)
                    })
                    
                    # Count fillers by type
                    filler_type = filler["type"]
                    result["verbal"]["filler_counts"][filler_type] = result["verbal"]["filler_counts"].get(filler_type, 0) + 1
            
            # Convert accumulated gestures to events
            gestures = nonverbal.get("gestures", [])
            for gesture in gestures:
                if isinstance(gesture, dict) and "type" in gesture:
                    result["events"].append({
                        "time": gesture.get("frame_index", 0) / DEFAULT_FPS,
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
        
    def _enhance_result_with_incremental_data(self, result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Enhance the final analysis result with data from incremental processing.
        
        Args:
            result: Result from final analysis pipeline
            
        Returns:
            Enhanced result with incremental data
        """
        # Ensure result exists
        if result is None:
            logger.warning("Received None result, creating empty result")
            result = {"status": "error", "error": "No analysis result was generated", "timestamp": time.time()}
        
        # Add debug info about incremental processing
        result.setdefault("quality", {}).setdefault("debug", {})
        debug = result["quality"]["debug"]
        debug["incremental_processing"] = True
        debug["incremental_steps"] = len(self.partial_results) if hasattr(self, "partial_results") else 0
        debug["total_frames_received"] = len(self.all_frames) if hasattr(self, "all_frames") else 0
        
        # Add missing incremental fillers to events
        try:
            # Get existing filler times from result
            result_filler_times = {
                round(event["time"], 1)
                for event in result.get("events", [])
                if event.get("kind") == "filler" and "time" in event
            }
            
            # Add incremental fillers that are missing
            acc_state = self.accumulated_state or {}
            verbal = acc_state.get("verbal", {})
            filler_instances = verbal.get("filler_instances", [])
            
            result.setdefault("events", [])
            for filler in filler_instances:
                if isinstance(filler, dict) and "time" in filler:
                    rounded_time = round(filler["time"], 1)
                    if rounded_time not in result_filler_times:
                        result["events"].append({
                            "time": filler["time"],
                            "kind": "filler",
                            "label": filler.get("type", "um"),
                            "duration": filler.get("duration", 0.3),
                            "source": "incremental"
                        })
        except Exception as e:
            logger.error(f"Error processing fillers: {e}")
        
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
                            
                            # Send the result to the REST endpoint and wait for completion
                            try:
                                await send_analysis_result(user_id, result)
                                # Send final status to client
                                await websocket.send_json({
                                    "status": "completed",
                                    "message": "Análisis completado exitosamente",
                                    "timestamp": time.time()
                                })
                            except Exception as send_error:
                                logger.error(f"Failed to save analysis to backend: {send_error}")
                                await websocket.send_json({
                                    "status": "completed_with_warning",
                                    "message": "Análisis completado pero no se pudo guardar en el servidor",
                                    "error": str(send_error),
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
                            
                        # Send to REST endpoint and wait for completion
                        try:
                            await send_analysis_result(user_id, result)
                            # Notify client
                            await websocket.send_json({
                                "status": "completed",
                                "message": "Análisis completado por inactividad",
                                "timestamp": time.time()
                            })
                        except Exception as send_error:
                            logger.error(f"Failed to save analysis to backend after timeout: {send_error}")
                            await websocket.send_json({
                                "status": "completed_with_warning",
                                "message": "Análisis completado por inactividad pero no se pudo guardar",
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
                
                try:
                    await send_analysis_result(user_id, result)
                    logger.info(f"Analysis completed and saved after disconnect for user {user_id}")
                except Exception as send_error:
                    logger.error(f"Failed to save analysis after disconnect: {send_error}")
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
