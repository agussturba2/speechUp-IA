"""
Video buffer management component for incremental oratory analysis.

Handles frame buffering, video writing, and frame tracking.
"""

import logging
from typing import List, Optional, Deque
from collections import deque
from pathlib import Path

import numpy as np

from video.video_writer import StreamingVideoWriter, VideoCodecError, VideoWriterError
from video.realtime import decode_frame_data
from .config import config as incremental_config

logger = logging.getLogger(__name__)


class VideoBufferManager:
    """
    Manages video frames and writing to video files.
    
    Responsibilities:
    - Frame buffering with size limits
    - Frame decoding
    - Video file creation and writing
    - Frame tracking for incremental analysis
    """
    
    def __init__(
        self,
        width: int,
        height: int,
        fps: float = None,
        buffer_size: int = None,
        max_frames: int = None
    ):
        """
        Initialize video buffer manager.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second for video output
            buffer_size: Size of rolling buffer
            max_frames: Maximum frames to store (prevents OOM)
        """
        self.width = width
        self.height = height
        self.fps = fps or incremental_config.default_fps
        
        # Buffer configuration
        buffer_size = buffer_size or incremental_config.frame_buffer_size
        max_frames = max_frames or incremental_config.max_all_frames_buffer
        
        # Frame buffers
        self.frame_buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self.all_frames: Deque[np.ndarray] = deque(maxlen=max_frames)
        self.frame_count = 0
        
        # Video writer
        self._video_writer: Optional[StreamingVideoWriter] = None
        self._video_path: Optional[Path] = None
        
        logger.info(
            f"VideoBufferManager initialized: {width}x{height} @ {self.fps} FPS, "
            f"buffer_size={buffer_size}, max_frames={max_frames}"
        )
    
    def add_frame(self, frame_data: bytes) -> bool:
        """
        Add a video frame to buffers.
        
        Args:
            frame_data: Raw frame data to decode
            
        Returns:
            True if successful, False otherwise
        """
        # Decode frame
        frame = decode_frame_data(frame_data, self.width, self.height)
        if frame is None:
            logger.warning("Failed to decode frame data")
            return False
        
        # Add to buffers
        self.frame_buffer.append(frame)
        self.all_frames.append(frame)
        self.frame_count += 1
        
        # Write to video if active
        if self._video_writer is not None:
            try:
                self._video_writer.write_frame(frame)
            except (VideoWriterError, ValueError) as e:
                logger.error(f"Failed to write frame to video: {e}")
                # Don't fail the add operation
        
        return True
    
    def get_recent_frames(self, count: int = None) -> List[np.ndarray]:
        """
        Get recent frames from buffer.
        
        Args:
            count: Number of frames to get (all if None)
            
        Returns:
            List of recent frames
        """
        if count is None:
            return list(self.frame_buffer)
        return list(self.frame_buffer)[-count:]
    
    def get_all_frames(self) -> List[np.ndarray]:
        """Get all stored frames."""
        return list(self.all_frames)
    
    def start_video_writing(self) -> None:
        """
        Start writing frames to video file.
        Creates a StreamingVideoWriter for incremental writing.
        """
        if self._video_writer is not None:
            logger.warning("Video writer already active")
            return
        
        try:
            self._video_writer = StreamingVideoWriter(
                width=self.width,
                height=self.height,
                fps=self.fps
            )
            self._video_writer.open()
            logger.info(f"Started video writing: {self._video_writer.output_path}")
            
        except VideoCodecError as e:
            logger.error(f"Failed to start video writing: {e}")
            self._video_writer = None
    
    def finalize_video(self) -> Optional[Path]:
        """
        Finalize video file and return path.
        
        Returns:
            Path to video file, or None if failed
        """
        if self._video_writer is not None:
            try:
                video_path = self._video_writer.finalize()
                self._video_path = video_path
                logger.info(f"Video finalized: {video_path}")
                return video_path
            except VideoWriterError as e:
                logger.error(f"Failed to finalize video: {e}")
                self._cleanup_video_writer()
                return None
        
        # No active writer, create video from stored frames
        return self._create_video_from_frames()
    
    def _create_video_from_frames(self) -> Optional[Path]:
        """
        Create video file from stored frames.
        
        Returns:
            Path to video file, or None if failed
        """
        if not self.all_frames:
            logger.warning("No frames available to create video")
            return None
        
        # Find valid frame for dimensions
        test_frame = None
        for frame in self.all_frames:
            if frame is not None and frame.size > 0:
                test_frame = frame
                break
        
        if test_frame is None:
            logger.error("No valid frames found")
            return None
        
        h, w = test_frame.shape[:2]
        logger.info(f"Creating video from {len(self.all_frames)} frames ({w}x{h})")
        
        try:
            writer = StreamingVideoWriter(w, h, self.fps)
            frames_written = writer.write_frames(list(self.all_frames))
            video_path = writer.finalize()
            
            self._video_path = video_path
            logger.info(f"Video created: {video_path} ({frames_written} frames)")
            return video_path
            
        except (VideoCodecError, VideoWriterError) as e:
            logger.error(f"Failed to create video: {e}")
            return None
    
    def get_video_path(self) -> Optional[Path]:
        """Get path to video file if available."""
        return self._video_path
    
    def _cleanup_video_writer(self) -> None:
        """Clean up video writer."""
        if self._video_writer is not None:
            try:
                self._video_writer.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up video writer: {e}")
            finally:
                self._video_writer = None
    
    def reset(self) -> None:
        """Reset buffer manager state."""
        self.frame_buffer.clear()
        self.all_frames.clear()
        self.frame_count = 0
        self._cleanup_video_writer()
        self._video_path = None
        logger.debug("VideoBufferManager reset")
    
    def close(self) -> None:
        """Clean up resources."""
        self._cleanup_video_writer()
        self.frame_buffer.clear()
        self.all_frames.clear()
        logger.info("VideoBufferManager closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
