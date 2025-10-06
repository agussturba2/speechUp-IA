"""
Video writer module with robust codec fallback support.

Provides a clean interface for writing video frames with automatic codec selection,
error handling, and resource management.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple
from contextlib import contextmanager

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class VideoCodecError(Exception):
    """Raised when all video codecs fail."""
    pass


class VideoWriterError(Exception):
    """Raised when video writing operations fail."""
    pass


class StreamingVideoWriter:
    """
    Streaming video writer with automatic codec fallback.
    
    Manages video file creation, frame writing, and resource cleanup.
    Supports multiple codec fallbacks for cross-platform compatibility.
    """
    
    # Codec fallback chain (codec_fourcc, description)
    CODEC_FALLBACKS: List[Tuple[str, str]] = [
        ('XVID', 'XVID codec - widely compatible'),
        ('MJPG', 'Motion JPEG codec'),
        ('DIB ', 'Uncompressed RGB'),
    ]
    
    MIN_VALID_FILE_SIZE = 100  # bytes
    
    def __init__(
        self,
        width: int,
        height: int,
        fps: float = 30.0,
        output_path: Optional[Path] = None,
    ):
        """
        Initialize video writer.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
            output_path: Output path (if None, creates temp file)
            
        Raises:
            ValueError: If dimensions are invalid
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")
        if fps <= 0:
            raise ValueError(f"Invalid FPS: {fps}")
            
        self.width = width
        self.height = height
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._codec_name: Optional[str] = None
        self._frames_written = 0
        self._is_finalized = False
        
        # Create output file
        if output_path is None:
            fd, self._output_path = tempfile.mkstemp(suffix=".avi")
            os.close(fd)
        else:
            self._output_path = str(output_path)
            
        logger.info(f"VideoWriter initialized: {self._output_path}, {width}x{height} @ {fps} FPS")
        
    @property
    def output_path(self) -> str:
        """Get the output file path."""
        return self._output_path
    
    @property
    def frames_written(self) -> int:
        """Get number of frames written."""
        return self._frames_written
    
    @property
    def is_open(self) -> bool:
        """Check if writer is open."""
        return self._writer is not None and self._writer.isOpened()
    
    def _try_create_writer(self, codec_fourcc: str) -> bool:
        """
        Try to create VideoWriter with specified codec.
        
        Args:
            codec_fourcc: FourCC codec string
            
        Returns:
            True if successful, False otherwise
        """
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
            writer = cv2.VideoWriter(
                self._output_path,
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            
            if writer.isOpened():
                self._writer = writer
                self._codec_name = codec_fourcc
                logger.info(f"VideoWriter created with {codec_fourcc} codec")
                return True
            else:
                logger.debug(f"Failed to open VideoWriter with {codec_fourcc}")
                return False
                
        except Exception as e:
            logger.debug(f"Error creating VideoWriter with {codec_fourcc}: {e}")
            return False
    
    def open(self) -> None:
        """
        Open the video writer with codec fallback.
        
        Raises:
            VideoCodecError: If all codecs fail
        """
        if self._writer is not None:
            logger.warning("VideoWriter already open")
            return
            
        # Try each codec in fallback chain
        for codec_fourcc, description in self.CODEC_FALLBACKS:
            logger.debug(f"Trying {description}")
            if self._try_create_writer(codec_fourcc):
                return
        
        # All codecs failed
        raise VideoCodecError(
            f"Failed to create VideoWriter with any codec. Tried: "
            f"{[c[0] for c in self.CODEC_FALLBACKS]}"
        )
    
    def write_frame(self, frame: np.ndarray) -> None:
        """
        Write a single frame to video.
        
        Args:
            frame: Frame as numpy array (height, width, 3)
            
        Raises:
            VideoWriterError: If writing fails
            ValueError: If frame dimensions don't match
        """
        if self._is_finalized:
            raise VideoWriterError("Cannot write to finalized video")
            
        if frame is None or frame.size == 0:
            raise ValueError("Invalid frame: empty or None")
            
        # Validate frame dimensions
        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            raise ValueError(
                f"Frame size mismatch: expected {self.width}x{self.height}, "
                f"got {w}x{h}"
            )
        
        # Lazy open
        if not self.is_open:
            self.open()
        
        try:
            self._writer.write(frame)
            self._frames_written += 1
        except Exception as e:
            raise VideoWriterError(f"Failed to write frame {self._frames_written}: {e}")
    
    def write_frames(self, frames: List[np.ndarray]) -> int:
        """
        Write multiple frames to video.
        
        Args:
            frames: List of frames
            
        Returns:
            Number of frames successfully written
            
        Raises:
            VideoWriterError: If writing fails critically
        """
        written = 0
        for i, frame in enumerate(frames):
            try:
                self.write_frame(frame)
                written += 1
            except ValueError as e:
                logger.warning(f"Skipping invalid frame {i}: {e}")
            except VideoWriterError as e:
                logger.error(f"Failed to write frame {i}: {e}")
                raise
        
        return written
    
    def finalize(self) -> Path:
        """
        Finalize video and close writer.
        
        Returns:
            Path to the video file
            
        Raises:
            VideoWriterError: If finalization fails or no frames written
        """
        if self._is_finalized:
            logger.warning("Video already finalized")
            return Path(self._output_path)
        
        # Close writer
        if self._writer is not None:
            try:
                self._writer.release()
                logger.info(f"VideoWriter released, {self._frames_written} frames written")
            except Exception as e:
                logger.error(f"Error releasing VideoWriter: {e}")
            finally:
                self._writer = None
        
        # Validate output file
        if not os.path.exists(self._output_path):
            raise VideoWriterError(f"Video file was not created: {self._output_path}")
        
        file_size = os.path.getsize(self._output_path)
        if file_size < self.MIN_VALID_FILE_SIZE:
            raise VideoWriterError(
                f"Video file too small ({file_size} bytes), likely invalid"
            )
        
        if self._frames_written == 0:
            raise VideoWriterError("No frames were written to video")
        
        # Verify video can be read
        cap = cv2.VideoCapture(self._output_path)
        if not cap.isOpened():
            cap.release()
            raise VideoWriterError(f"Video file cannot be opened by OpenCV: {self._output_path}")
        
        actual_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if actual_frame_count == 0:
            raise VideoWriterError("Video has 0 frames according to OpenCV")
        
        logger.info(
            f"Video finalized: {self._output_path}, "
            f"{actual_frame_count} frames, {file_size} bytes"
        )
        
        self._is_finalized = True
        return Path(self._output_path)
    
    def close(self) -> None:
        """Close writer without finalizing (for error cleanup)."""
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception as e:
                logger.error(f"Error closing VideoWriter: {e}")
            finally:
                self._writer = None
    
    def cleanup(self) -> None:
        """Clean up video file and close writer."""
        self.close()
        if os.path.exists(self._output_path):
            try:
                os.remove(self._output_path)
                logger.info(f"Removed video file: {self._output_path}")
            except Exception as e:
                logger.error(f"Failed to remove video file: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            # Error occurred, cleanup
            self.cleanup()
        else:
            # Success, finalize
            try:
                self.finalize()
            except VideoWriterError as e:
                logger.error(f"Failed to finalize video: {e}")
                self.cleanup()
                raise
        return False
    
    def __del__(self):
        """Destructor - ensure resources are released."""
        self.close()


@contextmanager
def create_video_from_frames(
    frames: List[np.ndarray],
    fps: float = 30.0,
    output_path: Optional[Path] = None
) -> Path:
    """
    Context manager to create video from frames.
    
    Args:
        frames: List of video frames
        fps: Frames per second
        output_path: Optional output path
        
    Yields:
        Path to created video file
        
    Example:
        with create_video_from_frames(frames, fps=30) as video_path:
            process(video_path)
    """
    if not frames:
        raise ValueError("No frames provided")
    
    # Get dimensions from first valid frame
    test_frame = None
    for frame in frames:
        if frame is not None and frame.size > 0:
            test_frame = frame
            break
    
    if test_frame is None:
        raise ValueError("No valid frames found")
    
    h, w = test_frame.shape[:2]
    
    writer = StreamingVideoWriter(w, h, fps, output_path)
    try:
        writer.write_frames(frames)
        video_path = writer.finalize()
        yield video_path
    finally:
        writer.close()
