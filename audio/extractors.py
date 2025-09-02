"""
Audio extraction functionality for the oratory feedback system.
Provides utilities to extract audio from various media formats.
"""

import os
import tempfile
import numpy as np
import librosa
from pydub import AudioSegment
import logging
import subprocess

from typing import Tuple

logger = logging.getLogger(__name__)

class AudioExtractor:
    """Audio extraction from video files for oratory analysis."""
    
    def __init__(self, sampling_rate: int = 16000):
        """
        Initialize audio extractor.
        
        Args:
            sampling_rate: Sampling frequency for analysis
        """
        self.sampling_rate = sampling_rate
        logger.info(f"AudioExtractor initialized with sampling_rate={sampling_rate}")

    def extract_from_video(self, video_path: str) -> Tuple[np.ndarray, int]:
        """
        Extract audio from a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple with audio array and sample rate
        """
        logger.info(f"Extracting audio from {video_path}")
        
        try:
            # Validate that file exists
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"Video file not found: {video_path}")
                
            # Create temporary file for audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # First try using subprocess directly with ffmpeg for better control

            try:
                logger.info("Attempting direct FFmpeg extraction...")
                # Run FFmpeg command with error output redirected
                cmd = [
                    "ffmpeg", 
                    "-i", video_path,
                    "-vn",  # No video 
                    "-acodec", "pcm_s16le",  # PCM 16-bit little-endian audio
                    "-ar", str(self.sampling_rate),  # Sample rate
                    "-ac", "1",  # Mono
                    "-y",  # Overwrite output files
                    temp_audio_path
                ]
                
                # First check if the file has audio streams
                probe_cmd = ["ffprobe", "-i", video_path, "-show_streams", "-select_streams", "a", "-loglevel", "error"]
                probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
                
                if not probe_result.stdout.strip():
                    logger.warning(f"No audio streams found in video: {video_path}")
                    # Return silent audio of 1 second duration
                    logger.info("Returning silent audio as fallback")
                    silent_audio = np.zeros(self.sampling_rate)
                    return silent_audio, self.sampling_rate
                
                # Execute ffmpeg command
                process = subprocess.run(cmd, capture_output=True, text=True)
                
                if process.returncode != 0:
                    logger.warning(f"FFmpeg direct extraction failed: {process.stderr}")
                    raise Exception(f"FFmpeg error: {process.stderr}")
                    
                # Load the extracted audio with librosa
                audio, sr = librosa.load(temp_audio_path, sr=self.sampling_rate)
                
            except (subprocess.SubprocessError, Exception) as ffmpeg_err:
                logger.warning(f"Direct FFmpeg extraction failed, falling back to pydub: {ffmpeg_err}")
                
                # Fallback to pydub if direct approach fails
                try:
                    # Extract audio with pydub
                    video = AudioSegment.from_file(video_path)
                    video.export(temp_audio_path, format="wav")
                    
                    # Load with librosa
                    audio, sr = librosa.load(temp_audio_path, sr=self.sampling_rate)
                except Exception as pydub_err:
                    logger.error(f"Pydub extraction also failed: {pydub_err}")
                    # If all else fails, return silent audio
                    silent_audio = np.zeros(self.sampling_rate)
                    return silent_audio, self.sampling_rate
            
            # Remove temporary file
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Could not delete temporary audio file: {e}")
            
            if len(audio) == 0:
                logger.warning("Extracted audio is empty, returning silent audio")
                silent_audio = np.zeros(self.sampling_rate)
                return silent_audio, self.sampling_rate
            
            logger.info(f"Audio extracted successfully: {len(audio)/sr:.2f} seconds")
            return audio, sr
            
        except Exception as e:
            logger.error(f"Error extracting audio: {str(e)}")
            # Instead of raising exception, return silent audio
            logger.info("Returning silent audio due to extraction failure")
            silent_audio = np.zeros(self.sampling_rate)
            return silent_audio, self.sampling_rate
            
    def load_audio_file(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple with audio array and sample rate
        """
        logger.info(f"Loading audio from {audio_path}")
        
        try:
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            logger.info(f"Audio loaded successfully: {len(audio)/sr:.2f} seconds")
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio: {str(e)}")
            raise
