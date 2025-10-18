"""
Audio processing component for incremental oratory analysis.

Handles audio buffering, VAD, transcription, and filler detection.
"""

import logging
import os
import tempfile
import asyncio
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.io.wavfile as wav

from audio.asr import transcribe_wav
from audio.speech import SpeechSegmenter
from audio.text_metrics import detect_spanish_fillers
from .config import config as incremental_config
from .models import AudioAnalysisResult, Word, FillerInstance, PauseSegment, PauseAnalysis

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Processes audio data incrementally with VAD, transcription, and filler detection.
    
    Responsibilities:
    - Audio buffer management
    - Voice Activity Detection (VAD)
    - Speech transcription with Whisper
    - Filler word detection
    - Position tracking for incremental processing
    """
    
    def __init__(
        self,
        sample_rate: int = None,
        vad_mode: int = None,
        speech_threshold: float = None,
        executor: Optional[ThreadPoolExecutor] = None
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Audio sample rate (Hz)
            vad_mode: WebRTC VAD aggressiveness (0-3)
            speech_threshold: Minimum speech percentage to detect speech
            executor: ThreadPoolExecutor for async transcription
        """
        self.sample_rate = sample_rate or incremental_config.audio_sample_rate
        self.bytes_per_sample = incremental_config.audio_bytes_per_sample
        self.normalization_factor = incremental_config.audio_normalization_factor
        self.speech_threshold = speech_threshold or incremental_config.speech_detection_threshold
        
        # Buffer management
        self.audio_buffer = bytearray()
        self.audio_buffer_duration = 0.0
        self._buffer_offset = 0  # Track removed bytes
        self._last_processed_position = 0  # Track processing position
        self._partial_sample = bytearray()
        
        # Processing components
        vad_mode = vad_mode or incremental_config.vad_mode
        self.speech_segmenter = SpeechSegmenter(vad_mode=vad_mode)
        
        # Executor for async transcription
        self._executor = executor or ThreadPoolExecutor(max_workers=1)
        self._owns_executor = executor is None
        
        logger.info(
            f"AudioProcessor initialized: sample_rate={self.sample_rate}Hz, "
            f"vad_mode={vad_mode}, speech_threshold={self.speech_threshold}%"
        )
    
    def add_audio(self, audio_data: bytes) -> None:
        """
        Add audio data to the buffer.
        
        Args:
            audio_data: Raw PCM audio data (16-bit, mono)
        """
        if not audio_data:
            return
        
        # Log first audio chunk to diagnose format issues
        if len(self.audio_buffer) == 0 and len(audio_data) >= 100:
            import numpy as np
            sample_data = np.frombuffer(audio_data[:100], dtype=np.int16)
            dc_offset = sample_data.mean()
            logger.error(f"[AUDIO RECEIVED FROM CLIENT] first_chunk_size={len(audio_data)} bytes, first_50_samples: min={sample_data.min()}, max={sample_data.max()}, mean={dc_offset:.2f}, std={sample_data.std():.2f}")
            if abs(dc_offset) > 1000:
                logger.error(f"⚠️ LARGE DC OFFSET DETECTED: {dc_offset:.0f} - Audio may need centering")

        # Prepend any pending partial sample
        if self._partial_sample:
            audio_data = bytes(self._partial_sample) + audio_data
            self._partial_sample.clear()

        # Ensure data aligns to sample boundaries
        remainder = len(audio_data) % self.bytes_per_sample
        if remainder:
            # Store leftover bytes for the next chunk
            self._partial_sample.extend(audio_data[-remainder:])
            audio_data = audio_data[:-remainder]

        if not audio_data:
            return

        # Calculate duration for aligned data
        bytes_per_ms = self.sample_rate * self.bytes_per_sample // 1000
        duration_ms = len(audio_data) / float(bytes_per_ms)
        duration_sec = duration_ms / 1000.0

        self.audio_buffer.extend(audio_data)
        self.audio_buffer_duration += duration_sec
        
        # Limit buffer size
        max_buffer_bytes = incremental_config.max_audio_buffer_bytes
        if len(self.audio_buffer) > max_buffer_bytes:
            bytes_to_remove = len(self.audio_buffer) - max_buffer_bytes
            # Align removal to sample boundaries
            remainder = bytes_to_remove % self.bytes_per_sample
            if remainder:
                bytes_to_remove -= remainder
            if bytes_to_remove > 0:
                self._buffer_offset += bytes_to_remove
                self.audio_buffer = bytearray(self.audio_buffer[bytes_to_remove:])

                # Adjust duration
                removed_duration = bytes_to_remove / float(bytes_per_ms) / 1000.0
                self.audio_buffer_duration = max(0, self.audio_buffer_duration - removed_duration)

                logger.debug(f"Audio buffer trimmed: removed {bytes_to_remove} bytes")
    
    def get_unprocessed_bytes(self) -> int:
        """Get number of unprocessed audio bytes."""
        return len(self.audio_buffer) - self._last_processed_position
    
    def get_audio_segment(self, include_overlap: bool = True) -> bytes:
        """
        Get audio segment for processing.
        
        Args:
            include_overlap: Include overlap for continuity
            
        Returns:
            Audio segment as bytes
        """
        if include_overlap:
            overlap_bytes = incremental_config.audio_overlap_bytes
            start_pos = max(0, self._last_processed_position - overlap_bytes)
        else:
            start_pos = self._last_processed_position
        
        return bytes(self.audio_buffer[start_pos:])

    def export_buffer_to_wav(self, output_path: Optional[str] = None) -> Optional[str]:
        """Persist the accumulated audio buffer to a WAV file.

        Args:
            output_path: Optional destination path. If omitted, a temporary file is created.

        Returns:
            Path to the written WAV file, or None if the buffer is empty or the export fails.
        """
        if not self.audio_buffer and not self._partial_sample:
            logger.debug("Audio buffer empty, skipping export")
            return None

        path = output_path
        fd = None
        try:
            if path is None:
                fd, path = tempfile.mkstemp(suffix=".wav")
                os.close(fd)

            buffer_bytes = bytes(self.audio_buffer)
            if self._partial_sample:
                logger.debug("Discarding %d pending partial bytes before export", len(self._partial_sample))
            # Ensure alignment
            remainder = len(buffer_bytes) % self.bytes_per_sample
            if remainder:
                logger.debug("Discarding %d trailing bytes to align samples during export", remainder)
                buffer_bytes = buffer_bytes[:-remainder]

            audio_np = np.frombuffer(buffer_bytes, dtype=np.int16)
            if audio_np.size == 0:
                logger.debug("Audio buffer contains no samples after conversion")
                if output_path is None and path and os.path.exists(path):
                    os.remove(path)
                return None

            wav.write(path, self.sample_rate, audio_np)
            self._partial_sample.clear()
            logger.info(f"Exported audio buffer to {path}")
            return path

        except Exception as e:
            logger.error(f"Failed to export audio buffer: {e}", exc_info=True)
            if output_path is None and path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
            return None


    async def process_audio(self, language: str = "es") -> Optional[AudioAnalysisResult]:
        """
        Process accumulated audio with VAD, transcription, and filler detection.
        
        Args:
            language: Language code for transcription
            
        Returns:
            AudioAnalysisResult with analysis data, or None if insufficient audio
        """
        # Check if we have enough audio
        unprocessed_bytes = self.get_unprocessed_bytes()
        min_bytes = incremental_config.min_audio_for_transcription_bytes
        
        if unprocessed_bytes < min_bytes:
            return None
        
        try:
            # Get audio segment with overlap
            audio_segment = self.get_audio_segment(include_overlap=True)
            remainder = len(audio_segment) % self.bytes_per_sample
            if remainder:
                logger.debug("Trimming %d trailing bytes from audio segment to preserve sample alignment", remainder)
                audio_segment = audio_segment[:-remainder]
            if not audio_segment:
                logger.debug("Audio segment empty after alignment trimming")
                return None
            
            # Convert to numpy array
            audio_np = np.frombuffer(
                audio_segment,
                dtype=np.int16
            ).astype(np.float32) / self.normalization_factor
            
            # Remove DC offset if present (center around 0)
            dc_offset = audio_np.mean()
            if abs(dc_offset) > 0.01:  # Threshold for normalized audio
                logger.error(f"[DC OFFSET CORRECTION] Removing DC offset: {dc_offset:.4f}")
                audio_np = audio_np - dc_offset
            
            audio_duration = len(audio_np) / float(self.sample_rate)
            logger.info(f"Processing {audio_duration:.2f}s of audio")
            
            # 1. Voice Activity Detection
            segments = self.speech_segmenter.get_speech_segments(audio_np, self.sample_rate)
            pause_analysis_dict = self.speech_segmenter.analyze_pauses(segments)
            
            speech_detected = pause_analysis_dict.get("speech_percent", 0) > self.speech_threshold
            
            # Extract pauses
            pauses = []
            for is_speech, start, end in segments:
                if not is_speech:
                    pauses.append(PauseSegment(
                        start_time=start,
                        duration=end - start
                    ))
            
            # Create pause analysis model
            pause_analysis = PauseAnalysis(
                speech_percent=pause_analysis_dict["speech_percent"],
                avg_speech_segment=pause_analysis_dict["avg_speech_segment"],
                avg_pause_length=pause_analysis_dict["avg_pause_length"],
                pause_frequency=pause_analysis_dict["pause_frequency"]
            )
            
            # Initialize result
            words: List[Word] = []
            fillers: List[FillerInstance] = []
            
            # 2. Transcription (only if speech detected)
            if speech_detected and len(audio_np) > 0:
                words, fillers = await self._transcribe_and_analyze(
                    audio_np, 
                    audio_duration, 
                    language
                )
            
            # Update processing position
            self._last_processed_position = len(self.audio_buffer)
            
            return AudioAnalysisResult(
                duration_sec=audio_duration,
                speech_detected=speech_detected,
                words=words,
                fillers=fillers,
                pauses=pauses,
                pause_analysis=pause_analysis
            )
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}", exc_info=True)
            # Skip this segment
            self._last_processed_position = len(self.audio_buffer)
            return None
    
    async def _transcribe_and_analyze(
        self,
        audio_np: np.ndarray,
        audio_duration: float,
        language: str
    ) -> tuple[List[Word], List[FillerInstance]]:
        """
        Transcribe audio and analyze for fillers.
        
        Args:
            audio_np: Audio as numpy array
            audio_duration: Duration in seconds
            language: Language code
            
        Returns:
            Tuple of (words, fillers)
        """
        words: List[Word] = []
        fillers: List[FillerInstance] = []
        
        try:
            # Save to temporary WAV file
            temp_wav_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(temp_wav_fd)
            
            # Write WAV
            audio_int16 = (audio_np * self.normalization_factor).astype(np.int16)
            logger.error(f"[AUDIO BEFORE WAV WRITE] shape={audio_int16.shape}, dtype={audio_int16.dtype}, min={audio_int16.min()}, max={audio_int16.max()}, mean={audio_int16.mean():.2f}, std={audio_int16.std():.2f}")
            wav.write(temp_wav_path, self.sample_rate, audio_int16)
            
            # Transcribe using Whisper (async)
            logger.info("Transcribing audio with Whisper (async)")
            loop = asyncio.get_event_loop()
            transcript_result = await loop.run_in_executor(
                self._executor,
                transcribe_wav,
                temp_wav_path,
                language
            )
            
            # Clean up temp file
            try:
                os.unlink(temp_wav_path)
            except Exception:
                pass
            
            # Process transcription result
            if transcript_result.get("ok"):
                text = transcript_result.get("text", "").strip()
                logger.info(f"Transcription: '{text[:100]}...'")
                
                if text:
                    # Extract words
                    word_list = text.split()
                    words = [
                        Word(word=w, index=i)
                        for i, w in enumerate(word_list)
                    ]
                    
                    # Detect fillers
                    filler_analysis = detect_spanish_fillers(text)
                    filler_counts = filler_analysis.get("filler_counts", {})
                    
                    # Create filler instances
                    for filler_type, count in filler_counts.items():
                        for i in range(count):
                            # Distribute fillers across audio duration
                            time_offset = (i + incremental_config.filler_time_distribution_offset) * \
                                        (audio_duration / max(1, count))
                            
                            fillers.append(FillerInstance(
                                type=filler_type,
                                time=time_offset,
                                duration=incremental_config.default_filler_duration
                            ))
                    
                    logger.info(f"Detected {len(fillers)} fillers: {filler_counts}")
            else:
                logger.warning(f"Transcription failed: {transcript_result.get('error', 'Unknown')}")
                
        except Exception as e:
            logger.error(f"Error in transcription: {e}", exc_info=True)
        
        return words, fillers
    
    def reset(self) -> None:
        """Reset processor state."""
        self.audio_buffer = bytearray()
        self.audio_buffer_duration = 0.0
        self._buffer_offset = 0
        self._last_processed_position = 0
        self._partial_sample.clear()
        logger.debug("AudioProcessor reset")
    
    def close(self) -> None:
        """Clean up resources."""
        if self._owns_executor and self._executor:
            self._executor.shutdown(wait=False)
        self.reset()
        logger.info("AudioProcessor closed")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
