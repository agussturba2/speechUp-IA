# C:/Users/USURIO/Desktop/TensorFlow/audio/analyzer.py

import os
import uuid
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from audio.emotion import EmotionDetector
from audio.extractors import AudioExtractor
from audio.processors import AudioProcessor
from audio.prosody import ProsodyAnalyzer
from audio.recognition import SpeechRecognizer
from audio.speech import SpeechSegmenter
from utils.logging import LogContext, get_logger, log_execution_time

logger = get_logger(__name__)


class OratoryAnalyzer:
    """
    Orchestrates the audio analysis pipeline for oratory presentations.

    This class is composed of specialized sub-components to handle each
    part of the analysis, making the system modular and testable.
    """

    def __init__(
            self,
            sampling_rate: int = 16000,
            vad_mode: int = 2,
            recognition_lang: str = "es",
    ):
        """
        Initializes all necessary audio analysis components.
        """
        self.extractor = AudioExtractor(sampling_rate=sampling_rate)
        self.processor = AudioProcessor(top_db=30)
        self.prosody = ProsodyAnalyzer()
        self.segmenter = SpeechSegmenter(vad_mode=vad_mode)
        self.recognizer = SpeechRecognizer(lang=recognition_lang)
        self.emotion = EmotionDetector()

        logger.info(
            f"OratoryAnalyzer initialized with sampling_rate={sampling_rate}, "
            f"vad_mode={vad_mode}, recognition_lang='{recognition_lang}'"
        )

    def _load_and_prepare_audio(self, media_path: str) -> Tuple[np.ndarray, int]:
        """
        Loads audio from a media file, then processes it for analysis.
        """
        try:
            ext = Path(media_path).suffix.lower()
            if ext in [".mp4", ".mkv", ".mov", ".avi"]:
                audio, sr = self.extractor.extract_from_video(media_path)
            else:
                audio, sr = self.extractor.load_audio_file(media_path)

            audio = self.processor.trim_silence(audio)
            audio = self.processor.normalize_audio(audio)

            if audio.size == 0:
                raise ValueError("Audio is empty after trimming silence.")

            return audio, sr
        except Exception as e:
            logger.error(f"Failed during audio loading or preparation: {e}", exc_info=True)
            raise IOError(f"Audio preprocessing failed: {e}") from e

    def _analyze_prosody(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyzes and returns prosody (pitch and energy) metrics."""
        try:
            pitch_stats = self.prosody.extract_pitch_stats(audio, sr)
            energy_stats = self.prosody.extract_energy_stats(audio)
            return {**pitch_stats, **energy_stats}
        except Exception as e:
            logger.warning(f"Prosody analysis failed: {e}", exc_info=True)
            return {"error": "Prosody analysis could not be completed."}

    def _analyze_segmentation(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyzes and returns speech segmentation and pause metrics."""
        try:
            segments = self.segmenter.get_speech_segments(audio, sr)
            pause_analysis = self.segmenter.analyze_pauses(segments)

            speech_segments = [seg for seg in segments if seg[0]]
            speech_duration = sum(seg[2] - seg[1] for seg in speech_segments)

            return {
                "speech_duration_seconds": float(speech_duration),
                **pause_analysis,
            }
        except Exception as e:
            logger.warning(f"Segmentation analysis failed: {e}", exc_info=True)
            return {
                "error": "Segmentation analysis could not be completed.",
                "speech_duration_seconds": 0.0,
            }

    def _analyze_transcription(
            self, audio: np.ndarray, sr: int, speech_duration: float
    ) -> Dict[str, Any]:
        """
        Gets a complete transcription analysis from the SpeechRecognizer.
        """
        try:
            # CORRECTED: Pass the speech_duration to the recognizer
            return self.recognizer.transcribe_audio(audio, sr, speech_duration)
        except Exception as e:
            logger.warning(f"Speech recognition failed: {e}", exc_info=True)
            return {"error": "Speech recognition could not be completed."}

    def _analyze_emotion(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyzes and returns emotion-related metrics."""
        try:
            return self.emotion.analyze(audio, sr)
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}", exc_info=True)
            return {"error": "Emotion analysis could not be completed."}

    @log_execution_time(logger)
    def analyze(self, media_path: str) -> Dict[str, Any]:
        """
        Analyzes a video or audio file and returns oratory metrics.
        """
        analysis_id = str(uuid.uuid4())
        logger.info(
            f"Starting audio analysis: {os.path.basename(media_path)}",
            extra={"analysis_id": analysis_id},
        )

        with LogContext(logger, analysis_id=analysis_id):
            try:
                audio, sr = self._load_and_prepare_audio(media_path)
            except (IOError, ValueError) as e:
                return {"error": str(e)}

            # --- Analysis Pipeline ---
            # Each step is now independent and returns its own results.
            segmentation_results = self._analyze_segmentation(audio, sr)
            speech_duration = segmentation_results.get("speech_duration_seconds", 0.0)

            # --- Assemble Final Report ---
            return {
                "metadata": {
                    "duration_seconds": float(audio.size / sr),
                    "speech_seconds": speech_duration,
                    "analysis_id": analysis_id,
                },
                "prosody": self._analyze_prosody(audio, sr),
                "segmentation": segmentation_results,
                "transcription": self._analyze_transcription(audio, sr, speech_duration),
                "emotion": self._analyze_emotion(audio, sr),
            }
