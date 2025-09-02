"""
High-level audio analysis service for oratory feedback.

This module provides the AudioAnalyzer class, which orchestrates various
audio processing tasks, including feature extraction, transcription,
and metric calculation.
"""

import logging
import uuid
from typing import Any, Dict, Optional, Tuple

import librosa
import numpy as np


# Local application imports from our refactored structure
# (Assuming these have been created as per previous refactoring exercises)
from audio.model_loader import YamnetModelLoader
from audio.recognition import SpeechRecognizer

logger = logging.getLogger(__name__)


class AudioAnalyzer:
    """
    Orchestrates the audio analysis pipeline for oratory presentations.

    This class extracts audio from a media file, processes it through various
    sub-modules (prosody, transcription, emotion), and aggregates the results.
    """

    def __init__(self, sampling_rate: int = 16000):
        """
        Initializes the audio analyzer and its components.

        Args:
            sampling_rate: The target sampling frequency for analysis.
        """
        self.sampling_rate = sampling_rate
        self.analysis_id = str(uuid.uuid4())

        # --- Dependency Injection & Model Loading ---
        # Models are loaded via dedicated loaders, making their state explicit.
        self.yamnet_model, self.yamnet_class_map = YamnetModelLoader().load()
        self.speech_recognizer = SpeechRecognizer(lang="es")

        # Check if transcription is available
        self.is_transcription_enabled = self.speech_recognizer.vosk_model is not None

        logger.info(
            f"AudioAnalyzer initialized with ID: {self.analysis_id}. "
            f"Transcription enabled: {self.is_transcription_enabled}"
        )

    def _extract_audio(self, media_path: str) -> Tuple[Optional[np.ndarray], int]:
        """
        Loads audio from a media file, extracting it from video if necessary.

        Returns:
            A tuple of (audio_array, sample_rate), or (None, 0) on failure.
        """
        logger.info(f"[{self.analysis_id}] Loading audio from: {media_path}")
        try:
            # Use librosa to handle both audio and video files directly
            audio, sr = librosa.load(media_path, sr=self.sampling_rate, mono=True)
            logger.info(f"[{self.analysis_id}] Audio loaded successfully: {len(audio) / sr:.2f}s")
            return audio, sr
        except Exception as e:
            logger.error(f"[{self.analysis_id}] Failed to load audio from {media_path}: {e}", exc_info=True)
            return None, 0

    def _get_prosody_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Calculates pitch and energy statistics."""
        try:
            f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
            f0_valid = f0[np.isfinite(f0)]
            pitch_stats = {
                "pitch_mean": float(np.mean(f0_valid)) if f0_valid.size > 0 else 0,
                "pitch_std": float(np.std(f0_valid)) if f0_valid.size > 0 else 0,
                "pitch_range": float(np.ptp(f0_valid)) if f0_valid.size > 0 else 0,
            }
        except Exception as e:
            logger.warning(f"[{self.analysis_id}] Pitch extraction failed: {e}")
            pitch_stats = {"pitch_mean": 0, "pitch_std": 0, "pitch_range": 0}

        rms = librosa.feature.rms(y=audio)[0]
        energy_stats = {
            "energy_mean": float(np.mean(rms)),
            "energy_std": float(np.std(rms)),
        }
        return {**pitch_stats, **energy_stats}

    def _get_emotion_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """Analyzes audio for emotional content using YAMNet."""
        if self.yamnet_model is None:
            return {"dominant_emotion": "unknown", "emotion_details": "YAMNet model not loaded."}

        try:
            scores, _, _ = self.yamnet_model(audio)
            mean_scores = np.mean(scores, axis=0)
            top5_indices = np.argsort(mean_scores)[-5:][::-1]

            labels = [self.yamnet_class_map[i].decode('utf-8') for i in top5_indices]
            scores = [float(mean_scores[i]) for i in top5_indices]

            # A simple rule-based mapping for dominant emotion
            dominant_emotion = "neutral"
            if "Speech" in labels[0] or "Music" in labels[0]:
                dominant_emotion = "neutral"
            if "Laughter" in labels or "Giggle" in labels:
                dominant_emotion = "joy"
            if "Crying" in labels or "Sobbing" in labels:
                dominant_emotion = "sadness"

            return {
                "dominant_emotion": dominant_emotion,
                "emotion_details": {"top_labels": labels, "top_scores": scores},
            }
        except Exception as e:
            logger.warning(f"[{self.analysis_id}] Emotion analysis failed: {e}")
            return {"dominant_emotion": "unknown", "emotion_details": f"Error: {e}"}

    def _generate_suggestions(self, metrics: Dict[str, Any]) -> list[str]:
        """Generates simple, rule-based suggestions from metrics."""
        suggestions = []
        # Ensure transcription data exists before creating suggestions
        transcription_metrics = metrics.get("transcription", {})
        if not transcription_metrics or "error" in transcription_metrics:
            return ["Cannot generate suggestions without transcription data."]

        wpm = transcription_metrics.get("wpm", 0)
        total_fillers = transcription_metrics.get("total_fillers", 0)

        if wpm > 170:  # Words per minute
            suggestions.append(
                "Your speaking rate is quite fast. Consider taking more pauses to let your ideas sink in.")
        if wpm < 110 and wpm > 0:
            suggestions.append("Your speaking rate is a bit slow. Try to add more energy and vary your pace.")
        if total_fillers > 5:
            suggestions.append(
                f"You used {total_fillers} filler words. Practice being comfortable with silence to reduce them.")
        if metrics.get("prosody", {}).get("pitch_range", 0) < 50:
            suggestions.append(
                "Your pitch is quite monotonous. Try varying your tone to make your speech more engaging.")

        return suggestions

    def analyze(self, media_path: str) -> Dict[str, Any]:
        """
        Analyzes a video or audio file and returns a structured dictionary of oratory metrics.
        """
        audio, sr = self._extract_audio(media_path)
        if audio is None:
            return {"error": "Failed to load or extract audio from the provided file."}

        # --- Core Analysis ---
        prosody_metrics = self._get_prosody_features(audio, sr)
        emotion_metrics = self._get_emotion_features(audio)

        # --- Transcription (Conditional) ---
        if self.is_transcription_enabled:
            # This is the corrected method call.
            transcription_metrics = self.speech_recognizer.transcribe_audio(audio, sr)
        else:
            transcription_metrics = {
                "error": "Transcription is disabled because the Vosk model could not be loaded."
            }

        # --- Assemble Final Results ---
        final_metrics = {
            "metadata": {
                "duration_seconds": librosa.get_duration(y=audio, sr=sr),
                "analysis_id": self.analysis_id,
            },
            "prosody": prosody_metrics,
            "emotion": emotion_metrics,
            "transcription": transcription_metrics,
        }

        final_metrics["suggestions"] = self._generate_suggestions(final_metrics)

        return final_metrics
