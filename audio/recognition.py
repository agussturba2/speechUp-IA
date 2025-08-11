# C:/Users/USURIO/Desktop/TensorFlow/audio/recognition.py

import json
import logging
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import vosk

from . import constants, audio_utils
from .model_loader import VoskModelLoader

logger = logging.getLogger(__name__)


class SpeechRecognizer:
    """
    Performs speech-to-text transcription and all related analysis
    (filler words, speech rate) using a Vosk model.
    """

    def __init__(self, lang: str = "es", model_path: Optional[str] = None):
        """Initializes the recognizer and loads the Vosk model."""
        loader = VoskModelLoader(lang, model_path)
        self.vosk_model = loader.load()

    def calculate_speech_rate(self, word_count: int, speech_duration_sec: float) -> int:
        """
        Calculates words per minute (WPM).

        Args:
            word_count: The total number of words spoken.
            speech_duration_sec: The total duration of active speech in seconds.

        Returns:
            The calculated words per minute.
        """
        if speech_duration_sec <= 0:
            return 0
        return int(word_count * 60 / speech_duration_sec)

    def analyze_fillers(self, words: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Counts occurrences of predefined filler words from a list of timed words.

        Args:
            words: A list of word dictionaries from a Vosk transcription result.

        Returns:
            A dictionary containing details about the filler words found.
        """
        if not words:
            return {"fillers_details": {}, "total_fillers": 0}

        # Use a Counter for an efficient way to count filler words
        filler_counter = Counter(
            word_data.get("word", "").lower()
            for word_data in words
            if word_data.get("word", "").lower() in constants.FILLERS_ES
        )
        return {
            "fillers_details": dict(filler_counter),
            "total_fillers": sum(filler_counter.values()),
        }

    def transcribe_audio(self, audio: np.ndarray, sr: int, speech_duration_sec: float) -> Dict[str, Any]:
        """
        Transcribes an audio signal and computes all related speech metrics.
        This is the single entry point for transcription analysis.
        """
        if self.vosk_model is None:
            return {"error": "Vosk model not loaded, transcription disabled."}

        # Prepare audio for Vosk
        wav_buffer = audio_utils.convert_to_wav_buffer(audio, original_sr=sr)

        try:
            # Perform transcription
            recognizer = vosk.KaldiRecognizer(self.vosk_model, float(sr))
            recognizer.SetWords(True)
            recognizer.AcceptWaveform(wav_buffer.read())
            result_json = recognizer.FinalResult()
            result = json.loads(result_json)
        except Exception as e:
            logger.error(f"Error during Vosk recognition: {e}", exc_info=True)
            return {"error": f"Recognition failed: {e}"}

        # Extract data from Vosk result
        full_text = result.get("text", "")
        word_timings = result.get("result", [])
        total_words = len(word_timings)

        # Call internal helper methods to get all metrics
        filler_analysis = self.analyze_fillers(word_timings)
        wpm = self.calculate_speech_rate(total_words, speech_duration_sec)

        # Assemble and return the complete, structured result
        return {
            "transcript": full_text,
            "word_timings": word_timings,
            "total_words": total_words,
            "words_per_minute": wpm,
            **filler_analysis,
        }
