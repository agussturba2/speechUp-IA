# audio/utils.py
"""Utility functions for audio data preprocessing."""

import io
import logging
import wave

import librosa
import numpy as np

from . import constants

logger = logging.getLogger(__name__)


def convert_to_wav_buffer(
        audio_data: np.ndarray,
        original_sr: int,
) -> io.BytesIO:
    """
    Converts a numpy audio array to a WAV format in an in-memory buffer.

    The audio is resampled to the standard Vosk sample rate (16kHz),
    converted to 16-bit mono, and stored in a BytesIO object.

    Args:
        audio_data: The raw audio signal as a numpy array.
        original_sr: The original sample rate of the audio data.

    Returns:
        An in-memory BytesIO buffer containing the WAV data.
    """
    if original_sr != constants.VOSK_SAMPLE_RATE:
        audio_data = librosa.resample(
            y=audio_data,
            orig_sr=original_sr,
            target_sr=constants.VOSK_SAMPLE_RATE
        )
        logger.debug(
            f"Audio resampled from {original_sr}Hz to {constants.VOSK_SAMPLE_RATE}Hz."
        )

    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(constants.WAV_CHANNELS)
        wf.setsampwidth(constants.WAV_SAMPLE_WIDTH_BYTES)
        wf.setframerate(constants.VOSK_SAMPLE_RATE)
        # Convert float audio to 16-bit integer
        audio_int16 = (audio_data * 32767).astype(np.int16)
        wf.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer
