# audio/metrics.py
"""Functions for calculating speech-related metrics from transcription data."""

from collections import Counter
from typing import Any, Dict, List

from . import constants


def calculate_wpm(total_words: int, duration_sec: float) -> int:
    """
    Calculates words per minute (WPM).

    Args:
        total_words: The total number of words spoken.
        duration_sec: The total duration of the speech in seconds.

    Returns:
        The calculated words per minute, or 0 if duration is zero.
    """
    if duration_sec <= 0:
        return 0
    return int(total_words * 60 / duration_sec)


def count_fillers(words: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Counts occurrences of predefined filler words from a list of timed words.

    Args:
        words: A list of word dictionaries from a Vosk transcription result.

    Returns:
        A dictionary with filler words as keys and their counts as values.
    """
    filler_counter = Counter()
    for word_data in words:
        word = word_data.get("word", "").lower()
        if word in constants.FILLERS_ES:
            filler_counter[word] += 1
    return dict(filler_counter)
