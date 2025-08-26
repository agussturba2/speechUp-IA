"""
Text metrics computation for speech analysis.

This module provides functions to compute words-per-minute (WPM) and
detect Spanish filler words from transcribed text.
"""

import re
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Spanish filler words (lowercase, unicode-aware)
SPANISH_FILLERS = [
    "eh", "ehh", "em", "ehm", "mmm",
    "este", "osea", "o sea", "tipo", "digamos", "nada", "viste", 
    "a ver", "bueno", "okey", "ok", "igual",
    # Additional Spanish fillers
    "¿no?", "viste", "digamos", "tipo", "nada", "bueno", "a ver"
]

def normalize_text_for_fillers(text: str) -> str:
    """
    Normalize text by removing accents and punctuation for filler detection.
    
    Args:
        text: Input text string
    
    Returns:
        Normalized text with accents removed and punctuation normalized
    """
    import unicodedata
    
    # Remove accents (decompose unicode and remove combining characters)
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    
    # Normalize punctuation and spacing
    text = text.replace('¿', '').replace('?', ' ? ').replace('¡', '').replace('!', ' ! ')
    text = text.replace('.', ' . ').replace(',', ' , ').replace(':', ' : ')
    text = text.replace(';', ' ; ').replace('(', ' ').replace(')', ' ')
    
    # Normalize multiple spaces
    text = ' '.join(text.split())
    
    return text.lower().strip()

def compute_wpm(transcript: str, speech_dur_sec: float) -> float:
    """
    Compute words-per-minute from transcript and speech duration.
    Returns 0.0 if duration < 3.0s or no words.
    """
    if not transcript or speech_dur_sec < 3.0:
        return 0.0
    words = len(transcript.strip().split())
    if words == 0:
        return 0.0
    return (words / speech_dur_sec) * 60.0

def detect_spanish_fillers(transcript: str) -> Dict[str, any]:
    """
    Detect Spanish filler words in transcript.
    
    Args:
        transcript: Transcribed text (lowercase, trimmed)
    
    Returns:
        Dict with filler analysis:
        {
            "fillers_per_min": float,           # Fillers per minute
            "filler_counts": Dict[str, int]     # Count per filler type
        }
    """
    if not transcript:
        return {"fillers_per_min": 0.0, "filler_counts": {}}
    
    # Normalize text for better filler detection
    normalized_text = normalize_text_for_fillers(transcript)
    
    # Initialize filler counts
    filler_counts = {}
    total_fillers = 0
    
    # Count each filler word (substring match on token boundaries)
    for filler in SPANISH_FILLERS:
        # Use word boundary regex to avoid partial matches
        # Handle special cases like "o sea" (bigram)
        if " " in filler:
            # For bigrams, use simple substring search with word boundaries
            pattern = r'\b' + re.escape(filler) + r'\b'
            count = len(re.findall(pattern, normalized_text))
        else:
            # For single words, use word boundary
            pattern = r'\b' + re.escape(filler) + r'\b'
            count = len(re.findall(pattern, normalized_text))
        
        if count > 0:
            filler_counts[filler] = count
            total_fillers += count
    
    # Calculate fillers per minute (assuming 1 minute speech if duration unknown)
    # This will be adjusted by the caller with actual duration
    fillers_per_min = float(total_fillers)
    
    logger.debug(f"Filler detection: {total_fillers} total fillers, types: {list(filler_counts.keys())}")
    
    return {
        "fillers_per_min": fillers_per_min,
        "filler_counts": filler_counts
    }

def normalize_fillers_per_minute(fillers_per_min: float, speech_dur_sec: float) -> float:
    """
    Normalize fillers per minute based on actual speech duration.
    
    Args:
        fillers_per_min: Raw fillers count
        speech_dur_sec: Speech duration in seconds
    
    Returns:
        Normalized fillers per minute
    """
    if speech_dur_sec <= 0:
        return 0.0
    
    # Convert to per-minute rate
    duration_minutes = speech_dur_sec / 60.0
    normalized_rate = (fillers_per_min / max(duration_minutes, 1e-6))
    
    return float(normalized_rate)

# --- TESTS ---
def _test_compute_wpm():
    assert compute_wpm("hola mundo", 2.0) == 0.0  # too short
    assert compute_wpm("", 10.0) == 0.0
    assert compute_wpm("uno dos tres cuatro cinco", 10.0) == 30.0
    assert compute_wpm("uno dos tres", 3.0) == 60.0
    print("compute_wpm guards OK")

def _smoke_test_transcribe_and_wpm():
    from audio.asr import transcribe_wav
    import tempfile
    import wave
    import numpy as np
    # Create a short WAV file (5s silence)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        wav_path = tmp_file.name
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            frames = np.zeros(16000*5, dtype=np.int16)
            wav_file.writeframes(frames.tobytes())
    try:
        result = transcribe_wav(wav_path, lang="es")
        assert "ok" in result
        assert result["ok"] in (True, False)
        wpm = compute_wpm(result["text"], result["duration_sec"])
        assert wpm == 0.0  # silence
        print("smoke test: silence yields wpm=0.0 as expected")
    finally:
        import os
        os.unlink(wav_path)

if __name__ == "__main__":
    _test_compute_wpm()
    _smoke_test_transcribe_and_wpm()
    print("All text_metrics tests passed.")
