# audio/constants.py
"""Centralized constants for the audio processing modules."""

# --- Vosk & Audio Processing ---
VOSK_MODEL_NAME_ES = "model-es"
VOSK_SAMPLE_RATE = 16000
WAV_SAMPLE_WIDTH_BYTES = 2  # 16-bit audio
WAV_CHANNELS = 1  # Mono

# --- Speech Metrics ---
# Common filler words in Spanish
FILLERS_ES = ["eh", "mmm", "este", "pues", "osea", "o sea"]
