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

# --- YAMNet Audio Classification ---
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
YAMNET_CLASS_MAP_URL = "https://storage.googleapis.com/yamnet/yamnet_class_map.csv"