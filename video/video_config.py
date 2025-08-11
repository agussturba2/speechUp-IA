# /video_config.py

"""
Centralized configuration for the video analysis pipeline.
"""

# Quality profiles define the trade-off between speed and accuracy.
# - width/height:      Frame resolution for analysis. Smaller is faster.
# - buffer_seconds:    Duration of video segments processed at once.
# - confidence:        Minimum confidence for MediaPipe model detections.
# - complexity:        Model complexity for MediaPipe (0=light, 1=full, 2=heavy).
# - smooth:            Whether to apply temporal smoothing to landmarks.
QUALITY_PROFILES = {
    "speed": {
        "width": 128, "height": 128, "buffer_seconds": 3,
        "confidence": 0.5, "complexity": 0, "smooth": False
    },
    "quality": {
        "width": 192, "height": 192, "buffer_seconds": 1,
        "confidence": 0.7, "complexity": 1, "smooth": True
    },
    "balanced": {
        "width": 160, "height": 160, "buffer_seconds": 2,
        "confidence": 0.6, "complexity": 0, "smooth": True
    },
}

# --- Caching ---
CACHE_EXPIRATION_SECONDS = 86400  # 24 hours
HASH_RESIZE_DIM = (16, 16)  # Small dimension for creating frame hashes

# --- Analysis ---
# Thresholds for generating advice based on detection ratios.
ADVICE_THRESHOLDS = {
    "face": {
        "poor": 0.70,  # Below this, advice is "poor"
        "good": 0.85,  # Above this, advice is "excellent"
    },
    "posture": {
        "poor": 0.60,
        "good": 0.80,
    }
}

# --- Performance ---
# Determines how many worker threads to use for frame processing.
# Considers available CPU cores and memory to avoid resource exhaustion.
# Formula: min(cpu_cores, available_ram_gb / ram_per_thread_gb)
WORKER_CONFIG = {
    "ram_per_thread_gb": 0.5
}
