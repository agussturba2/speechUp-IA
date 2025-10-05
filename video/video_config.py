# /video_config.py

"""
Centralized configuration for the video analysis pipeline.
"""


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
