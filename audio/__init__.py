"""Audio analysis package for oratory feedback system.

This module exports the main audio analysis components:
- OratoryAnalyzer: Main orchestrator class (primary entry point)
- Specialized components for advanced usage:
  - AudioExtractor: Extract audio from media files
  - AudioProcessor: Audio preprocessing 
  - ProsodyAnalyzer: Pitch and energy analysis
  - SpeechSegmenter: Voice activity detection
  - EmotionDetector: Speech emotion analysis
"""

# Modular components
from .processors import AudioProcessor
# from .prosody import ProsodyAnalyzer  # Removed - replaced with functional approach
from .speech import SpeechSegmenter

# ASR components
from .asr import transcribe_wav, is_asr_enabled
from .text_metrics import compute_wpm, detect_spanish_fillers, normalize_fillers_per_minute

__all__ = [
    "AudioProcessor",
    "SpeechSegmenter",
    # ASR exports
    "transcribe_wav",
    "is_asr_enabled", 
    "compute_wpm",
    "detect_spanish_fillers",
    "normalize_fillers_per_minute",
]
