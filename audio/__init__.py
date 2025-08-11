"""Audio analysis package for oratory feedback system.

This module exports the main audio analysis components:
- OratoryAnalyzer: Main orchestrator class (primary entry point)
- Specialized components for advanced usage:
  - AudioExtractor: Extract audio from media files
  - AudioProcessor: Audio preprocessing 
  - ProsodyAnalyzer: Pitch and energy analysis
  - SpeechSegmenter: Voice activity detection
  - SpeechRecognizer: Speech-to-text and analysis
  - EmotionDetector: Speech emotion analysis
"""

# Modular components
from .analyzer import OratoryAnalyzer
from .extractors import AudioExtractor
from .processors import AudioProcessor
from .prosody import ProsodyAnalyzer
from .speech import SpeechSegmenter
from .recognition import SpeechRecognizer
from .emotion import EmotionDetector

__all__ = [
    "OratoryAnalyzer",
    "AudioExtractor",
    "AudioProcessor",
    "ProsodyAnalyzer",
    "SpeechSegmenter",
    "SpeechRecognizer",
    "EmotionDetector",
]
