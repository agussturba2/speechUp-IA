"""
Audio preprocessing functionality for the oratory feedback system.
Provides functions for basic audio processing like normalization and silence trimming.
"""

import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio preprocessing for oratory analysis."""
    
    def __init__(self, top_db: int = 30):
        """
        Initialize audio processor.
        
        Args:
            top_db: Threshold (in decibels) for silence detection
        """
        self.top_db = top_db
        logger.info(f"AudioProcessor initialized with top_db={top_db}")
        
    def trim_silence(self, audio: np.ndarray) -> np.ndarray:
        """
        Trim initial and final silences using librosa.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Trimmed audio signal
        """
        yt, _ = librosa.effects.trim(audio, top_db=self.top_db)
        return yt
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to have max amplitude of 1.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Normalized audio signal
        """
        max_val = np.max(np.abs(audio))
        if max_val == 0:
            return audio
        return audio / max_val
