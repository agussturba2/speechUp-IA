"""
Prosody analysis functionality for the oratory feedback system.
Provides functions to analyze speech prosody features like pitch and energy.
"""

import numpy as np
import librosa
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ProsodyAnalyzer:
    """Prosody analysis for oratory speech."""
    
    def __init__(self):
        """Initialize prosody analyzer."""
        logger.info("ProsodyAnalyzer initialized")
        
    def extract_pitch_stats(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract pitch statistics from audio.
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            
        Returns:
            Dictionary with pitch mean, std dev, and range
        """
        try:
            f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
            f0 = f0[np.logical_not(np.isnan(f0))]
            if len(f0) == 0:
                return {"pitch_mean": 0, "pitch_std": 0, "pitch_range": 0}
            return {
                "pitch_mean": float(np.mean(f0)),
                "pitch_std": float(np.std(f0)),
                "pitch_range": float(np.max(f0) - np.min(f0)),
            }
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
            return {"pitch_mean": 0, "pitch_std": 0, "pitch_range": 0}
    
    def extract_energy_stats(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract energy/volume statistics from audio.
        
        Args:
            audio: Audio signal as numpy array
            
        Returns:
            Dictionary with energy mean and std dev
        """
        rms = librosa.feature.rms(y=audio)[0]
        return {
            "energy_mean": float(np.mean(rms)),
            "energy_std": float(np.std(rms)),
        }
