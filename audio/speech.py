"""
Speech segmentation and analysis functionality for the oratory feedback system.
Provides functions to detect speech/silence segments and analyze speech patterns.
"""

import numpy as np
import librosa
import logging
import webrtcvad
from typing import List, Tuple, Dict, Any
import itertools
from collections import Counter
import json

# Optional imports
try:
    import vosk  # Offline speech recognition
except ImportError:
    vosk = None  # Handle in code gracefully

logger = logging.getLogger(__name__)

class SpeechSegmenter:
    """Speech segmentation for voice activity detection and pause analysis."""
    
    def __init__(self, vad_mode: int = 2):
        """
        Initialize speech segmenter.
        
        Args:
            vad_mode: WebRTC VAD aggressiveness (0-3, higher is more aggressive)
        """
        self.vad_mode = vad_mode
        logger.info(f"SpeechSegmenter initialized with vad_mode={vad_mode}")
        
    def get_speech_segments(self, audio: np.ndarray, sr: int, frame_ms: int = 30) -> List[Tuple[bool, float, float]]:
        """
        Return list of speech/silence segments as tuples: (is_speech, start_sec, end_sec).
        
        Args:
            audio: Audio signal as numpy array
            sr: Sample rate
            frame_ms: Frame size in milliseconds
            
        Returns:
            List of tuples (is_speech, start_sec, end_sec)
        """
        if sr != 16000:
            audio16 = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000
        else:
            audio16 = audio
        pcm_data = (audio16 * 32768).astype(np.int16).tobytes()
        frame_len = int(frame_ms * sr / 1000) * 2  # bytes
        vad = webrtcvad.Vad(self.vad_mode)
        segments = []
        is_speech_prev = None
        start = 0.0
        for i in range(0, len(pcm_data), frame_len):
            frame = pcm_data[i:i + frame_len]
            if len(frame) < frame_len:
                break
            try:
                is_speech = vad.is_speech(frame, 16000)
                if is_speech != is_speech_prev and is_speech_prev is not None:
                    segments.append((is_speech_prev, start, i / len(pcm_data) * (len(audio) / sr)))
                    start = i / len(pcm_data) * (len(audio) / sr)
                is_speech_prev = is_speech
            except Exception:
                continue
        if is_speech_prev is not None:
            segments.append((is_speech_prev, start, len(audio) / sr))
        return segments
        
    def analyze_pauses(self, segments: List[Tuple[bool, float, float]]) -> Dict[str, Any]:
        """
        Analyze pause patterns in segmented speech.
        
        Args:
            segments: List of segments from get_speech_segments
            
        Returns:
            Dictionary with pause metrics
        """
        if not segments:
            return {
                "speech_percent": 0.0,
                "avg_speech_segment": 0.0,
                "avg_pause_length": 0.0,
                "pause_frequency": 0.0,
            }
            
        total_dur = segments[-1][2] - segments[0][1]
        if total_dur <= 0:
            return {
                "speech_percent": 0.0,
                "avg_speech_segment": 0.0,
                "avg_pause_length": 0.0,
                "pause_frequency": 0.0,
            }
            
        speech_segments = [seg for seg in segments if seg[0]]
        pause_segments = [seg for seg in segments if not seg[0]]
        
        speech_dur = sum(seg[2] - seg[1] for seg in speech_segments)
        speech_percent = speech_dur / total_dur * 100 if total_dur > 0 else 0
        
        avg_speech_segment = speech_dur / len(speech_segments) if speech_segments else 0
        avg_pause_length = sum(seg[2] - seg[1] for seg in pause_segments) / len(pause_segments) if pause_segments else 0
        pause_frequency = len(pause_segments) / (total_dur / 60) if total_dur > 0 else 0
            
        return {
            "speech_percent": float(speech_percent),
            "avg_speech_segment": float(avg_speech_segment),
            "avg_pause_length": float(avg_pause_length),
            "pause_frequency": float(pause_frequency),
        }
