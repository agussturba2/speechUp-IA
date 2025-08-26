"""
Automatic Speech Recognition (ASR) module using OpenAI Whisper.

This module provides transcription capabilities for audio files, with support for
both CPU and GPU processing, configurable models, and graceful fallbacks.
"""

import os
import logging
import time
import traceback
from typing import Dict, Optional, Any, List, Tuple
import tempfile
import subprocess

logger = logging.getLogger(__name__)

# Configurable constants
TIMEOUT_SEC = 60
MAX_WINDOW_SEC = float(os.getenv("SPEECHUP_ASR_MAX_WINDOW_SEC", "20"))

# Debug flag
DEBUG_ASR = os.getenv("SPEECHUP_DEBUG_ASR", "0") == "1"

def _log_debug(msg):
    if DEBUG_ASR:
        logger.debug(msg)

def _log_info(msg):
    if DEBUG_ASR:
        logger.info(msg)

def _get_device_info() -> str:
    # Allow override
    device_env = os.getenv("WHISPER_DEVICE", None)
    if device_env in {"cpu", "cuda", "mps"}:
        return device_env
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    except ImportError:
        return "cpu"

def _get_optimal_model(device: str) -> str:
    """
    Select optimal Whisper model based on device and environment override.
    
    Args:
        device: Device type ('cpu', 'cuda', 'mps')
    
    Returns:
        Model name for optimal performance/accuracy trade-off
    """
    # Allow explicit override
    env_model = os.getenv("SPEECHUP_ASR_MODEL", None)
    if env_model:
        return env_model
    
    # Auto-select based on device
    if device == "cpu":
        return "base"  # CPU: balance speed vs accuracy
    else:
        return "small"  # GPU: prioritize accuracy with reasonable speed



def trim_wav_segment(src_wav: str, dst_wav: str, t_start: float = 0.0, t_dur: float = 20.0) -> bool:
    """Trim a WAV file to a segment using ffmpeg. Returns True if successful."""
    try:
        import ffmpeg
        (
            ffmpeg
            .input(src_wav, ss=t_start, t=t_dur)
            .output(dst_wav, acodec='pcm_s16le', ac=1, ar='16k', loglevel='error')
            .overwrite_output()
            .run()
        )
        return True
    except Exception as e:
        _log_debug(f"ffmpeg trim failed: {e}\n{traceback.format_exc()}")
        return False

def get_wav_duration(wav_path: str) -> float:
    try:
        import wave
        with wave.open(wav_path, 'rb') as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        return 0.0

def transcribe_wav(wav_path: str, lang: Optional[str] = None) -> Dict[str, Any]:
    """
    Transcribe up to MAX_WINDOW_SEC of a WAV file using Whisper.
    Returns a structured dict with status and error info.
    """
    # Resolve device and model selection
    used_device = _get_device_info()
    used_model = _get_optimal_model(used_device)
    
    cache_dir = os.path.expanduser("~/.cache/whisper")
    os.makedirs(cache_dir, exist_ok=True)
    
    error = None
    temp_trim = None
    
    try:
        import whisper
        
        # Get duration and handle trimming
        duration_sec = get_wav_duration(wav_path)
        used_window = min(duration_sec, MAX_WINDOW_SEC)
        
        wav_for_asr = wav_path
        if duration_sec > MAX_WINDOW_SEC:
            temp_trim = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_trim.close()
            if trim_wav_segment(wav_path, temp_trim.name, 0.0, MAX_WINDOW_SEC):
                wav_for_asr = temp_trim.name
                _log_info(f"Trimmed WAV to {MAX_WINDOW_SEC}s for ASR: {wav_for_asr}")
            else:
                logger.warning("ffmpeg trim failed, using full file for ASR")
        
        if DEBUG_ASR:
            logger.info(f"ASR enabled: model={used_model}, device={used_device}, duration={duration_sec:.2f}s, used_window={used_window:.2f}s")
        
        # Load model and transcribe
        start = time.time()
        model = whisper.load_model(used_model, download_root=cache_dir, device=used_device)
        _log_debug(f"Loaded Whisper model '{used_model}' on device '{used_device}'")
        
        # Transcribe with language handling
        result = model.transcribe(
            wav_for_asr,
            language=lang,  # Can be None for auto-detection
            task="transcribe",
            verbose=False,
            no_speech_threshold=0.6
        )
        elapsed = time.time() - start
        
        # Preserve transcript fidelity - only strip whitespace
        raw_text = (result.get("text") or "").strip()
        text_len = len(raw_text)
        transcript_short = raw_text[:200].strip() if raw_text else ""
        
        # Get duration from result or fallback
        result_duration = result.get("duration", used_window)
        duration_sec = result_duration if result_duration else used_window
        
        # Compute real STT confidence from Whisper segments
        segments = result.get("segments", []) or []  # ensure list
        stt_conf, debug_stats = _compute_real_stt_confidence(segments, used_device, used_model, text_len)
        
        if DEBUG_ASR:
            logger.info(f"ASR transcript (first 120 chars): {raw_text[:120]}")
            logger.info(f"STT confidence: {stt_conf:.3f}")
            logger.info(f"ASR completed in {elapsed:.2f}s")
        
        return {
            "ok": True,
            "text": raw_text,
            "transcript_short": transcript_short,
            "duration_sec": duration_sec,
            "stt_confidence": stt_conf,
            "debug": debug_stats
        }
        
    except Exception as e:
        error = str(e)
        logger.warning(f"ASR failed: {error}")
        if DEBUG_ASR:
            logger.debug(traceback.format_exc())
        
        # Return failure with debug info
        return {
            "ok": False,
            "error": error,
            "stt_confidence": 0.3,
            "debug": {
                "avg_logprob_mean": None,
                "no_speech_prob_min": None,
                "no_speech_prob_max": None,
                "no_speech_prob_mean": None,
                "num_segments": 0,
                "device": used_device,
                "model": used_model
            }
        }
        
    finally:
        if temp_trim:
            try:
                os.remove(temp_trim.name)
            except Exception:
                pass

def _compute_real_stt_confidence(segments: List[dict], used_device: str, used_model: str, text_len: int) -> Tuple[float, Dict[str, Any]]:
    """
    Compute real STT confidence from Whisper segments.
    
    Args:
        segments: List of Whisper segments
        used_device: Device used for transcription
        used_model: Model used for transcription
        text_len: Length of transcribed text
    
    Returns:
        Tuple of (confidence_score, debug_stats)
    """
    try:
        # Build a list of "valid" segments using very permissive rules
        valid_segments = []
        lp_vals = []
        nsp_vals = []
        
        for segment in segments:
            # A segment is valid if it has non-empty text OR no_speech_prob < 0.95
            seg_text = segment.get("text", "").strip()
            avg_logprob = segment.get("avg_logprob", None)
            no_speech_prob = segment.get("no_speech_prob", None)
            
            # Set defaults if metrics are missing
            if avg_logprob is None:
                avg_logprob = -0.8
            if no_speech_prob is None:
                no_speech_prob = 0.5
            
            # Very permissive validation: segment is valid if it has text OR low no_speech_prob
            if seg_text or no_speech_prob < 0.95:
                valid_segments.append(segment)
                lp_vals.append(avg_logprob)
                nsp_vals.append(no_speech_prob)
        
        # If there are still ZERO valid segments BUT text_len > 0 (we clearly transcribed something),
        # fabricate ONE pseudo-segment to prevent confidence from collapsing to 0.30 fallback
        if not valid_segments and text_len > 0:
            # Create pseudo-segment with reasonable defaults
            lp_vals = [-0.6]  # Moderate confidence
            nsp_vals = [0.4]  # Moderate speech probability
            valid_segments = [{"text": "pseudo", "avg_logprob": -0.6, "no_speech_prob": 0.4}]
        
        if not valid_segments:
            return 0.3, {
                "avg_logprob_median": None,
                "avg_logprob_mean": None,
                "no_speech_prob_median": None,
                "no_speech_prob_min": None,
                "no_speech_prob_max": None,
                "no_speech_prob_mean": None,
                "lp_norm": None,
                "conf_final": 0.3,
                "num_segments": 0,
                "device": used_device,
                "model": used_model,
                "used_topk": False,
                "k": 0
            }
        
        # Compute robust statistics using medians
        lp_med = sorted(lp_vals)[len(lp_vals) // 2] if lp_vals else -1.0
        nsp_med = sorted(nsp_vals)[len(nsp_vals) // 2] if nsp_vals else 1.0
        
        # Also compute means for reporting
        lp_mean = sum(lp_vals) / len(lp_vals) if lp_vals else -1.0
        nsp_mean = sum(nsp_vals) / len(nsp_vals) if nsp_vals else 1.0
        nsp_min = min(nsp_vals) if nsp_vals else None
        nsp_max = max(nsp_vals) if nsp_vals else None
        
        # Normalize logprob from [-2.5, 0] to [0, 1] (clip before scaling)
        lp_clipped = max(-2.5, min(0.0, lp_med))
        lp_norm = (lp_clipped + 2.5) / 2.5
        
        # Final confidence (robust blend): 70% from logprob, 30% from no_speech_prob
        stt_conf_robust = max(0.0, min(1.0, 0.7 * lp_norm + 0.3 * (1.0 - nsp_med)))
        
        # Optional top-k fallback if there are >=4 segments
        stt_conf_topk = stt_conf_robust
        used_topk = False
        k = 0
        
        if len(valid_segments) >= 4:
            k = max(2, int((len(valid_segments) * 0.5) + 0.5))  # ceil equivalent
            # Get top-k segments by highest avg_logprob
            segment_lp_pairs = list(zip(valid_segments, lp_vals))
            segment_lp_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by logprob descending
            top_k_segments = segment_lp_pairs[:k]
            top_k_lp_vals = [pair[1] for pair in top_k_segments]
            
            # Compute median of top-k
            top_k_lp_med = sorted(top_k_lp_vals)[len(top_k_lp_vals) // 2] if top_k_lp_vals else lp_med
            
            # Normalize top-k logprob
            top_k_lp_clipped = max(-2.5, min(0.0, top_k_lp_med))
            top_k_lp_norm = (top_k_lp_clipped + 2.5) / 2.5
            
            # Compute top-k confidence
            stt_conf_topk = max(0.0, min(1.0, 0.7 * top_k_lp_norm + 0.3 * (1.0 - nsp_med)))
            used_topk = True
        
        # Choose the better of robust-median score and top-k score
        stt_conf = max(stt_conf_robust, stt_conf_topk)
        
        # Build comprehensive debug dict
        debug_stats = {
            "num_segments": int(len(valid_segments)),
            "avg_logprob_median": float(lp_med),
            "avg_logprob_mean": float(lp_mean),
            "no_speech_prob_median": float(nsp_med),
            "no_speech_prob_mean": float(nsp_mean),
            "no_speech_prob_min": float(nsp_min) if nsp_min is not None else None,
            "no_speech_prob_max": float(nsp_max) if nsp_max is not None else None,
            "lp_norm": float(lp_norm),
            "conf_final": float(stt_conf),
            "device": used_device,
            "model": used_model,
            "used_topk": used_topk,
            "k": int(k)
        }
        
        return stt_conf, debug_stats
        
    except Exception as e:
        logger.warning(f"Error computing STT confidence: {e}")
        return 0.3, {
            "avg_logprob_median": None,
            "avg_logprob_mean": None,
            "no_speech_prob_median": None,
            "no_speech_prob_min": None,
            "no_speech_prob_max": None,
            "no_speech_prob_mean": None,
            "lp_norm": None,
            "conf_final": 0.3,
            "num_segments": 0,
            "device": used_device,
            "model": used_model,
            "used_topk": False,
            "k": 0
        }

def is_asr_enabled() -> bool:
    """Check if ASR is enabled via environment variable."""
    return os.getenv("SPEECHUP_USE_ASR", "0") == "1"
