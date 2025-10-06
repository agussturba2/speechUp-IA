"""
Automatic Speech Recognition (ASR) module using OpenAI Whisper.

This module provides transcription capabilities for audio files, with support for
both CPU and GPU processing, configurable models, and graceful fallbacks.
"""

import os
import logging
import time
import traceback
import tempfile
import statistics
from typing import Dict, Optional, Any, List, Tuple

# Optional imports with graceful fallbacks
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False
    ffmpeg = None

try:
    import wave
    WAVE_AVAILABLE = True
except ImportError:
    WAVE_AVAILABLE = False
    wave = None

logger = logging.getLogger(__name__)

# Configurable constants
TIMEOUT_SEC = 60
MAX_WINDOW_SEC = float(os.getenv("SPEECHUP_ASR_MAX_WINDOW_SEC", "20"))

# Debug flag
DEBUG_ASR = os.getenv("SPEECHUP_DEBUG_ASR", "0") == "1"

# Confidence calculation constants
CONFIDENCE_LOGPROB_WEIGHT = 0.7
CONFIDENCE_NOSPEECH_WEIGHT = 0.3
LOGPROB_MIN_THRESHOLD = -2.5
LOGPROB_MAX_THRESHOLD = 0.0
LOGPROB_RANGE = 2.5  # abs(LOGPROB_MIN_THRESHOLD - LOGPROB_MAX_THRESHOLD)
NO_SPEECH_THRESHOLD = 0.6  # Used in transcribe
SEGMENT_VALIDATION_THRESHOLD = 0.95  # Max no_speech_prob for valid segment
DEFAULT_AVG_LOGPROB = -0.8
DEFAULT_NO_SPEECH_PROB = 0.5
PSEUDO_SEGMENT_LOGPROB = -0.6
PSEUDO_SEGMENT_NO_SPEECH = 0.4
FALLBACK_CONFIDENCE = 0.3
TOP_K_MIN_SEGMENTS = 4
TOP_K_RATIO = 0.5

# Model cache to avoid reloading
_model_cache: Dict[str, Any] = {}
_device_cache: Optional[str] = None

def _log_debug(msg):
    if DEBUG_ASR:
        logger.debug(msg)

def _log_info(msg):
    if DEBUG_ASR:
        logger.info(msg)

def _get_device_info() -> str:
    """Get device info with caching to avoid repeated checks."""
    global _device_cache
    
    if _device_cache is not None:
        return _device_cache
    
    # Allow override
    device_env = os.getenv("WHISPER_DEVICE", None)
    if device_env in {"cpu", "cuda", "mps"}:
        _device_cache = device_env
        return _device_cache
    
    if TORCH_AVAILABLE and torch is not None:
        if torch.cuda.is_available():
            _device_cache = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _device_cache = "mps"
        else:
            _device_cache = "cpu"
    else:
        _device_cache = "cpu"
    
    return _device_cache

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



def _get_or_load_model(model_name: str, device: str, cache_dir: str):
    """Get cached model or load it if not in cache."""
    cache_key = f"{model_name}_{device}"
    
    if cache_key not in _model_cache:
        if not WHISPER_AVAILABLE or whisper is None:
            raise ImportError("Whisper is not available")
        
        _log_info(f"Loading Whisper model '{model_name}' on device '{device}' (first time, will be cached)")
        _model_cache[cache_key] = whisper.load_model(model_name, download_root=cache_dir, device=device)
        _log_debug(f"Model '{model_name}' loaded and cached with key '{cache_key}'")
    else:
        _log_debug(f"Using cached Whisper model '{cache_key}'")
    
    return _model_cache[cache_key]


def trim_wav_segment(src_wav: str, dst_wav: str, t_start: float = 0.0, t_dur: float = 20.0) -> bool:
    """Trim a WAV file to a segment using ffmpeg. Returns True if successful."""
    if not FFMPEG_AVAILABLE or ffmpeg is None:
        _log_debug("ffmpeg not available")
        return False
    
    try:
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
    """Get WAV file duration in seconds."""
    if not WAVE_AVAILABLE or wave is None:
        return 0.0
    
    try:
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
        if not WHISPER_AVAILABLE or whisper is None:
            raise ImportError("Whisper library is not installed")
        
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
        
        # Load or get cached model
        start = time.time()
        model = _get_or_load_model(used_model, used_device, cache_dir)
        load_time = time.time() - start
        _log_debug(f"Model ready in {load_time:.3f}s")
        
        # Transcribe
        transcribe_start = time.time()
        
        # Transcribe with language handling
        result = model.transcribe(
            wav_for_asr,
            language=lang,  # Can be None for auto-detection
            task="transcribe",
            verbose=False,
            no_speech_threshold=NO_SPEECH_THRESHOLD
        )
        transcribe_time = time.time() - transcribe_start
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
            logger.info(f"ASR completed in {elapsed:.2f}s (transcribe: {transcribe_time:.2f}s)")
        
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
            "stt_confidence": FALLBACK_CONFIDENCE,
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
                avg_logprob = DEFAULT_AVG_LOGPROB
            if no_speech_prob is None:
                no_speech_prob = DEFAULT_NO_SPEECH_PROB
            
            # Very permissive validation: segment is valid if it has text OR low no_speech_prob
            if seg_text or no_speech_prob < SEGMENT_VALIDATION_THRESHOLD:
                valid_segments.append(segment)
                lp_vals.append(avg_logprob)
                nsp_vals.append(no_speech_prob)
        
        # If there are still ZERO valid segments BUT text_len > 0 (we clearly transcribed something),
        # fabricate ONE pseudo-segment to prevent confidence from collapsing to fallback
        if not valid_segments and text_len > 0:
            # Create pseudo-segment with reasonable defaults
            lp_vals = [PSEUDO_SEGMENT_LOGPROB]  # Moderate confidence
            nsp_vals = [PSEUDO_SEGMENT_NO_SPEECH]  # Moderate speech probability
            valid_segments = [{"text": "pseudo", "avg_logprob": PSEUDO_SEGMENT_LOGPROB, "no_speech_prob": PSEUDO_SEGMENT_NO_SPEECH}]
        
        if not valid_segments:
            return FALLBACK_CONFIDENCE, {
                "avg_logprob_median": None,
                "avg_logprob_mean": None,
                "no_speech_prob_median": None,
                "no_speech_prob_min": None,
                "no_speech_prob_max": None,
                "no_speech_prob_mean": None,
                "lp_norm": None,
                "conf_final": FALLBACK_CONFIDENCE,
                "num_segments": 0,
                "device": used_device,
                "model": used_model,
                "used_topk": False,
                "k": 0
            }
        
        # Compute robust statistics using medians (efficient with statistics module)
        lp_med = statistics.median(lp_vals) if lp_vals else -1.0
        nsp_med = statistics.median(nsp_vals) if nsp_vals else 1.0
        
        # Also compute means for reporting
        lp_mean = sum(lp_vals) / len(lp_vals) if lp_vals else -1.0
        nsp_mean = sum(nsp_vals) / len(nsp_vals) if nsp_vals else 1.0
        nsp_min = min(nsp_vals) if nsp_vals else None
        nsp_max = max(nsp_vals) if nsp_vals else None
        
        # Normalize logprob from [LOGPROB_MIN_THRESHOLD, LOGPROB_MAX_THRESHOLD] to [0, 1] (clip before scaling)
        lp_clipped = max(LOGPROB_MIN_THRESHOLD, min(LOGPROB_MAX_THRESHOLD, lp_med))
        lp_norm = (lp_clipped - LOGPROB_MIN_THRESHOLD) / LOGPROB_RANGE
        
        # Final confidence (robust blend): weighted combination of logprob and no_speech_prob
        stt_conf_robust = max(0.0, min(1.0, CONFIDENCE_LOGPROB_WEIGHT * lp_norm + CONFIDENCE_NOSPEECH_WEIGHT * (1.0 - nsp_med)))
        
        # Optional top-k fallback if there are enough segments
        stt_conf_topk = stt_conf_robust
        used_topk = False
        k = 0
        
        if len(valid_segments) >= TOP_K_MIN_SEGMENTS:
            k = max(2, int((len(valid_segments) * TOP_K_RATIO) + 0.5))  # ceil equivalent
            # Get top-k segments by highest avg_logprob
            segment_lp_pairs = list(zip(valid_segments, lp_vals))
            segment_lp_pairs.sort(key=lambda x: x[1], reverse=True)  # Sort by logprob descending
            top_k_segments = segment_lp_pairs[:k]
            top_k_lp_vals = [pair[1] for pair in top_k_segments]
            
            # Compute median of top-k (efficient with statistics module)
            top_k_lp_med = statistics.median(top_k_lp_vals) if top_k_lp_vals else lp_med
            
            # Normalize top-k logprob
            top_k_lp_clipped = max(LOGPROB_MIN_THRESHOLD, min(LOGPROB_MAX_THRESHOLD, top_k_lp_med))
            top_k_lp_norm = (top_k_lp_clipped - LOGPROB_MIN_THRESHOLD) / LOGPROB_RANGE
            
            # Compute top-k confidence
            stt_conf_topk = max(0.0, min(1.0, CONFIDENCE_LOGPROB_WEIGHT * top_k_lp_norm + CONFIDENCE_NOSPEECH_WEIGHT * (1.0 - nsp_med)))
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
        return FALLBACK_CONFIDENCE, {
            "avg_logprob_median": None,
            "avg_logprob_mean": None,
            "no_speech_prob_median": None,
            "no_speech_prob_min": None,
            "no_speech_prob_max": None,
            "no_speech_prob_mean": None,
            "lp_norm": None,
            "conf_final": FALLBACK_CONFIDENCE,
            "num_segments": 0,
            "device": used_device,
            "model": used_model,
            "used_topk": False,
            "k": 0
        }

def is_asr_enabled() -> bool:
    """Check if ASR is enabled via environment variable."""
    return os.getenv("SPEECHUP_USE_ASR", "0") == "1"
