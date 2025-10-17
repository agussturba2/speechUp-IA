"""
Audio utilities for SpeechUp oratory analysis.
Provides VAD (Voice Activity Detection) and pause metrics computation.
"""
import os
os.environ.setdefault("SPEECHUP_DEBUG_VAD", "1")
import shutil
import wave
import tempfile
import logging
import math
import numpy as np
import soundfile as sf
import webrtcvad
import librosa
from typing import List, Tuple, Dict, Optional

from audio.prosody import apply_audio_prefiltering

logger = logging.getLogger(__name__)

DEBUG_VAD = os.getenv("SPEECHUP_DEBUG_VAD", "0") == "1"

def _log(msg: str):
    if DEBUG_VAD:
        print(f"[VAD] {msg}")

def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))

def _normalize_gain(y: np.ndarray, target_rms: float = 0.03) -> np.ndarray:
    """Simple gain normalization to avoid very low levels killing VAD."""
    cur = _rms(y)
    if cur <= 1e-9:
        return y
    g = target_rms / cur
    # avoid clipping
    y_out = np.clip(y * g, -1.0, 1.0)
    _log(f"rms_before={cur:.5f} gain={g:.2f} rms_after={_rms(y_out):.5f}")
    return y_out

def _frame_generator(y: np.ndarray, sr: int, frame_ms: int = 20):
    """Yield raw bytes frames (16-bit PCM) for webrtcvad."""
    frame_len = int(sr * frame_ms / 1000)
    if frame_len <= 0:
        frame_len = 320  # 20ms @ 16k
    # 16-bit PCM
    pcm = (y * 32768.0).astype(np.int16).tobytes()
    for i in range(0, len(pcm), frame_len * 2):
        chunk = pcm[i:i + frame_len * 2]
        if len(chunk) == frame_len * 2:
            yield chunk

def _collect_vad_segments(y: np.ndarray, sr: int, aggressiveness: int = 3, frame_ms: int = 20):
    """WebRTC-VAD segmentation with simple hangover."""
    vad = webrtcvad.Vad(int(aggressiveness))
    segments = []
    in_speech = False
    seg_start = 0.0
    t = 0.0
    hop_s = frame_ms / 1000.0

    for frame in _frame_generator(y, sr, frame_ms):
        is_speech = vad.is_speech(frame, sr)
        if is_speech and not in_speech:
            in_speech = True
            seg_start = t
        elif not is_speech and in_speech:
            in_speech = False
            segments.append({"start": float(seg_start), "end": float(t)})
        t += hop_s

    if in_speech:
        segments.append({"start": float(seg_start), "end": float(t)})

    # Merge tiny gaps and drop ultra-short segments
    merged = []
    min_dur = 0.08
    join_gap = 0.05
    for seg in segments:
        if not merged:
            merged.append(seg)
            continue
        if seg["start"] - merged[-1]["end"] <= join_gap:
            merged[-1]["end"] = seg["end"]
        else:
            merged.append(seg)
    merged = [s for s in merged if s["end"] - s["start"] >= min_dur]

    _log(f"webrtc_vad segments={len(merged)}")
    return merged

def _fallback_energy_segments(y: np.ndarray, sr: int):
    """Energy-based fallback using librosa.effects.split."""
    # top_db: lower → more sensitive
    intervals = librosa.effects.split(y, top_db=25, frame_length=1024, hop_length=256)
    segments = []
    for (s_i, e_i) in intervals:
        start = s_i / sr
        end = e_i / sr
        if end - start >= 0.12:
            segments.append({"start": float(start), "end": float(end)})
    _log(f"fallback_energy segments={len(segments)}")
    return segments

def extract_wav_mono_16k(video_path: str) -> Optional[str]:
    """
    Extract audio from video and convert to WAV (mono, 16kHz, 16-bit).
    
    Args:
        video_path: Path to input video file
        
    Returns:
        Path to temporary WAV file, or None if extraction fails
    """
    try:
        import ffmpeg
        if not check_ffmpeg():
            logger.warning("FFmpeg is not installed or not in PATH")
            return None
        # Create temporary WAV file
        fd, wav_path = tempfile.mkstemp(suffix='.wav')
        os.close(fd)
        
        # Extract audio: convert to mono, 16kHz, 16-bit PCM
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(
            stream, 
            wav_path,
            acodec='pcm_s16le',  # 16-bit PCM
            ac=1,                 # mono
            ar=16000,             # 16kHz sample rate
            loglevel='error'      # suppress ffmpeg output
        )
        
        # Run ffmpeg command
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)
        
        # Verify the file was created and has content
        if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
            logger.info(f"Successfully extracted audio to {wav_path}")
            return wav_path
        else:
            logger.warning("Audio extraction failed: output file is empty or missing")
            return None
            
    except ImportError:
        logger.warning("ffmpeg-python not available, skipping audio extraction")
        return None
    except Exception as e:
        logger.warning(f"Audio extraction failed: {e}")
        return None

def compute_vad_segments(wav_path: str):
    """
    Robust VAD:
    1) Load 16k mono wav
    2) Normalize gain
    3) Try WebRTC-VAD (aggr=1, 20ms). If empty -> fallback energy-based.
    """
    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    if y.size == 0:
        _log("empty audio")
        return []

    y = apply_audio_prefiltering(y, sr=sr)

    # Normalize and hard clip very low levels
    y = _normalize_gain(y, target_rms=0.02)

    segs = _collect_vad_segments(y, sr, aggressiveness=2, frame_ms=20)
    if not segs:
        _log("webrtc empty → fallback energy")
        segs = _fallback_energy_segments(y, sr)

    # clamp to file duration
    dur = len(y) / sr
    out = []
    for s in segs:
        st = max(0.0, min(float(s.get("start", 0.0)), dur))
        en = max(0.0, min(float(s.get("end", 0.0)), dur))
        if en > st:
            out.append({"start": st, "end": en})

    if not out and dur > 0.0:
        _log("no segments after cleanup → using full audio range")
        out.append({"start": 0.0, "end": float(dur)})

    if DEBUG_VAD:
        _log(f"final segments={len(out)} -> {out}")
    return out

def check_ffmpeg():
    """Check if FFmpeg is available in the system PATH."""
    return shutil.which("ffmpeg") is not None

def compute_pause_metrics(segments: list, total_dur: float):
    """
    Safe pause metrics: handle empty segments and non-finite durations.
    Returns:
      - avg_pause_sec
      - pause_rate_per_min
    """
    if not isinstance(segments, list):
        segments = []
    try:
        total_dur = float(total_dur)
    except Exception:
        total_dur = 0.0

    if total_dur <= 0:
        try:
            total_dur = max(float(s.get("end", 0.0)) for s in segments)
        except ValueError:
            total_dur = 0.0

    if total_dur <= 0:
        return {"avg_pause_sec": 0.0, "pause_rate_per_min": 0.0}

    # Sort + merge overlaps
    segs = []
    for s in segments:
        try:
            st = float(s.get("start"))
            en = float(s.get("end"))
            if en > st:
                segs.append((st, en))
        except Exception:
            continue
    segs.sort()

    merged = []
    for st, en in segs:
        if not merged or st > merged[-1][1]:
            merged.append([st, en])
        else:
            merged[-1][1] = max(merged[-1][1], en)

    speech_time = sum(en - st for st, en in merged)
    sil_time = max(0.0, total_dur - speech_time)

    # Approx: number of pauses = number of gaps between merged segments
    pauses = 0
    if merged:
        pauses = max(0, len(merged) - 1)
        # count leading/trailing silence as pauses if long enough
        if merged[0][0] >= 0.5:
            pauses += 1
        if total_dur - merged[-1][1] >= 0.5:
            pauses += 1

    avg_pause = sil_time / max(1, pauses) if pauses > 0 else 0.0
    rate_per_min = (pauses / total_dur) * 60.0 if total_dur > 0 else 0.0

    return {"avg_pause_sec": float(avg_pause), "pause_rate_per_min": float(rate_per_min)} 