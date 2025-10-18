"""
Robust prosody analysis module for SpeechUp.

Computes pitch, energy, and rhythm metrics from audio files with robust outlier handling.
Uses only voiced frames and applies median filtering to reduce spurious jumps.
Enhanced for flat speech detection and noisy room robustness.
"""

import logging
import numpy as np
import os
import librosa
from typing import Dict, List, Tuple, Optional

os.environ.setdefault("SPEECHUP_USE_PROSODY", "1")
os.environ.setdefault("SPEECHUP_DEBUG_PROSODY", "1")
os.environ.setdefault("SPEECHUP_PROSODY_PREFILTER", "1")
os.environ.setdefault("SPEECHUP_PROSODY_HPF_HZ", "80")
os.environ.setdefault("SPEECHUP_F0_MIN", "70")
os.environ.setdefault("SPEECHUP_F0_MAX", "350")
os.environ.setdefault("SPEECHUP_F0_PROFILE", "")
os.environ.setdefault("SPEECHUP_VOICED_PROB_THRESHOLD", "0.40")
os.environ.setdefault("SPEECHUP_PITCH_MEDIAN_WINDOW", "3")

DEBUG_PROS = os.getenv("SPEECHUP_DEBUG_PROSODY", "0") == "1"
def _plog(msg: str):
    if DEBUG_PROS:
        print(f"[PROS] {msg}")

logger = logging.getLogger(__name__)

# Shared constants for framing (frame-synchronous)
SR = 16000
HOP = 320      # 20 ms
WIN = 1024     # n_fft / frame_length

# Environment variable configuration with defaults
USE_PROSODY = int(os.getenv("SPEECHUP_USE_PROSODY", "1"))
PROSODY_PREFILTER = int(os.getenv("SPEECHUP_PROSODY_PREFILTER", "1"))
PROSODY_HPF_HZ = int(os.getenv("SPEECHUP_PROSODY_HPF_HZ", "80"))
F0_MIN = int(os.getenv("SPEECHUP_F0_MIN", "70"))
F0_MAX = int(os.getenv("SPEECHUP_F0_MAX", "350"))
F0_PROFILE = os.getenv("SPEECHUP_F0_PROFILE", "")
VOICED_PROB_THRESHOLD = float(os.getenv("SPEECHUP_VOICED_PROB_THRESHOLD", "0.40"))
PITCH_MEDIAN_WINDOW = int(os.getenv("SPEECHUP_PITCH_MEDIAN_WINDOW", "3"))
if PITCH_MEDIAN_WINDOW % 2 == 0:
    PITCH_MEDIAN_WINDOW += 1

# Adjust F0 bounds based on profile
if F0_PROFILE == "high":
    F0_MIN = 120
    F0_MAX = 450

logger.debug(f"Prosody config: USE_PROSODY={USE_PROSODY}, PREFILTER={PROSODY_PREFILTER}, "
            f"HPF_HZ={PROSODY_HPF_HZ}, F0_MIN={F0_MIN}, F0_MAX={F0_MAX}, PROFILE={F0_PROFILE}")
logger.debug(f"Framing: SR={SR}, HOP={HOP}, WIN={WIN}")


def load_wav_16k(wav_path: str) -> Tuple[np.ndarray, int, float]:
    """
    Load mono 16kHz PCM from path.
    
    Args:
        wav_path: Path to WAV file
        
    Returns:
        Tuple of (audio_array, sample_rate, duration_seconds)
    """
    try:
        import librosa
        y, sr = librosa.load(wav_path, sr=SR, mono=True)
        dur = len(y) / sr if sr > 0 else 0.0
        return y, sr, dur
    except Exception as e:
        logger.warning(f"Failed to load WAV file {wav_path}: {e}")
        return np.array([]), SR, 0.0


def apply_audio_prefiltering(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Apply lightweight pre-filtering: high-pass filter and spectral gating denoise.
    Maintains original length to avoid shape mismatches.
    
    Args:
        y: Audio array
        sr: Sample rate
        
    Returns:
        Pre-filtered audio array with same length
    """
    if not PROSODY_PREFILTER:
        logger.debug("Audio pre-filtering disabled via SPEECHUP_PROSODY_PREFILTER=0")
        return y
        
    try:
        import librosa
        import scipy.signal as signal
        
        # High-pass filter (80 Hz default) - use filtfilt to maintain length
        nyquist = sr / 2
        if PROSODY_HPF_HZ < nyquist:
            # Design Butterworth high-pass filter
            b, a = signal.butter(4, PROSODY_HPF_HZ / nyquist, btype='high')
            y_filtered = signal.filtfilt(b, a, y)
        else:
            y_filtered = y
        
        # Simple spectral gating denoise (conservative)
        # Compute spectral centroid as noise indicator - NEVER pass frame_length to spectral_centroid
        sc = librosa.feature.spectral_centroid(y=y_filtered, sr=sr, n_fft=WIN, hop_length=HOP, center=True)
        sc = sc.flatten()
        
        # Gate frames with very low spectral centroid (likely noise)
        noise_threshold = np.percentile(sc, 10)
        noise_mask = sc > noise_threshold
        
        # Apply gating by reducing amplitude of noisy frames
        # Convert frame-based mask to sample-based mask
        frame_length = WIN
        hop_length = HOP
        
        # Create sample-based noise mask
        sample_noise_mask = np.ones(len(y_filtered), dtype=bool)
        for i, is_noise in enumerate(noise_mask):
            start_sample = i * hop_length
            end_sample = min(start_sample + frame_length, len(y_filtered))
            sample_noise_mask[start_sample:end_sample] = is_noise
        
        # Apply gating while maintaining length
        y_denoised = y_filtered.copy()
        y_denoised[~sample_noise_mask] *= 0.3  # Reduce noise frames by 70%
        
        # Verify length preservation
        if len(y_denoised) != len(y):
            logger.warning(f"Prefilter changed length from {len(y)} to {len(y_denoised)}, using original")
            return y
        
        logger.debug(f"Audio pre-filtering: HPF={PROSODY_HPF_HZ}Hz, "
                    f"noise_gate_threshold={noise_threshold:.2f}")
        
        return y_denoised
        
    except Exception as e:
        logger.warning(f"Audio pre-filtering failed: {e}, using original audio")
        return y


def apply_median_filter_nan(data: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply median filter to data while ignoring NaN values.
    
    Args:
        data: Input array (may contain NaNs)
        window_size: Size of median filter window (must be odd)
        
    Returns:
        Filtered array with NaNs preserved
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
    
    half_window = window_size // 2
    filtered = np.full_like(data, np.nan)
    
    for i in range(len(data)):
        start = max(0, i - half_window)
        end = min(len(data), i + half_window + 1)
        window_data = data[start:end]
        
        # Only use non-NaN values in the window
        valid_data = window_data[~np.isnan(window_data)]
        if len(valid_data) > 0:
            filtered[i] = np.median(valid_data)
    
    return filtered


def compute_pitch_hz_robust(y: np.ndarray, sr: int = SR) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Compute robust pitch (F0) in Hz using pyin with yin fallback for low-energy speech.
    
    Args:
        y: Audio array (pre-filtered)
        sr: Sample rate (default: SR)
        
    Returns:
        Tuple of (f0_hz, voiced_mask, method_name) where voiced_mask indicates voiced frames
    """
    try:
        import librosa
        
        # Primary method: librosa.pyin with Hz parameters
        f0_hz, voiced_flag, voiced_prob = librosa.pyin(
            y, 
            fmin=70.0, 
            fmax=350.0, 
            sr=sr, 
            frame_length=WIN, 
            hop_length=HOP, 
            center=True,
            fill_na=np.nan
        )
        
        # Build voiced mask with configurable threshold for flat speech
        voiced_mask = (~np.isnan(f0_hz)) & (voiced_prob >= VOICED_PROB_THRESHOLD)
        voiced_count_pyin = np.sum(voiced_mask)
        
        logger.debug(f"PYIN: {voiced_count_pyin} voiced frames with threshold 0.60")
        
        # Fallback method: librosa.yin ONLY if no voiced frames with pyin
        if voiced_count_pyin == 0:
            f0_yin = librosa.yin(
                y, fmin=70.0, fmax=350.0,
                sr=sr, frame_length=WIN, hop_length=HOP, center=True
            )
            # yin returns array of F0 (Hz)
            voiced_mask_yin = f0_yin > 0
            voiced_count_yin = np.sum(voiced_mask_yin)
            
            logger.debug(f"YIN fallback: {voiced_count_yin} voiced frames")
            
            if voiced_count_yin > 0:
                f0_hz = f0_yin
                voiced_mask = voiced_mask_yin
                method_name = "yin"
                logger.debug(f"Selected YIN fallback method with {voiced_count_yin} voiced frames")
            else:
                method_name = "pyin"
                logger.debug("No voiced frames with either method")
        else:
            method_name = "pyin"
            logger.debug(f"Selected PYIN method with {voiced_count_pyin} voiced frames")
        
        if np.any(voiced_mask):
            # Apply median filter to reduce octave jumps
            f0_filtered = apply_median_filter_nan(f0_hz, window_size=PITCH_MEDIAN_WINDOW)
            
            # Recompute voiced mask after filtering
            if method_name == "pyin":
                voiced_mask = (~np.isnan(f0_filtered)) & (voiced_prob >= VOICED_PROB_THRESHOLD)
            else:
                voiced_mask = f0_filtered > 0
            
            # Log voiced frame statistics
            voiced_count = np.sum(voiced_mask)
            total_frames = len(f0_filtered)
            voiced_ratio = voiced_count / total_frames if total_frames > 0 else 0
            
            logger.debug(f"Pitch analysis ({method_name}): {voiced_count}/{total_frames} voiced frames "
                        f"({voiced_ratio*100:.1f}%), F0 bounds: {F0_MIN}-{F0_MAX}Hz")
            
            return f0_filtered, voiced_mask, method_name
        else:
            logger.warning("No voiced frames detected in audio with either method")
            return f0_hz, voiced_mask, method_name
        
    except Exception as e:
        logger.warning(f"Pitch computation failed: {e}")
        # Return empty arrays on failure
        return np.array([]), np.array([]), "failed"


def compute_energy_rms_aligned(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Compute RMS energy using same framing parameters as pitch analysis.
    
    Args:
        y: Audio array (pre-filtered)
        sr: Sample rate (default: SR)
        
    Returns:
        Array of RMS energy values
    """
    try:
        import librosa
        rms = librosa.feature.rms(y=y, frame_length=WIN, hop_length=HOP, center=True)[0]
        return rms
    except Exception as e:
        logger.warning(f"Energy computation failed: {e}")
        return np.array([])


def semitone_mapping(f0_hz: float) -> float:
    """
    Convert frequency to semitones using standard A4=440Hz reference.
    
    Args:
        f0_hz: Frequency in Hz
        
    Returns:
        Semitones above A0
    """
    return 12 * np.log2(f0_hz / 440.0) + 69


def compute_semitone_range_robust(f0_hz: np.ndarray, voiced_mask: np.ndarray) -> float:
    """
    Compute robust pitch range in semitones using percentiles.
    
    Args:
        f0_hz: Array of F0 values in Hz
        voiced_mask: Boolean mask indicating voiced frames
        
    Returns:
        Semitone range (90th percentile - 10th percentile)
    """
    try:
        # Extract voiced F0 values
        f0_voiced = f0_hz[voiced_mask]
        
        if len(f0_voiced) < 10:
            logger.debug(f"Insufficient voiced frames for semitone range: {len(f0_voiced)}")
            return 0.0
        
        # Convert to semitones using standard A4=440Hz reference with clipping
        st = 12 * np.log2(np.clip(f0_voiced, 1e-8, None) / 440.0) + 69
        
        # Use percentiles for robust range (90th - 10th)
        p90 = np.nanpercentile(st, 90)
        p10 = np.nanpercentile(st, 10)
        
        semitone_range = p90 - p10
        
        # Clip to [0, 36] for backward compatibility
        semitone_range = np.clip(semitone_range, 0, 36)
        
        logger.debug(f"Semitone range: P90={p90:.1f}, P10={p10:.1f}, range={semitone_range:.1f}")
        
        return float(semitone_range)
        
    except Exception as e:
        logger.warning(f"Semitone range computation failed: {e}")
        return 0.0


def compute_pitch_cv_robust(f0_hz: np.ndarray, voiced_mask: np.ndarray) -> float:
    """
    Compute robust pitch coefficient of variation on voiced frames only.
    
    Args:
        f0_hz: Array of F0 values in Hz
        voiced_mask: Boolean mask indicating voiced frames
        
    Returns:
        Coefficient of variation (std/mean) on voiced frames, clipped to [0, 1]
    """
    try:
        # Extract voiced F0 values
        f0_voiced = f0_hz[voiced_mask]
        
        if len(f0_voiced) < 5:
            logger.debug(f"Insufficient voiced frames for pitch CV: {len(f0_voiced)}")
            return 0.0
        
        # Remove any remaining NaNs
        f0_voiced = f0_voiced[~np.isnan(f0_voiced)]
        
        if len(f0_voiced) < 5:
            return 0.0
        
        mean_f0 = np.mean(f0_voiced)
        std_f0 = np.std(f0_voiced, ddof=1)
        
        if mean_f0 <= 0 or np.isnan(mean_f0):
            return 0.0
        
        pitch_cv = std_f0 / (mean_f0 + 1e-8)
        
        # Clamp to reasonable range [0, 1]
        pitch_cv = np.clip(pitch_cv, 0, 1)
        
        return float(pitch_cv)
        
    except Exception as e:
        logger.warning(f"Pitch CV computation failed: {e}")
        return 0.0


def compute_energy_cv_robust(rms_energy: np.ndarray, voiced_mask: np.ndarray) -> float:
    """
    Compute robust energy coefficient of variation on voiced frames in log-RMS domain.
    
    Args:
        rms_energy: Array of RMS energy values
        voiced_mask: Boolean mask indicating voiced frames
        
    Returns:
        Coefficient of variation on log-RMS energy, clipped to [0, 1]
    """
    try:
        # Extract voiced energy values
        energy_voiced = rms_energy[voiced_mask]
        
        if len(energy_voiced) < 5:
            logger.debug(f"Insufficient voiced frames for energy CV: {len(energy_voiced)}")
            return 0.0
        
        # Work in log-RMS domain for stability
        lrms = np.log(energy_voiced + 1e-8)
        
        # Remove any remaining NaNs
        lrms = lrms[~np.isnan(lrms)]
        
        if len(lrms) < 5:
            return 0.0
        
        mean_lrms = np.mean(lrms)
        std_lrms = np.std(lrms, ddof=1)
        
        if abs(mean_lrms) <= 1e-8:
            return 0.0
        
        energy_cv = std_lrms / (abs(mean_lrms) + 1e-8)
        
        # Clamp to reasonable range [0, 1]
        energy_cv = np.clip(energy_cv, 0, 1)
        
        return float(energy_cv)
        
    except Exception as e:
        logger.warning(f"Energy CV computation failed: {e}")
        return 0.0


def merge_vad_segments(segments: List[Tuple[float, float]], min_gap: float = 0.2, min_duration: float = 0.1) -> List[Tuple[float, float]]:
    """
    Merge VAD segments with small gaps and filter by minimum duration.
    
    Args:
        segments: List of (start, end) speech segments
        min_gap: Maximum gap to merge (seconds)
        min_duration: Minimum segment duration (seconds)
        
    Returns:
        Merged and filtered segments
    """
    if not segments:
        return []
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x[0])
    
    merged = []
    current_start, current_end = sorted_segments[0]
    
    for start, end in sorted_segments[1:]:
        gap = start - current_end
        
        if gap <= min_gap:
            # Merge segments
            current_end = end
        else:
            # Add current segment if it meets duration requirement
            if current_end - current_start >= min_duration:
                merged.append((current_start, current_end))
            # Start new segment
            current_start, current_end = start, end
    
    # Add final segment
    if current_end - current_start >= min_duration:
        merged.append((current_start, current_end))
    
    return merged


def rhythm_consistency_from_vad(segments, total_dur: float) -> float:
    """
    Compute a simple rhythm consistency score in [0..1].
    Approach: use coefficient of variation (CV = std/mean) of either
    speech segment DURATIONS or the GAPS between segments, whichever
    yields more samples. Consistency = 1 / (1 + CV).
    """
    # Guards
    try:
        total_dur = float(total_dur)
    except Exception:
        total_dur = 0.0

    if not isinstance(segments, list) or len(segments) == 0 or total_dur <= 0:
        _plog("rhythm: no segments or invalid total_dur -> 0.0")
        return 0.0

    # sanitize and sort
    clean = []
    for s in segments:
        try:
            st = float(s.get("start"))
            en = float(s.get("end"))
            if en > st and st >= 0.0 and en <= (total_dur + 1e-6):
                clean.append((st, en))
        except Exception:
            continue
    if not clean:
        _plog("rhythm: cleaned segments empty -> 0.0")
        return 0.0

    clean.sort(key=lambda x: x[0])

    # speech durations
    durs = [en - st for st, en in clean if (en - st) > 0]
    # inter-segment gaps (silences entre tramos de habla)
    gaps = []
    for i in range(1, len(clean)):
        prev_en = clean[i-1][1]
        cur_st  = clean[i][0]
        gap = max(0.0, cur_st - prev_en)
        gaps.append(gap)

    def _consistency_from(values):
        vals = np.asarray(values, dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0.0]
        if vals.size < 2:
            return None  # not enough data
        mean = float(np.mean(vals))
        std  = float(np.std(vals))
        if mean <= 0.0:
            return None
        cv = std / mean
        # map to [0..1], lower CV = higher consistency
        return 1.0 / (1.0 + cv)

    c_durs = _consistency_from(durs)
    c_gaps = _consistency_from(gaps)

    # Prefer the one with more samples; fallback to the other; else 0
    choice = None
    if c_durs is not None and c_gaps is not None:
        # pick based on which list had more samples
        choice = c_durs if len(durs) >= len(gaps) else c_gaps
    elif c_durs is not None:
        choice = c_durs
    elif c_gaps is not None:
        choice = c_gaps
    else:
        choice = 0.0

    # clamp
    score = float(max(0.0, min(1.0, choice)))
    _plog(f"rhythm: durs_n={len(durs)} gaps_n={len(gaps)} -> {score:.3f}")
    return score


def compute_prosody_metrics(y: np.ndarray, sr: int, segments: List[Tuple[float, float]], total_dur: float) -> Dict[str, float]:
    """
    Compute all prosody metrics from audio array and VAD segments with robust outlier handling.
    
    Args:
        y: Audio array (mono, 16kHz)
        sr: Sample rate
        segments: List of (start, end) speech segments (after merge/filter)
        total_dur: Total duration in seconds
        
    Returns:
        Dictionary with robust prosody metrics
    """
    if not USE_PROSODY:
        logger.debug("Prosody analysis disabled via SPEECHUP_USE_PROSODY=0")
        return {
            "pitch_mean_hz": 0.0,
            "pitch_range_semitones": 0.0,
            "pitch_cv": 0.0,
            "energy_cv": 0.0,
            "rhythm_consistency": 0.0
        }
    
    # Defensive checks
    if len(y) == 0:
        logger.warning("Empty audio array, returning default prosody metrics")
        return {
            "pitch_mean_hz": 0.0,
            "pitch_range_semitones": 0.0,
            "pitch_cv": 0.0,
            "energy_cv": 0.0,
            "rhythm_consistency": 0.0
        }
    
    if not segments:
        logger.warning("Empty segments, returning default prosody metrics")
        return {
            "pitch_mean_hz": 0.0,
            "pitch_range_semitones": 0.0,
            "pitch_cv": 0.0,
            "energy_cv": 0.0,
            "rhythm_consistency": 0.0
        }
    
    if total_dur <= 0:
        logger.warning("Invalid total duration, returning default prosody metrics")
        return {
            "pitch_mean_hz": 0.0,
            "pitch_range_semitones": 0.0,
            "pitch_cv": 0.0,
            "energy_cv": 0.0,
            "rhythm_consistency": 0.0
        }
    
    try:
        # Apply audio pre-filtering (or skip if disabled)
        y_filtered = apply_audio_prefiltering(y)
        
        # Compute robust pitch and energy with consistent framing
        f0_hz, voiced_mask, method_name = compute_pitch_hz_robust(y_filtered)
        rms_energy = compute_energy_rms_aligned(y_filtered)
        
        # Safety check for array lengths
        if len(f0_hz) == 0 or len(rms_energy) == 0:
            logger.warning("Pitch or energy computation failed, returning defaults")
            return {
                "pitch_mean_hz": 0.0,
                "pitch_range_semitones": 0.0,
                "pitch_cv": 0.0,
                "energy_cv": 0.0,
                "rhythm_consistency": 0.0
            }
        
        # Log lengths before truncation
        len_pyin = len(f0_hz) if method_name == "pyin" else 0
        len_yin = len(f0_hz) if method_name == "yin" else 0
        len_rms = len(rms_energy)
        logger.debug(f"Frame lengths: len_pyin={len_pyin}, len_yin={len_yin}, len_rms={len_rms}")
        
        # Align lengths BEFORE masking
        L = min(len(rms_energy), len(f0_hz))
        f0_hz = f0_hz[:L]
        voiced_mask = voiced_mask[:L]
        rms_energy = rms_energy[:L]
        
        logger.debug(f"Aligned to length L={L}")
        
        # Check if we have any voiced frames after masking
        if not np.any(voiced_mask):
            logger.warning("No voiced frames after masking, returning zeros")
            return {
                "pitch_mean_hz": 0.0,
                "pitch_range_semitones": 0.0,
                "pitch_cv": 0.0,
                "energy_cv": 0.0,
                "rhythm_consistency": 0.0
            }
        
        # Compute pitch metrics on voiced frames only
        f0_voiced = f0_hz[voiced_mask]
        pitch_mean_hz = float(np.nanmean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        pitch_range_semitones = compute_semitone_range_robust(f0_hz, voiced_mask)
        pitch_cv = compute_pitch_cv_robust(f0_hz, voiced_mask)
        
        # Compute energy CV on voiced frames only
        energy_cv = compute_energy_cv_robust(rms_energy, voiced_mask)
        
        # Rhythm (safe)
        try:
            rhythm = rhythm_consistency_from_vad(segments, total_dur)
        except Exception as e:
            _plog(f"rhythm exception: {e}")
            rhythm = 0.0
        
        # Log key metrics for debugging
        voiced_ratio = voiced_mask.mean() if L > 0 else 0
        
        logger.debug(f"Prosody summary ({method_name}): voiced_ratio={voiced_ratio:.3f}, "
                    f"st_range={pitch_range_semitones:.1f}, pitch_cv={pitch_cv:.3f}, "
                    f"energy_cv={energy_cv:.2f}, rhythm={rhythm:.2f}")
        
        # ensamblar respuesta (mantener las otras métricas calculadas arriba)
        out = {
            "pitch_mean_hz": float(pitch_mean_hz) if 'pitch_mean_hz' in locals() else 0.0,
            "pitch_range_semitones": float(pitch_range_semitones) if 'pitch_range_semitones' in locals() else 0.0,
            "pitch_cv": float(pitch_cv) if 'pitch_cv' in locals() else 0.0,
            "energy_cv": float(energy_cv) if 'energy_cv' in locals() else 0.0,
            "rhythm_consistency": float(rhythm),
        }
        _plog(f"prosody out: {out}")
        return out
        
    except Exception as e:
        logger.warning(f"Prosody computation failed: {e}, returning defaults")
        return {
            "pitch_mean_hz": 0.0,
            "pitch_range_semitones": 0.0,
            "pitch_cv": 0.0,
            "energy_cv": 0.0,
            "rhythm_consistency": 0.0
        }


def compute_prosody_metrics_from_path(wav_path: str, segments: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Wrapper function to compute prosody metrics from a WAV file path.
    
    Args:
        wav_path: Path to WAV file
        segments: List of (start, end) speech segments
        
    Returns:
        Dictionary with robust prosody metrics
    """
    try:
        y, sr, dur = load_wav_16k(wav_path)
        return compute_prosody_metrics(y, sr, segments, dur)
    except Exception as e:
        logger.warning(f"Failed to compute prosody metrics from path {wav_path}: {e}")
        return {
            "pitch_mean_hz": 0.0,
            "pitch_range_semitones": 0.0,
            "pitch_cv": 0.0,
            "energy_cv": 0.0,
            "rhythm_consistency": 0.0
        }
