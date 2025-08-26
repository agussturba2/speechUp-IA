"""
Dynamic scoring system for SpeechUp analysis results.

Converts raw metrics into user-friendly scores (0-100) for:
- fluency, clarity, delivery_confidence, pronunciation, pace, engagement
"""

from __future__ import annotations
from typing import Dict, Any
import math


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    """Clamp value between lo and hi, handling NaN/inf gracefully."""
    try:
        if math.isnan(x) or math.isinf(x):
            return 0.0
    except Exception:
        pass
    return max(lo, min(hi, x))


def _get(d: Dict[str, Any], path: str, default: float = 0.0) -> float:
    """
    Safe getter for nested dicts: path like 'verbal.wpm'
    """
    cur: Any = d
    try:
        for p in path.split("."):
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        if cur is None:
            return default
        return float(cur)
    except Exception:
        return default


# --------------------------
# Normalizers (0-100)
# --------------------------
def _score_wpm(wpm: float) -> float:
    """
    Ideal band: 120–160 WPM -> 100
    Tapers linearly to 0 at 80 and 200.
    """
    if wpm <= 80 or wpm >= 200:
        return 0.0
    if 120 <= wpm <= 160:
        return 100.0
    if wpm < 120:
        # 80 -> 0 ; 120 -> 100
        return _clamp((wpm - 80) / (120 - 80) * 100)
    # wpm > 160: 160 -> 100 ; 200 -> 0
    return _clamp((200 - wpm) / (200 - 160) * 100)


def _score_fillers_per_min(x: float) -> float:
    """
    0 fillers/min -> 100 ; >=10 -> 0 (linear)
    """
    if x <= 0:
        return 100.0
    if x >= 10:
        return 0.0
    return _clamp((10 - x) / 10 * 100)


def _score_pause_rate_per_min(x: float) -> float:
    """
    <= 20 pausas/min -> ~100 ; >= 60 -> 0 (linear)
    """
    if x <= 20:
        return 100.0
    if x >= 60:
        return 0.0
    return _clamp((60 - x) / (60 - 20) * 100)


def _score_avg_pause_sec(x: float) -> float:
    """
    0.2s ideal (100). 1.5s o más -> 0. Linear in between.
    """
    if x <= 0.2:
        return 100.0
    if x >= 1.5:
        return 0.0
    return _clamp((1.5 - x) / (1.5 - 0.2) * 100)


def _score_pitch_range_st(x: float) -> float:
    """
    0 st -> 0, 12 st -> 100 (linear, cap at 12)
    """
    if x <= 0:
        return 0.0
    if x >= 12:
        return 100.0
    return _clamp(x / 12.0 * 100)


def _score_energy_cv(x: float) -> float:
    """
    0.0 -> 0 ; 0.30 -> 100 ; cap at 0.30
    """
    if x <= 0:
        return 0.0
    if x >= 0.30:
        return 100.0
    return _clamp(x / 0.30 * 100)


def _score_rhythm_consistency(x: float) -> float:
    """
    Expect [0..1]: 0 -> 0 ; 1 -> 100
    """
    return _clamp(x * 100.0)


def _score_gaze(g: float) -> float:
    """
    Prefer 70–90% mirando pantalla (triangular peak at 0.85).
    """
    g = max(0.0, min(1.0, g))
    # Triangle peak at 0.85
    peak = 0.85
    # distance from peak (0..1)
    d = abs(g - peak) / max(peak, 1 - peak)
    return _clamp((1 - d) * 100)


def _score_head_stability(x: float) -> float:
    """
    Expect [0..1]. >0.98 is great (~100). <0.85 is poor (~0).
    Linear mapping 0.85->0, 0.98->100 (clamped).
    """
    if x <= 0.85:
        return 0.0
    if x >= 0.98:
        return 100.0
    return _clamp((x - 0.85) / (0.98 - 0.85) * 100)


def _score_gesture_rate(gr: float) -> float:
    """
    0 -> 0 ; 10+ gestos/min -> 100 (cap)
    """
    if gr <= 0:
        return 0.0
    if gr >= 10:
        return 100.0
    return _clamp(gr / 10.0 * 100)


def _score_gesture_amplitude(a: float) -> float:
    """
    0.0 -> 0 ; 0.6 -> 100 (cap)
    """
    if a <= 0:
        return 0.0
    if a >= 0.6:
        return 100.0
    return _clamp(a / 0.6 * 100)


def _score_expression_variability(x: float) -> float:
    """
    Expect [0..1]. 0 -> 0 ; 1 -> 100
    """
    return _clamp(x * 100.0)


def _score_stt_conf(c: float) -> float:
    """
    STT confidence [0..1] -> [0..100]
    """
    return _clamp(c * 100.0)


# --------------------------
# Score aggregation
# --------------------------
def compute_scores(analysis: Dict[str, Any]) -> Dict[str, int]:
    """
    Build 6 simple user-friendly scores (0..100), integers:
      - fluency
      - clarity
      - delivery_confidence
      - pronunciation
      - pace
      - engagement
    """
    # Verbal
    wpm                 = _get(analysis, "verbal.wpm", 0.0)
    fillers_per_min     = _get(analysis, "verbal.fillers_per_min", 0.0)
    avg_pause_sec       = _get(analysis, "verbal.avg_pause_sec", 0.0)
    pause_rate_per_min  = _get(analysis, "verbal.pause_rate_per_min", 0.0)
    stt_conf            = _get(analysis, "verbal.stt_confidence", 0.0)

    # Prosody
    pitch_range_st      = _get(analysis, "prosody.pitch_range_semitones", 0.0)
    energy_cv           = _get(analysis, "prosody.energy_cv", 0.0)
    rhythm_consistency  = _get(analysis, "prosody.rhythm_consistency", 0.0)

    # Nonverbal
    gaze_screen_pct     = _get(analysis, "nonverbal.gaze_screen_pct", 0.0)     # 0..1
    head_stability      = _get(analysis, "nonverbal.head_stability", 0.0)      # 0..1
    gesture_rate        = _get(analysis, "nonverbal.gesture_rate_per_min", 0.0)
    gesture_amplitude   = _get(analysis, "nonverbal.gesture_amplitude", 0.0)
    expr_var            = _get(analysis, "nonverbal.expression_variability", 0.0) # 0..1

    # Component scores
    s_wpm       = _score_wpm(wpm)
    s_fillers   = _score_fillers_per_min(fillers_per_min)
    s_pause_rt  = _score_pause_rate_per_min(pause_rate_per_min)
    s_pause_avg = _score_avg_pause_sec(avg_pause_sec)

    s_pitch_rng = _score_pitch_range_st(pitch_range_st)
    s_energy    = _score_energy_cv(energy_cv)
    s_rhythm    = _score_rhythm_consistency(rhythm_consistency)

    s_gaze      = _score_gaze(gaze_screen_pct)
    s_head      = _score_head_stability(head_stability)
    s_g_rate    = _score_gesture_rate(gesture_rate)
    s_g_amp     = _score_gesture_amplitude(gesture_amplitude)
    s_expr      = _score_expression_variability(expr_var)

    s_stt       = _score_stt_conf(stt_conf)

    # Composite scores (weights sum to ~1.0 each)
    # Pace: how close to ideal WPM
    pace = s_wpm

    # Fluency: few fillers + reasonable pauses + decent pace
    fluency = (
        0.45 * s_fillers +
        0.25 * s_pause_rt +
        0.15 * s_pause_avg +
        0.15 * s_wpm
    )

    # Clarity: voice dynamics + rhythm + transcription confidence
    clarity = (
        0.30 * s_energy +
        0.30 * s_pitch_rng +
        0.25 * s_rhythm +
        0.15 * s_stt
    )

    # Delivery confidence: eye contact + head stability + vocal confidence + gestures
    base_confidence = (
        0.60 * s_head +
        0.40 * s_gaze
    )
    
    # Prosody rhythm bonus for high head stability
    rhythm = rhythm_consistency
    if rhythm >= 0.50 and head_stability >= 0.95:
        base_confidence = min(100.0, base_confidence + 5.0)
    
    delivery_confidence = base_confidence

    # Pronunciation: we proxy with STT confidence and (slightly) pace (too fast hurts)
    pronunciation = (
        0.80 * s_stt +
        0.20 * s_wpm
    )

    # Engagement: gestures + facial expression + vocal dynamics + gaze
    # Increased weights for gestures and expressions, reduced penalty for face coverage when gestures are active
    gesture_score = min(100.0, (gesture_rate / 8.0) * 100.0)  # 8+/min ~ 100
    expr_score = min(100.0, expr_var * 100.0)
    
    # Base face visibility score
    face_score = max(0.0, 100.0 * (gaze_screen_pct / 0.35))  # 35% face fills to 100
    face_score = min(face_score, 100.0)
    
    # Soften face penalty if gestures are active
    if gesture_rate >= 6.0:
        face_score = max(face_score, 70.0)
    
    # Combine with higher weights on gestures & expressions
    engagement = (
        0.45 * gesture_score +
        0.35 * expr_score +
        0.20 * face_score
    )

    out = {
        "fluency":              int(round(_clamp(fluency))),
        "clarity":              int(round(_clamp(clarity))),
        "delivery_confidence":  int(round(_clamp(delivery_confidence))),
        "pronunciation":        int(round(_clamp(pronunciation))),
        "pace":                 int(round(_clamp(pace))),
        "engagement":           int(round(_clamp(engagement))),
    }
    return out
