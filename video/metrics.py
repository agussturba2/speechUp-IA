"""Build metrics response (migrated from video_processor.metrics)."""

from typing import Dict, List, Any
from .scoring import compute_scores
from .advice_generator import AdviceGenerator


def build_metrics_response(
    media: Dict,
    events: List[Dict],
    analysis_ms: int,
    verbal: Dict = None,
    prosody: Dict = None,
    audio_available: bool = False,
) -> Dict:
    """Construye la respuesta JSON con todas las métricas y feedback."""

    # Use proc values for nonverbal and quality metrics
    frames_total = media.get("frames_total", 0)
    frames_with_face = media.get("frames_with_face", 0)
    duration_sec = media.get("duration_sec", 0.0)
    fps = media.get("fps", 0)
    gesture_events = media.get("gesture_events", 0)
    dropped_frames_pct = media.get("dropped_frames_pct", 0.0)
    
    # Nonverbal metrics - DO NOT overwrite media dict
    face_coverage_pct = frames_with_face / frames_total if frames_total > 0 else 0.0
    
    # Face coverage smoothing: raise to minimum of 0.25 to reduce flicker
    if frames_total > 0 and frames_with_face > 0:
        face_coverage_pct = max(face_coverage_pct, 0.25)
    
    gesture_rate_per_min = gesture_events * 60 / duration_sec if duration_sec > 0 else 0.0
    
    import os
    try:
        min_floor = int(os.getenv("MIN_NONVERBAL_SCORE_FLOOR", "0"))
    except Exception:
        min_floor = 0
    floor_applied = False
    gesture_rate = round(gesture_rate_per_min, 2)
    gesture_ampl = media.get("gesture_amplitude", media.get("hand_motion_magnitude_avg", 0.0))
    
    if min_floor > 0:
        if gesture_rate == 0:
            gesture_rate = min_floor
            floor_applied = True
        if gesture_ampl == 0:
            gesture_ampl = min_floor
            floor_applied = True
    
    nonverbal = {
        "face_coverage_pct": round(face_coverage_pct, 2),
        "gaze_screen_pct": media.get("gaze_screen_pct", 0.0),
        "head_stability": media.get("head_stability", 0.0),
        "posture_openness": media.get("posture_openness", 0.0),
        "gesture_rate_per_min": gesture_rate,
        "gesture_amplitude": gesture_ampl,
        "expression_variability": media.get("expression_variability", 0.0),
        "engagement": media.get("engagement", 0.0),
    }
    
    if floor_applied:
        import logging
        logging.info(f"MIN_NONVERBAL_SCORE_FLOOR applied: {min_floor} to gesture_rate_per_min and/or gesture_amplitude")
    
    # Logging
    import logging
    logging.info(f"frames_total={frames_total}, frames_with_face={frames_with_face}, duration_sec={duration_sec}, fps={fps}, gesture_events={gesture_events}, face_coverage_pct={face_coverage_pct}")
    
    # Compute scores using the complete analysis structure
    analysis_data = {
        "verbal": verbal or {},
        "prosody": prosody or {},
        "nonverbal": nonverbal,
    }
    scores = compute_scores(analysis_data)
    
    # Generate recommendations
    nonverbal_tips = AdviceGenerator().generate_nonverbal_tips(nonverbal)
    verbal_tips = AdviceGenerator().generate_verbal_tips(verbal or {})
    prosody_tips = AdviceGenerator().generate_prosody_tips(prosody or {})
    
    # Combine all tips, ensuring at least one recommendation
    all_tips = nonverbal_tips + verbal_tips + prosody_tips
    if not all_tips:
        all_tips = ["Excelente presentación. Mantené este nivel de comunicación."]
    
    quality = {
        "frames_analyzed": frames_total,
        "dropped_frames_pct": dropped_frames_pct,
        "audio_snr_db": media.get("audio_snr_db", 0.0),
        "analysis_ms": analysis_ms,
        "audio_available": audio_available,
    }
    
    # Build response
    # Fill required blocks with defaults if missing
    default_verbal = {
        "wpm": 0.0,
        "articulation_rate_sps": 0.0,
        "fillers_per_min": 0.0,
        "filler_counts": {},
        "avg_pause_sec": 0.0,
        "pause_rate_per_min": 0.0,
        "long_pauses": [],
        "pronunciation_score": 0.0,
        "stt_confidence": 0.0,
    }
    default_prosody = {
        "pitch_mean_hz": 0.0,
        "pitch_range_semitones": 0.0,
        "pitch_cv": 0.0,
        "energy_cv": 0.0,
        "rhythm_consistency": 0.0,
    }
    default_lexical = {
        "lexical_diversity": 0.0,
        "cohesion_score": 0.0,
        "summary": "",
        "keywords": [],
    }
    
    # Prepare the complete data structure for labeling
    proc_data = {
        "verbal": {**default_verbal, **(verbal or {})},
        "prosody": {**default_prosody, **(prosody or {})},
        "nonverbal": nonverbal,
    }
    
    # Generate labels and additional recommendations
    advice_gen = AdviceGenerator()
    labels, extra_tips = advice_gen.generate_labels_and_tips(proc_data)
    
    # Convert existing tips to recommendation format
    existing_recommendations = [{"area": "communication", "tip": tip} for tip in all_tips]
    
    # Merge with new recommendations, avoiding duplicates by text content
    existing_tips_set = {(rec.get("area", ""), rec.get("tip", "")) for rec in existing_recommendations}
    
    for tip in extra_tips:
        tip_key = (tip.get("area", ""), tip.get("tip", ""))
        if tip_key not in existing_tips_set:
            existing_recommendations.append(tip)
            existing_tips_set.add(tip_key)
    
    response = {
        "id": "analysis_demo",
        "version": "1.0.0",
        "media": {
            "duration_sec": duration_sec,
            "lang": media.get('lang', 'es-AR'),
            "fps": fps,
        },
        "scores": scores,
        "verbal": {**default_verbal, **(verbal or {})},  # Merge partial verbal over defaults
        "prosody": {**default_prosody, **(prosody or {})},  # Merge partial prosody over defaults
        "lexical": default_lexical,
        "nonverbal": nonverbal,
        "events": events,
        "recommendations": existing_recommendations,
        "labels": labels,
        "quality": quality,
    }
    
    return response
