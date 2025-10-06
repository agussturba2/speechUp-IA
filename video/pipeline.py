# video/pipeline.py
import logging
import os
from typing import Dict, Any, List, Tuple
import math

import cv2
import numpy as np
import mediapipe as mp

# Audio processing imports
import time
from .audio_utils import extract_wav_mono_16k, compute_vad_segments, compute_pause_metrics
from .scoring import compute_scores
from audio.asr import transcribe_wav
from audio.text_metrics import compute_wpm, detect_spanish_fillers, normalize_fillers_per_minute
from audio.prosody import compute_prosody_metrics

logger = logging.getLogger(__name__)

def _flag(name: str, default_on: bool = True) -> bool:
    """
    Read feature flag from env with safe defaults and cast to boolean.
    default_on=True -> if env var missing, treat as enabled.
    Truthy values accepted: 1, true, yes, on (case-insensitive).
    """
    raw = os.getenv(name)
    if raw is None:
        return default_on
    return raw.strip().lower() in ("1", "true", "yes", "on")

def clamp(val: float, lo: float, hi: float) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, val))

def _get_face_center_from_holistic(face_landmarks) -> tuple:
    """Extract face center coordinates from holistic face landmarks."""
    if not face_landmarks:
        return None, None
    nose = face_landmarks.landmark[0]  # Nose tip as face center proxy
    return nose.x, nose.y

def compute_posture_openness(pose_landmarks, frame_width: int) -> float:
    """Compute posture openness based on shoulder span."""
    if not pose_landmarks:
        return 0.5  # Default neutral posture
    
    try:
        # Get shoulder landmarks
        left_shoulder = pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.RIGHT_SHOULDER]
        
        # Compute shoulder span in normalized coordinates
        shoulder_span = math.sqrt((right_shoulder.x - left_shoulder.x)**2 + (right_shoulder.y - left_shoulder.y)**2)
        
        # Normalize by frame width (assuming landmarks are in normalized coordinates)
        shoulder_span_norm = shoulder_span
        
        # Map to posture_openness via clamp: (shoulder_span_norm - 0.15) / (0.35 - 0.15)
        # This maps typical shoulder spans [0.15, 0.35] to [0, 1]
        posture_openness = clamp((shoulder_span_norm - 0.15) / (0.35 - 0.15), 0.0, 1.0)
        
        return posture_openness
    except (IndexError, AttributeError):
        return 0.5  # Safe default

def compute_expression_variability(face_landmarks) -> float:
    """Compute expression variability using FaceMesh landmarks."""
    if not face_landmarks:
        return 0.0
    
    try:
        # Smile score proxy: horizontal mouth distance / vertical mouth distance
        # Using mouth corner landmarks (61, 291) and top/bottom mouth (13, 14)
        left_corner = face_landmarks.landmark[61]
        right_corner = face_landmarks.landmark[291]
        top_mouth = face_landmarks.landmark[13]
        bottom_mouth = face_landmarks.landmark[14]
        
        # Horizontal mouth distance
        mouth_width = math.sqrt((right_corner.x - left_corner.x)**2 + (right_corner.y - left_corner.y)**2)
        # Vertical mouth distance
        mouth_height = math.sqrt((top_mouth.x - bottom_mouth.x)**2 + (top_mouth.y - bottom_mouth.y)**2)
        
        # Avoid division by zero
        if mouth_height > 0.001:
            smile_ratio = mouth_width / mouth_height
        else:
            smile_ratio = 1.0
        
        # Brow raise proxy: average distance between eyebrow and eye for both sides
        # Using eyebrow landmarks (70, 300) and eye landmarks (33, 133)
        left_brow = face_landmarks.landmark[70]
        right_brow = face_landmarks.landmark[300]
        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[133]
        
        left_brow_raise = math.sqrt((left_brow.x - left_eye.x)**2 + (left_brow.y - left_eye.y)**2)
        right_brow_raise = math.sqrt((right_brow.x - right_eye.x)**2 + (right_brow.y - right_eye.y)**2)
        brow_raise = (left_brow_raise + right_brow_raise) / 2.0
        
        # Simple normalization and combination
        # Normalize to [0, 1] ranges and combine with weights
        smile_norm = clamp(smile_ratio / 2.0, 0.0, 1.0)  # Assuming max ratio is ~2.0
        brow_norm = clamp(brow_raise / 0.1, 0.0, 1.0)   # Assuming max distance is ~0.1
        
        # Combine with weights: 60% smile, 40% brow
        expression_variability = 0.6 * smile_norm + 0.4 * brow_norm
        
        return clamp(expression_variability, 0.0, 1.0)
        
    except (IndexError, AttributeError):
        return 0.0  # Safe default

def run_analysis_pipeline(video_path: str) -> Dict[str, Any]:
    """
    Run complete video analysis pipeline with optimized MediaPipe processing.
    
    Performance optimizations:
    - Uses single mp_holistic for face/pose/hands (eliminates redundant mp_face)
    - Organized accumulators for better data locality
    - Samples at ~10fps to balance accuracy vs speed
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary containing comprehensive analysis results
    """
    t0 = time.time()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Could not open video: %s", video_path)
        return {
            "frames_total": 0,
            "frames_with_face": 0,
            "fps": 0.0,
            "duration_sec": 0.0,
            "dropped_frames_pct": 0.0,
            "gesture_events": 0,
            "events": [],
            "media": {},
        }

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = (frame_count / fps) if (fps and frame_count) else 0.0

    frames_total = 0
    frames_with_face = 0
    gesture_events = 0
    dropped_frames_pct = 0.0
    media = {}

    # Accumulators for nonverbal metrics (organized for clarity)
    accumulators = {
        'face_center_dist': 0.0,
        'head_motion': 0.0,
        'hand_motion_magnitude': 0.0,
        'posture_openness': 0.0,
        'expression_variability': 0.0,
    }
    prev_head = None
    prev_hands = None
    
    # Gesture event detection constants
    WARMUP_SEC = 0.5  # ignore first 0.5s
    mp_holistic = mp.solutions.holistic.Holistic(static_image_mode=False, model_complexity=0)
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Sample at ~10 fps for performance
    sample_every = max(int((fps or 30) // 10), 1)
    idx = 0
    sampled_frames = 0
    
    # Initialize gesture detection tracking
    def _bucket_key(ts):
        """Convert timestamp to 5-second bucket key."""
        b0 = int(ts // 5) * 5
        return f"{b0}-{b0+5}"
    
    # Gesture detection parameters with hysteresis
    MIN_AMP = float(os.getenv("SPEECHUP_GESTURE_MIN_AMP", "0.18"))
    MIN_DUR_S = float(os.getenv("SPEECHUP_GESTURE_MIN_DUR", "0.08"))
    COOLDOWN_S = float(os.getenv("SPEECHUP_GESTURE_COOLDOWN", "0.25"))
    REQUIRE_FACE = os.getenv("SPEECHUP_GESTURE_REQUIRE_FACE", "1").lower() in ("1", "true", "True", "yes", "on")
    MAX_EVENTS = int(os.getenv("SPEECHUP_MAX_EVENTS", "200"))
    HYST_LOW_MULT = float(os.getenv("SPEECHUP_GESTURE_HYST_LOW_MULT", "0.55"))
    HYST_LOW = HYST_LOW_MULT * MIN_AMP  # Hysteresis low threshold
    MAX_SEG_S = float(os.getenv("SPEECHUP_GESTURE_MAX_SEG_S", "2.5"))  # Safety max segment length
    LONG_PAUSE_S = float(os.getenv("SPEECHUP_LONG_PAUSE_SEC", "0.8"))
    
    # Gesture state machine variables
    gesture_active = False
    gesture_start_idx = -1
    gesture_face_hits = 0
    gesture_sum_amp = 0.0
    gesture_n_amp = 0
    gesture_last_end_t = float("-inf")
    gesture_candidates = 0
    confirmed_events = []
    gesture_buckets = {}
    
    # Face detection tracking
    has_face = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        frames_total += 1
        if idx % sample_every != 0:
            continue

        sampled_frames += 1
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Holistic analysis (pose + hands + face detection)
        hol = mp_holistic.process(rgb)
        
        # Face mesh for expression analysis
        face_mesh_res = mp_face_mesh.process(rgb)
        
        # Face detection from holistic face landmarks
        has_face = hol.face_landmarks is not None
        if has_face:
            frames_with_face += 1
            # Calculate face center from nose landmark
            nose = hol.face_landmarks.landmark[0]  # Nose tip
            cx, cy = nose.x, nose.y
            accumulators['face_center_dist'] += float(np.hypot(cx - 0.5, cy - 0.5))

        # Posture analysis
        if hol.pose_landmarks:
            posture_openness_val = compute_posture_openness(hol.pose_landmarks, w)
            accumulators['posture_openness'] += posture_openness_val
            
            # Head motion tracking
            nose = hol.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.NOSE]
            head = np.array([nose.x, nose.y], dtype=np.float32)
            if prev_head is not None:
                accumulators['head_motion'] += float(np.linalg.norm(head - prev_head))
            prev_head = head

        # Expression analysis
        if face_mesh_res.multi_face_landmarks:
            face_landmarks = face_mesh_res.multi_face_landmarks[0]
            expression_var = compute_expression_variability(face_landmarks)
            accumulators['expression_variability'] += expression_var

        # Hand motion and gesture detection with windowing FSM
        left = hol.left_hand_landmarks
        right = hol.right_hand_landmarks
        
        # Consider one hand if only one is visible (left or right)
        if left or right:
            # Initialize hand positions
            lx, ly = 0.0, 0.0
            rx, ry = 0.0, 0.0
            
            if left:
                lw = left.landmark[mp.solutions.holistic.HandLandmark.WRIST]
                lx, ly = lw.x, lw.y
            if right:
                rw = right.landmark[mp.solutions.holistic.HandLandmark.WRIST]
                rx, ry = rw.x, rw.y
            
            # Create hands array with current positions
            hands = np.array([lx, ly, rx, ry], dtype=np.float32)
            
            if prev_hands is not None:
                # Calculate motion delta
                delta = float(np.linalg.norm(hands - prev_hands))
                accumulators['hand_motion_magnitude'] += delta
                
                # Compute current time in seconds for this frame
                t_sec = idx / (fps or 30.0)
                
                # Gesture hysteresis state machine
                if t_sec >= WARMUP_SEC:
                    if not gesture_active:
                        # Check if we should start a new gesture window
                        if delta >= MIN_AMP and (t_sec - gesture_last_end_t) >= COOLDOWN_S:
                            # Open window
                            gesture_active = True
                            gesture_start_idx = idx
                            gesture_face_hits = 1 if has_face else 0
                            gesture_sum_amp = delta
                            gesture_n_amp = 1
                    else:
                        # Accumulate in active window
                        gesture_face_hits += 1 if has_face else 0
                        gesture_sum_amp += delta
                        gesture_n_amp += 1
                        
                        # Check if we should close the window (hysteresis)
                        if delta < HYST_LOW:
                            # Close window - process candidate
                            gesture_candidates += 1
                            end_idx = min(idx, gesture_start_idx + int(MAX_SEG_S * (fps or 30.0)) - 1)
                            duration = (end_idx - gesture_start_idx + 1) / (fps or 30.0)
                            
                            if duration >= MIN_DUR_S:
                                # Face ratio check at decision time
                                if not REQUIRE_FACE or (gesture_face_hits / max(1, gesture_n_amp)) >= 0.5:
                                    start_t = gesture_start_idx / (fps or 30.0)
                                    end_t = (end_idx + 1) / (fps or 30.0)
                                    mean_amp = gesture_sum_amp / max(1, gesture_n_amp)
                                    
                                    # Duration warning for debugging
                                    if duration > MAX_SEG_S + 0.05:
                                        logger.warning("Long gesture segment detected: %.2fs (start=%.1fs, end=%.1fs)", 
                                                     duration, start_t, end_t)
                                    
                                    # Confidence normalized above MIN_AMP
                                    conf = max(0.0, min(1.0, (mean_amp - MIN_AMP) / max(1e-6, (0.9 - MIN_AMP))))
                                    
                                    confirmed_events.append({
                                        "t": float(start_t),            # backward compatibility
                                        "end_t": float(end_t),
                                        "duration": float(end_t - start_t),
                                        "frame": int(gesture_start_idx),
                                        "kind": "gesture",
                                        "label": "hand_motion",
                                        "amplitude": float(mean_amp),
                                        "score": None,
                                        "confidence": float(conf),
                                    })
                                    
                                    # Update tracking
                                    gesture_last_end_t = end_t
                                    
                                    # Update bucket counts
                                    bucket_key = _bucket_key(start_t)
                                    gesture_buckets[bucket_key] = gesture_buckets.get(bucket_key, 0) + 1
                            
                            # Reset window state
                            gesture_active = False
                            gesture_start_idx = -1
                            gesture_face_hits = 0
                            gesture_sum_amp = 0.0
                            gesture_n_amp = 0
                    
            prev_hands = hands

    cap.release()
    
    # Process any remaining active gesture window
    if gesture_active:
        # Close final window with same logic
        gesture_candidates += 1
        end_idx = min(idx - 1, gesture_start_idx + int(MAX_SEG_S * (fps or 30.0)) - 1)  # Safety max
        duration = (end_idx - gesture_start_idx + 1) / (fps or 30.0)
        
        if duration >= MIN_DUR_S:
            # Face ratio check at decision time
            if not REQUIRE_FACE or (gesture_face_hits / max(1, gesture_n_amp)) >= 0.5:
                start_t = gesture_start_idx / (fps or 30.0)
                end_t = (end_idx + 1) / (fps or 30.0)
                mean_amp = gesture_sum_amp / max(1, gesture_n_amp)
                
                # Confidence normalized above MIN_AMP
                conf = max(0.0, min(1.0, (mean_amp - MIN_AMP) / max(1e-6, (0.9 - MIN_AMP))))
                
                confirmed_events.append({
                    "t": float(start_t),            # backward compatibility
                    "end_t": float(end_t),
                    "duration": float(end_t - start_t),
                    "frame": int(gesture_start_idx),
                    "kind": "gesture",
                    "label": "hand_motion",
                    "amplitude": float(mean_amp),
                    "score": None,
                    "confidence": float(conf),
                })
                
                # Update bucket counts
                bucket_key = _bucket_key(start_t)
                gesture_buckets[bucket_key] = gesture_buckets.get(bucket_key, 0) + 1

    # Compute derived metrics (optimized from accumulators)
    mean_face_center_dist = accumulators['face_center_dist'] / max(frames_with_face, 1)
    normalized_head_motion = accumulators['head_motion'] / max(frames_with_face, 1)
    hand_motion_magnitude_avg = accumulators['hand_motion_magnitude'] / max(sampled_frames, 1)
    posture_openness = accumulators['posture_openness'] / max(sampled_frames, 1)
    expression_variability = accumulators['expression_variability'] / max(sampled_frames, 1)
    
    # Comprehensive gesture statistics using confirmed events
    gesture_amplitudes = [e.get("amplitude", 0.0) for e in confirmed_events if e.get("amplitude")]
    gesture_stats = {
        "total_events": len(confirmed_events),
        "duration_sec": duration_sec,
        "rate_per_min": (len(confirmed_events) * 60) / max(duration_sec, 1e-6),
        "amplitude_mean": float(np.mean(gesture_amplitudes)) if gesture_amplitudes else 0.0,
        "amplitude_p95": float(np.percentile(gesture_amplitudes, 95)) if gesture_amplitudes else 0.0,
        "frames_with_face": frames_with_face,
        "frames_total": frames_total
    }
    
    # Calculate coverage percentage across 5-second buckets
    total_buckets = max(1, int(duration_sec // 5) + 1)
    buckets_with_events = sum(1 for count in gesture_buckets.values() if count > 0)
    coverage_pct = (buckets_with_events / total_buckets) * 100.0
    
    # Gesture rate per minute (for backward compatibility)
    gesture_rate_per_min = gesture_stats["rate_per_min"]
    
    # Gaze screen percentage (inverted face center distance)
    gaze_screen_pct = clamp(1.0 - (mean_face_center_dist / 0.5), 0.0, 1.0)
    
    # Head stability (inverted head motion)
    head_stability = clamp(1.0 - (4.0 * normalized_head_motion), 0.0, 1.0)
    
    # Gesture amplitude: normalize hand motion magnitude to [0,1]
    # Use 0.1 as "big gesture" baseline for normalization
    gesture_amplitude = clamp(hand_motion_magnitude_avg / 0.1, 0.0, 1.0)
    
    # Engagement metric: 60% gesture rate + 40% gesture amplitude
    # Normalize gesture rate by assuming 10 gestures per minute = 1.0
    gesture_rate_norm = clamp(gesture_rate_per_min / 10.0, 0.0, 1.0)
    engagement = 0.6 * gesture_rate_norm + 0.4 * gesture_amplitude
    
    # Improve posture_openness: clamp maximum to 0.95 to avoid saturation
    posture_openness = clamp(posture_openness, 0.0, 0.95)
    
    # Ensure all nonverbal metrics are normalized to [0,1]
    expression_variability = clamp(expression_variability, 0.0, 1.0)
    gaze_screen_pct = clamp(gaze_screen_pct, 0.0, 1.0)
    head_stability = clamp(head_stability, 0.0, 1.0)
    gesture_amplitude = clamp(gesture_amplitude, 0.0, 1.0)
    engagement = clamp(engagement, 0.0, 1.0)

    dropped_frames_pct = 0.0  # Placeholder for future implementation

    # Build result dictionary
    proc = {
        "frames_total": frames_total,
        "frames_with_face": frames_with_face,
        "fps": fps,
        "duration_sec": duration_sec,
        "dropped_frames_pct": dropped_frames_pct,
        "gesture_events": gesture_events,
        "events": confirmed_events[:MAX_EVENTS],  # Use confirmed events, truncate only at end
        "gesture_stats": gesture_stats,  # Include comprehensive stats
        "media": media,
        # Derived metrics (accumulators removed from output for cleaner API)
        "mean_face_center_dist": mean_face_center_dist,
        "normalized_head_motion": normalized_head_motion,
        "hand_motion_magnitude_avg": hand_motion_magnitude_avg,
        "posture_openness": posture_openness,
        "expression_variability": expression_variability,
        "gesture_rate_per_min": gesture_rate_per_min,
        "gaze_screen_pct": gaze_screen_pct,
        "head_stability": head_stability,
        "gesture_amplitude": gesture_amplitude,
        "engagement": engagement,
        "verbal": {
            "wpm": 0.0,
            "articulation_rate_sps": 0.0,
            "fillers_per_min": 0.0,
            "filler_counts": {},
            "avg_pause_sec": 0.0,
            "pause_rate_per_min": 0.0,
            "long_pauses": [],
            "pronunciation_score": 0.0,
            "stt_confidence": 0.0,
            "transcript_short": None,
        },
        "prosody": {
            "pitch_mean_hz": 0.0,
            "pitch_range_semitones": 0.0,
            "pitch_cv": 0.0,
            "energy_cv": 0.0,
            "rhythm_consistency": 0.0,
        },
    }
    

    # Audio processing for pause metrics (gated by SPEECHUP_USE_AUDIO)
    # Flags con default ON
    use_audio   = _flag("SPEECHUP_USE_AUDIO", default_on=True)
    use_asr     = _flag("SPEECHUP_USE_ASR", default_on=True)
    use_prosody = _flag("SPEECHUP_USE_PROSODY", default_on=True)

    wav_path = None
    segments = []
    
    if use_audio or use_prosody:
        t_audio0 = time.time()
        try:
            # Extract audio from video
            wav_path = extract_wav_mono_16k(video_path)
            
            if wav_path:
                # Compute VAD segments
                try:
                    segments = compute_vad_segments(wav_path) or []
                except Exception as e:
                    logger.warning("VAD failed: %s", e)
                    segments = []
                
                # Compute pause metrics
                pause_metrics = compute_pause_metrics(segments, duration_sec)
                
                # Update verbal metrics with pause data
                proc["verbal"].update({
                    "avg_pause_sec": pause_metrics.get("avg_pause_sec", 0.0),
                    "pause_rate_per_min": pause_metrics.get("pause_rate_per_min", 0.0),
                    "long_pauses": pause_metrics.get("long_pauses", []),
                })
                
                # Prosody processing (defaults already initialized)

                if use_prosody:
                    t_pro0 = time.time()
                    try:
                        from audio.prosody import compute_prosody_metrics_from_path
                        prosody = compute_prosody_metrics_from_path(wav_path, segments)
                        proc["prosody"].update({
                            "pitch_mean_hz": float(prosody.get("pitch_mean_hz", 0.0)),
                            "pitch_range_semitones": float(prosody.get("pitch_range_semitones", 0.0)),
                            "pitch_cv": float(prosody.get("pitch_cv", 0.0)),
                            "energy_cv": float(prosody.get("energy_cv", 0.0)),
                            "rhythm_consistency": float(prosody.get("rhythm_consistency", 0.0)),
                        })
                    except Exception as e:
                        logger.warning("Prosody failed, defaults applied: %s", e)

                proc["audio_available"] = True
                
                # Log prosody metrics if enabled
                if use_prosody and proc["prosody"].get("pitch_mean_hz", 0.0) > 0:
                    logger.info(
                        "Prosody metrics: pitch_mean=%.1fHz, range=%.1fst, pitch_cv=%.2f, "
                        "energy_cv=%.2f, rhythm=%.2f",
                        proc["prosody"].get("pitch_mean_hz", 0.0),
                        proc["prosody"].get("pitch_range_semitones", 0.0),
                        proc["prosody"].get("pitch_cv", 0.0),
                        proc["prosody"].get("energy_cv", 0.0),
                        proc["prosody"].get("rhythm_consistency", 0.0)
                    )
                
                # ASR processing (defaults + guarded)
                asr_error = None
                if (use_audio or use_asr) and wav_path:
                    t_asr0 = time.time()
                    try:
                        # Get VAD speech duration if available
                        speech_dur_sec = sum((e.get("end", 0.0) - e.get("start", 0.0)) for e in segments if e.get("end", 0.0) > e.get("start", 0.0)) if segments else 0.0
                        logger.info("VAD segments: %s, computed speech_dur_sec: %.2fs", len(segments), speech_dur_sec)
                        asr_result = transcribe_wav(wav_path, lang=None) or {}
                        
                        # Use trimmed duration if VAD not available
                        if speech_dur_sec < 0.1:
                            speech_dur_sec = asr_result.get("duration_sec", 0.0)
                        

                        if asr_result.get("ok"):
                            text = asr_result.get("text", "") or ""
                            dur  = float(asr_result.get("duration_sec") or duration_sec or 0.0)
                            
                            # WPM and fillers only if we have duration
                            wpm = compute_wpm(text, dur) if dur and dur > 0 else 0.0
                            fillers = detect_spanish_fillers(text) or {"fillers_per_min": 0.0, "filler_counts": {}}
                            fillers_pm = normalize_fillers_per_minute(fillers.get("fillers_per_min", 0.0), dur) if dur and dur > 0 else 0.0

                            # Transcript management configuration
                            include_full = os.getenv("SPEECHUP_INCLUDE_TRANSCRIPT", "0") in ("1", "true", "True", "yes", "on")
                            max_preview = int(os.getenv("SPEECHUP_TRANSCRIPT_PREVIEW_MAX", "1200"))
                            
                            full_text = asr_result.get("text", "") or ""
                            proc["verbal"].update({
                                "wpm": float(wpm),
                                "fillers_per_min": float(fillers_pm),
                                "filler_counts": fillers.get("filler_counts", {}),
                                "stt_confidence": float(asr_result.get("stt_confidence", 0.0)),
                                "transcript_len": len(full_text),
                                "transcript_short": full_text[:max_preview] if full_text else None,
                            })
                            
                            # Include full transcript if enabled
                            if include_full:
                                proc["verbal"]["transcript_full"] = full_text

                            # Fill missing verbal metrics
                            syll_per_word_es = 2.3  # Spanish average syllables per word
                            proc["verbal"]["articulation_rate_sps"] = float(max(0.0, (wpm * syll_per_word_es) / 60.0))
                            proc["verbal"]["pronunciation_score"] = float(max(0.0, min(1.0, proc["verbal"].get("stt_confidence", 0.0))))
                            
                            # Derive long_pauses from VAD segments
                            if segments and len(segments) > 1:
                                long_pauses = []
                                for i in range(len(segments) - 1):
                                    gap_start = segments[i].get("end", 0.0)
                                    gap_end = segments[i + 1].get("start", 0.0)
                                    gap_duration = gap_end - gap_start
                                    
                                    if gap_duration >= LONG_PAUSE_S:
                                        long_pauses.append({
                                            "start": float(gap_start),
                                            "end": float(gap_end),
                                            "duration": float(gap_duration)
                                        })
                                
                                if long_pauses:
                                    proc["verbal"]["long_pauses"] = long_pauses

                        else:
                            asr_error = asr_result.get("error", "asr_not_ok")
                            logger.warning("ASR not ok: %s", asr_error)
                    except Exception as e:
                        asr_error = str(e)
                        logger.exception("ASR stage failed: %s", e)

            else:
                proc["audio_available"] = False
                
        except Exception as e:
            logger.warning(f"Audio processing failed: {e}")
            proc["audio_available"] = False
        finally:
            # Always cleanup temporary WAV file
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass
    
    # Compute dynamic scores based on analysis results
    try:
        proc["scores"] = compute_scores(proc)
        logger.info("Dynamic scores computed successfully")
    except Exception as e:
        logger.warning("Score computation failed, using defaults: %s", e)
        # Fallback to default scores if computation fails
        proc["scores"] = {
            "fluency": 65,
            "clarity": 65,
            "delivery_confidence": 65,
            "pronunciation": 65,
            "pace": 65,
            "engagement": 65,
        }
    
    # Surface ASR error and gesture statistics in quality block
    q = proc.setdefault("quality", {})
    dbg = q.setdefault("debug", {})
    
    # Asegurarse de que asr_error esté definida antes de usarla
    if 'asr_error' in locals() and asr_error:
        dbg["asr_error"] = asr_error
    
    # Add comprehensive gesture diagnostics to debug
    dbg.update({
        "gesture_events_total": gesture_stats["total_events"],
        "gesture_candidates_total": gesture_candidates,
        "gesture_events_returned": min(len(confirmed_events), MAX_EVENTS),
        "gesture_buckets_5s": gesture_buckets,
        "coverage_pct": float(coverage_pct),
        "last_frame_ts": float(duration_sec),
        "face_present_ratio": float(frames_with_face) / max(frames_total, 1),
        "frames_with_face": gesture_stats["frames_with_face"],
        "max_events_config": MAX_EVENTS,
        "gesture_params": {
            "min_amp": MIN_AMP,
            "min_dur": MIN_DUR_S,
            "cooldown": COOLDOWN_S,
            "require_face": REQUIRE_FACE,
            "hyst_low": HYST_LOW,
            "hyst_low_mult": HYST_LOW_MULT,
            "max_seg_s": MAX_SEG_S
        }
    })
    
    # Log gesture detection summary
    last_event_time = max([e.get("end_t", 0.0) for e in confirmed_events], default=0.0)
    logger.info(
        "GESTURES -> total=%d, candidates=%d, rate=%.2f/min, coverage=%.1f%%, last_event=%.1fs",
        len(confirmed_events),
        gesture_candidates,
        gesture_stats["rate_per_min"],
        coverage_pct,
        last_event_time
    )
    
    # Log gesture parameters
    logger.info(
        "GESTURE PARAMS -> min_amp=%.3f, low=%.3f (%.2fx), min_dur=%.3fs, cooldown=%.3fs, max_seg=%.1fs, require_face=%s",
        MIN_AMP,
        HYST_LOW,
        HYST_LOW_MULT,
        MIN_DUR_S,
        COOLDOWN_S,
        MAX_SEG_S,
        REQUIRE_FACE
    )
    
    logger.info("Total pipeline time: %.1f ms", (time.time() - t0) * 1000)
    return proc
