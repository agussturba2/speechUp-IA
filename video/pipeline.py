# video/pipeline.py
import logging
import os
import tempfile
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, suppress

import cv2
import numpy as np
import mediapipe as mp

# Audio processing imports
import time
from .audio_utils import extract_wav_mono_16k, compute_vad_segments, compute_pause_metrics
from .scoring import compute_scores
from audio.asr import transcribe_wav
from audio.text_metrics import compute_wpm, detect_spanish_fillers, normalize_fillers_per_minute
from audio.prosody import compute_prosody_metrics_from_path

# Custom exceptions for better error handling
class AudioProcessingError(Exception):
    """Base exception for audio processing errors."""
    pass


class VADError(AudioProcessingError):
    """Voice Activity Detection failed."""
    pass


class ASRError(AudioProcessingError):
    """Automatic Speech Recognition failed."""
    pass


class ProsodyError(AudioProcessingError):
    """Prosody analysis failed."""
    pass


logger = logging.getLogger(__name__)


@contextmanager
def temp_wav_file(video_path: str):
    """
    Context manager for temporary WAV file extraction and cleanup.
    
    Args:
        video_path: Path to video file
        
    Yields:
        Path to temporary WAV file, or None if extraction fails
        
    Ensures:
        Temporary file is always cleaned up, even on errors
    """
    wav_path = None
    try:
        wav_path = extract_wav_mono_16k(video_path)
        yield wav_path
    finally:
        if wav_path and os.path.exists(wav_path):
            with suppress(OSError):
                os.remove(wav_path)
                logger.debug(f"Cleaned up temporary WAV: {wav_path}")


@dataclass
class PipelineConfig:
    """
    Centralized configuration for video analysis pipeline.
    
    All magic numbers and environment variables consolidated here for:
    - Better documentation
    - Easier testing
    - Type safety
    - Single source of truth
    """
    # Video processing
    sample_fps_target: int = 10  # Target sampling rate (~10 fps)
    warmup_sec: float = 0.5  # Ignore first N seconds for gesture detection
    
    # Gesture detection thresholds
    gesture_min_amp: float = 0.18  # Minimum motion amplitude to trigger gesture
    gesture_min_dur: float = 0.08  # Minimum gesture duration in seconds
    gesture_cooldown: float = 0.25  # Cooldown between gestures
    gesture_hyst_low_mult: float = 0.55  # Hysteresis low threshold multiplier
    gesture_max_seg: float = 2.5  # Maximum segment length (safety)
    gesture_require_face: bool = True  # Require face visibility for gestures
    
    # Posture analysis thresholds
    shoulder_span_min: float = 0.15  # Min normalized shoulder span
    shoulder_span_max: float = 0.35  # Max normalized shoulder span
    
    # Expression analysis thresholds
    max_smile_ratio: float = 2.0  # Maximum smile width/height ratio
    max_brow_distance: float = 0.1  # Maximum brow-eye distance
    
    # Gesture normalization constants
    gesture_big_baseline: float = 0.1  # "Big gesture" baseline for normalization
    gesture_rate_target: float = 10.0  # Target gestures/min for 1.0 score
    head_motion_scale: float = 4.0  # Head motion scaling factor
    face_center_scale: float = 0.5  # Face center distance scaling
    
    # Pause detection
    long_pause_sec: float = 0.8  # Threshold for "long pause" detection
    
    # Event limits
    max_events: int = 200  # Maximum events to return
    
    # Feature flags
    use_audio: bool = True
    use_asr: bool = True
    use_prosody: bool = True
    
    # Transcript settings
    include_full_transcript: bool = False
    transcript_preview_max: int = 1200
    
    # MediaPipe settings
    mp_model_complexity: int = 0  # 0=fastest, 2=most accurate
    mp_min_detection_confidence: float = 0.5
    mp_min_tracking_confidence: float = 0.5
    
    @classmethod
    def from_env(cls) -> 'PipelineConfig':
        """
        Load configuration from environment variables.
        Falls back to defaults if not set.
        """
        def _flag(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in ("1", "true", "yes", "on")
        
        return cls(
            # Gesture detection from env
            gesture_min_amp=float(os.getenv("SPEECHUP_GESTURE_MIN_AMP", "0.18")),
            gesture_min_dur=float(os.getenv("SPEECHUP_GESTURE_MIN_DUR", "0.08")),
            gesture_cooldown=float(os.getenv("SPEECHUP_GESTURE_COOLDOWN", "0.25")),
            gesture_hyst_low_mult=float(os.getenv("SPEECHUP_GESTURE_HYST_LOW_MULT", "0.55")),
            gesture_max_seg=float(os.getenv("SPEECHUP_GESTURE_MAX_SEG_S", "2.5")),
            gesture_require_face=_flag("SPEECHUP_GESTURE_REQUIRE_FACE", True),
            
            # Pause detection
            long_pause_sec=float(os.getenv("SPEECHUP_LONG_PAUSE_SEC", "0.8")),
            
            # Event limits
            max_events=int(os.getenv("SPEECHUP_MAX_EVENTS", "200")),
            
            # Feature flags
            use_audio=_flag("SPEECHUP_USE_AUDIO", True),
            use_asr=_flag("SPEECHUP_USE_ASR", True),
            use_prosody=_flag("SPEECHUP_USE_PROSODY", True),
            
            # Transcript settings
            include_full_transcript=_flag("SPEECHUP_INCLUDE_TRANSCRIPT", False),
            transcript_preview_max=int(os.getenv("SPEECHUP_TRANSCRIPT_PREVIEW_MAX", "1200")),
        )
    
    @property
    def gesture_hyst_low(self) -> float:
        """Computed hysteresis low threshold."""
        return self.gesture_hyst_low_mult * self.gesture_min_amp


def clamp(val: float, lo: float, hi: float) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, val))

def _get_face_center_from_holistic(face_landmarks) -> tuple:
    """Extract face center coordinates from holistic face landmarks."""
    if not face_landmarks:
        return None, None
    nose = face_landmarks.landmark[0]  # Nose tip as face center proxy
    return nose.x, nose.y

def compute_posture_openness(pose_landmarks, frame_width: int, config: PipelineConfig) -> float:
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
        
        # Map to posture_openness via clamp using config thresholds
        # This maps typical shoulder spans [min, max] to [0, 1]
        span_range = config.shoulder_span_max - config.shoulder_span_min
        posture_openness = clamp((shoulder_span_norm - config.shoulder_span_min) / span_range, 0.0, 1.0)
        
        return posture_openness
    except (IndexError, AttributeError):
        return 0.5  # Safe default

def compute_expression_variability(face_landmarks, config: PipelineConfig) -> float:
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
        
        # Simple normalization and combination using config thresholds
        # Normalize to [0, 1] ranges and combine with weights
        smile_norm = clamp(smile_ratio / config.max_smile_ratio, 0.0, 1.0)
        brow_norm = clamp(brow_raise / config.max_brow_distance, 0.0, 1.0)
        
        # Combine with weights: 60% smile, 40% brow
        expression_variability = 0.6 * smile_norm + 0.4 * brow_norm
        
        return clamp(expression_variability, 0.0, 1.0)
        
    except (IndexError, AttributeError):
        return 0.0  # Safe default


@dataclass
class VideoFrameResults:
    """Results from video frame processing."""
    frames_total: int
    frames_with_face: int
    sampled_frames: int
    accumulators: Dict[str, float]
    confirmed_events: List[Dict[str, Any]]
    gesture_candidates: int
    gesture_buckets: Dict[str, int]
    fps: float = 0.0
    duration_sec: float = 0.0


def _process_video_frames_from_path(
    video_path: str,
    config: PipelineConfig
) -> VideoFrameResults:
    """
    Process video frames for face detection, pose, hands, and gesture detection.
    Opens its own VideoCapture for parallel processing.
    
    Args:
        video_path: Path to video file
        config: Pipeline configuration
        
    Returns:
        VideoFrameResults with all accumulated metrics and events
    """
    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return VideoFrameResults(
            frames_total=0,
            frames_with_face=0,
            sampled_frames=0,
            accumulators={},
            confirmed_events=[],
            gesture_candidates=0,
            gesture_buckets={},
            fps=0.0,
            duration_sec=0.0
        )
    
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = (frame_count / fps) if (fps and frame_count) else 0.0
    frames_total = 0
    frames_with_face = 0
    sampled_frames = 0
    
    # Accumulators for nonverbal metrics
    accumulators = {
        'face_center_dist': 0.0,
        'head_motion': 0.0,
        'hand_motion_magnitude': 0.0,
        'posture_openness': 0.0,
        'expression_variability': 0.0,
    }
    prev_head = None
    prev_hands = None
    
    # Initialize MediaPipe models
    # Using single mp_holistic for face/pose/hands detection (eliminates redundant mp_face_mesh)
    mp_holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=config.mp_model_complexity,
        min_detection_confidence=config.mp_min_detection_confidence,
        min_tracking_confidence=config.mp_min_tracking_confidence
    )
    
    # Sampling configuration
    sample_every = max(int((fps or 30) // config.sample_fps_target), 1)
    idx = 0
    
    # Gesture detection state
    def _bucket_key(ts):
        """Convert timestamp to 5-second bucket key."""
        b0 = int(ts // 5) * 5
        return f"{b0}-{b0+5}"
    
    gesture_active = False
    gesture_start_idx = -1
    gesture_face_hits = 0
    gesture_sum_amp = 0.0
    gesture_n_amp = 0
    gesture_last_end_t = float("-inf")
    gesture_candidates = 0
    confirmed_events = []
    gesture_buckets = {}
    has_face = False
    
    # Main frame processing loop
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
        
        # Face detection from holistic
        has_face = hol.face_landmarks is not None
        if has_face:
            frames_with_face += 1
            nose = hol.face_landmarks.landmark[0]
            cx, cy = nose.x, nose.y
            accumulators['face_center_dist'] += float(np.hypot(cx - 0.5, cy - 0.5))
        
        # Posture analysis
        if hol.pose_landmarks:
            posture_openness_val = compute_posture_openness(hol.pose_landmarks, w, config)
            accumulators['posture_openness'] += posture_openness_val
            
            # Head motion tracking
            nose = hol.pose_landmarks.landmark[mp.solutions.holistic.PoseLandmark.NOSE]
            head = np.array([nose.x, nose.y], dtype=np.float32)
            if prev_head is not None:
                accumulators['head_motion'] += float(np.linalg.norm(head - prev_head))
            prev_head = head
        
        # Expression analysis using holistic face landmarks (468 landmarks available)
        if hol.face_landmarks:
            expression_var = compute_expression_variability(hol.face_landmarks, config)
            accumulators['expression_variability'] += expression_var
        
        # Hand motion and gesture detection
        left = hol.left_hand_landmarks
        right = hol.right_hand_landmarks
        
        if left or right:
            lx, ly = 0.0, 0.0
            rx, ry = 0.0, 0.0
            
            if left:
                lw = left.landmark[mp.solutions.holistic.HandLandmark.WRIST]
                lx, ly = lw.x, lw.y
            if right:
                rw = right.landmark[mp.solutions.holistic.HandLandmark.WRIST]
                rx, ry = rw.x, rw.y
            
            hands = np.array([lx, ly, rx, ry], dtype=np.float32)
            
            if prev_hands is not None:
                delta = float(np.linalg.norm(hands - prev_hands))
                accumulators['hand_motion_magnitude'] += delta
                
                t_sec = idx / (fps or 30.0)
                
                # Gesture hysteresis state machine
                if t_sec >= config.warmup_sec:
                    if not gesture_active:
                        if delta >= config.gesture_min_amp and (t_sec - gesture_last_end_t) >= config.gesture_cooldown:
                            gesture_active = True
                            gesture_start_idx = idx
                            gesture_face_hits = 1 if has_face else 0
                            gesture_sum_amp = delta
                            gesture_n_amp = 1
                    else:
                        gesture_face_hits += 1 if has_face else 0
                        gesture_sum_amp += delta
                        gesture_n_amp += 1
                        
                        if delta < config.gesture_hyst_low:
                            gesture_candidates += 1
                            end_idx = min(idx, gesture_start_idx + int(config.gesture_max_seg * (fps or 30.0)) - 1)
                            duration = (end_idx - gesture_start_idx + 1) / (fps or 30.0)
                            
                            if duration >= config.gesture_min_dur:
                                if not config.gesture_require_face or (gesture_face_hits / max(1, gesture_n_amp)) >= 0.5:
                                    start_t = gesture_start_idx / (fps or 30.0)
                                    end_t = (end_idx + 1) / (fps or 30.0)
                                    mean_amp = gesture_sum_amp / max(1, gesture_n_amp)
                                    
                                    if duration > config.gesture_max_seg + 0.05:
                                        logger.warning("Long gesture segment: %.2fs (start=%.1fs, end=%.1fs)",
                                                     duration, start_t, end_t)
                                    
                                    conf = max(0.0, min(1.0, (mean_amp - config.gesture_min_amp) / 
                                                   max(1e-6, (0.9 - config.gesture_min_amp))))
                                    
                                    confirmed_events.append({
                                        "t": float(start_t),
                                        "end_t": float(end_t),
                                        "duration": float(end_t - start_t),
                                        "frame": int(gesture_start_idx),
                                        "kind": "gesture",
                                        "label": "hand_motion",
                                        "amplitude": float(mean_amp),
                                        "score": None,
                                        "confidence": float(conf),
                                    })
                                    
                                    gesture_last_end_t = end_t
                                    bucket_key = _bucket_key(start_t)
                                    gesture_buckets[bucket_key] = gesture_buckets.get(bucket_key, 0) + 1
                            
                            gesture_active = False
                            gesture_start_idx = -1
                            gesture_face_hits = 0
                            gesture_sum_amp = 0.0
                            gesture_n_amp = 0
            
            prev_hands = hands
    
    # Process any remaining active gesture
    if gesture_active:
        gesture_candidates += 1
        end_idx = min(idx - 1, gesture_start_idx + int(config.gesture_max_seg * (fps or 30.0)) - 1)
        duration = (end_idx - gesture_start_idx + 1) / (fps or 30.0)
        
        if duration >= config.gesture_min_dur:
            if not config.gesture_require_face or (gesture_face_hits / max(1, gesture_n_amp)) >= 0.5:
                start_t = gesture_start_idx / (fps or 30.0)
                end_t = (end_idx + 1) / (fps or 30.0)
                mean_amp = gesture_sum_amp / max(1, gesture_n_amp)
                
                conf = max(0.0, min(1.0, (mean_amp - config.gesture_min_amp) / 
                           max(1e-6, (0.9 - config.gesture_min_amp))))
                
                confirmed_events.append({
                    "t": float(start_t),
                    "end_t": float(end_t),
                    "duration": float(end_t - start_t),
                    "frame": int(gesture_start_idx),
                    "kind": "gesture",
                    "label": "hand_motion",
                    "amplitude": float(mean_amp),
                    "score": None,
                    "confidence": float(conf),
                })
                
                bucket_key = _bucket_key(start_t)
                gesture_buckets[bucket_key] = gesture_buckets.get(bucket_key, 0) + 1
    
    # Release video capture
    cap.release()
    
    return VideoFrameResults(
        frames_total=frames_total,
        frames_with_face=frames_with_face,
        sampled_frames=sampled_frames,
        accumulators=accumulators,
        confirmed_events=confirmed_events,
        gesture_candidates=gesture_candidates,
        gesture_buckets=gesture_buckets,
        fps=fps,
        duration_sec=duration_sec
    )


def _process_audio_analysis(
    video_path: str,
    duration_sec: float,
    config: PipelineConfig
) -> Dict[str, Any]:
    """
    Process audio: extraction, VAD, ASR, prosody, and pause metrics.
    
    Args:
        video_path: Path to video file
        duration_sec: Video duration in seconds
        config: Pipeline configuration
        
    Returns:
        Dictionary with verbal and prosody metrics
    """
    result = {
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
        "audio_available": False,
        "asr_error": None,
    }
    
    if not (config.use_audio or config.use_prosody):
        return result
    
    segments = []
    
    with temp_wav_file(video_path) as wav_path:
        if not wav_path:
            logger.warning("Audio extraction failed for: %s", video_path)
            return result
        
        # VAD segments
        try:
            segments = compute_vad_segments(wav_path) or []
        except (IOError, RuntimeError, ValueError) as e:
            logger.warning("VAD failed: %s", e, exc_info=True)
            segments = []
        except Exception as e:
            logger.error("Unexpected VAD error: %s", e, exc_info=True)
            raise VADError(f"VAD computation failed: {e}") from e
        
        # Pause metrics
        pause_metrics = compute_pause_metrics(segments, duration_sec)
        result["verbal"].update({
            "avg_pause_sec": pause_metrics.get("avg_pause_sec", 0.0),
            "pause_rate_per_min": pause_metrics.get("pause_rate_per_min", 0.0),
            "long_pauses": pause_metrics.get("long_pauses", []),
        })
        
        # Prosody
        if config.use_prosody:
            try:
                prosody = compute_prosody_metrics_from_path(wav_path, segments)
                result["prosody"].update({
                    "pitch_mean_hz": float(prosody.get("pitch_mean_hz", 0.0)),
                    "pitch_range_semitones": float(prosody.get("pitch_range_semitones", 0.0)),
                    "pitch_cv": float(prosody.get("pitch_cv", 0.0)),
                    "energy_cv": float(prosody.get("energy_cv", 0.0)),
                    "rhythm_consistency": float(prosody.get("rhythm_consistency", 0.0)),
                })
                
                if result["prosody"]["pitch_mean_hz"] > 0:
                    logger.info(
                        "Prosody: pitch=%.1fHz, range=%.1fst, pitch_cv=%.2f, energy_cv=%.2f, rhythm=%.2f",
                        result["prosody"]["pitch_mean_hz"],
                        result["prosody"]["pitch_range_semitones"],
                        result["prosody"]["pitch_cv"],
                        result["prosody"]["energy_cv"],
                        result["prosody"]["rhythm_consistency"]
                    )
            except (IOError, RuntimeError, ValueError) as e:
                logger.warning("Prosody computation failed: %s", e, exc_info=True)
            except Exception as e:
                logger.error("Unexpected prosody error: %s", e, exc_info=True)
                raise ProsodyError(f"Prosody analysis failed: {e}") from e
        
        result["audio_available"] = True
        
        # ASR
        if (config.use_audio or config.use_asr) and wav_path:
            try:
                speech_dur_sec = sum((e.get("end", 0.0) - e.get("start", 0.0)) 
                                    for e in segments if e.get("end", 0.0) > e.get("start", 0.0)) if segments else 0.0
                logger.info("VAD segments: %s, speech_dur_sec: %.2fs", len(segments), speech_dur_sec)
                
                asr_result = transcribe_wav(wav_path, lang=None) or {}
                
                if speech_dur_sec < 0.1:
                    speech_dur_sec = asr_result.get("duration_sec", 0.0)
                
                if asr_result.get("ok"):
                    text = asr_result.get("text", "") or ""
                    dur = float(asr_result.get("duration_sec") or duration_sec or 0.0)
                    
                    wpm = compute_wpm(text, dur) if dur > 0 else 0.0
                    fillers = detect_spanish_fillers(text) or {"fillers_per_min": 0.0, "filler_counts": {}}
                    fillers_pm = normalize_fillers_per_minute(fillers.get("fillers_per_min", 0.0), dur) if dur > 0 else 0.0
                    
                    full_text = asr_result.get("text", "") or ""
                    result["verbal"].update({
                        "wpm": float(wpm),
                        "fillers_per_min": float(fillers_pm),
                        "filler_counts": fillers.get("filler_counts", {}),
                        "stt_confidence": float(asr_result.get("stt_confidence", 0.0)),
                        "transcript_len": len(full_text),
                        "transcript_short": full_text[:config.transcript_preview_max] if full_text else None,
                    })
                    
                    if config.include_full_transcript:
                        result["verbal"]["transcript_full"] = full_text
                    
                    # Articulation rate
                    syll_per_word_es = 2.3
                    result["verbal"]["articulation_rate_sps"] = float(max(0.0, (wpm * syll_per_word_es) / 60.0))
                    result["verbal"]["pronunciation_score"] = float(max(0.0, min(1.0, result["verbal"]["stt_confidence"])))
                    
                    # Long pauses from VAD
                    if segments and len(segments) > 1:
                        long_pauses = []
                        for i in range(len(segments) - 1):
                            gap_start = segments[i].get("end", 0.0)
                            gap_end = segments[i + 1].get("start", 0.0)
                            gap_duration = gap_end - gap_start
                            
                            if gap_duration >= config.long_pause_sec:
                                long_pauses.append({
                                    "start": float(gap_start),
                                    "end": float(gap_end),
                                    "duration": float(gap_duration)
                                })
                        
                        if long_pauses:
                            result["verbal"]["long_pauses"] = long_pauses
                else:
                    result["asr_error"] = asr_result.get("error", "asr_not_ok")
                    logger.warning("ASR not ok: %s", result["asr_error"])
            except (IOError, RuntimeError, ValueError) as e:
                result["asr_error"] = str(e)
                logger.warning("ASR transcription failed: %s", e, exc_info=True)
            except Exception as e:
                result["asr_error"] = str(e)
                logger.error("Unexpected ASR error: %s", e, exc_info=True)
                raise ASRError(f"ASR processing failed: {e}") from e
    
    return result


def run_analysis_pipeline(video_path: str, config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    """
    Run complete video analysis pipeline with optimized MediaPipe processing.
    
    Refactored architecture:
    - Modular processing with specialized functions
    - Centralized configuration management
    - Improved maintainability and testability
    
    Performance optimizations:
    - Uses single mp_holistic for face/pose/hands (eliminates redundant mp_face_mesh)
    - Parallel processing: video and audio analysis run concurrently
    - Organized accumulators for better data locality
    - Samples at ~10fps to balance accuracy vs speed
    
    Args:
        video_path: Path to video file
        config: Pipeline configuration (loads from env if None)
        
    Returns:
        Dictionary containing comprehensive analysis results
    """
    if config is None:
        config = PipelineConfig.from_env()
    
    t0 = time.time()
    
    # Quick video metadata check
    cap_check = cv2.VideoCapture(video_path)
    if not cap_check.isOpened():
        logger.warning("Could not open video: %s", video_path)
        cap_check.release()
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
    
    fps = float(cap_check.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = (frame_count / fps) if (fps and frame_count) else 0.0
    cap_check.release()
    
    logger.info(f"Starting parallel processing: video (frames) + audio (ASR/prosody)")
    
    # Parallel processing: video frames + audio analysis
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        video_future = executor.submit(_process_video_frames_from_path, video_path, config)
        audio_future = executor.submit(_process_audio_analysis, video_path, duration_sec, config)
        
        # Wait for both to complete
        video_results = video_future.result()
        audio_results = audio_future.result()
    
    logger.info(f"Parallel processing completed")
    
    # Extract video metrics
    frames_total = video_results.frames_total
    frames_with_face = video_results.frames_with_face
    sampled_frames = video_results.sampled_frames
    accumulators = video_results.accumulators
    confirmed_events = video_results.confirmed_events
    gesture_candidates = video_results.gesture_candidates
    gesture_buckets = video_results.gesture_buckets
    
    # Use fps and duration from video results (more accurate)
    fps = video_results.fps or fps
    duration_sec = video_results.duration_sec or duration_sec
    
    # Compute derived metrics from accumulators
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
    gaze_screen_pct = clamp(1.0 - (mean_face_center_dist / config.face_center_scale), 0.0, 1.0)
    
    # Head stability (inverted head motion)
    head_stability = clamp(1.0 - (config.head_motion_scale * normalized_head_motion), 0.0, 1.0)
    
    # Gesture amplitude: normalize hand motion magnitude to [0,1]
    gesture_amplitude = clamp(hand_motion_magnitude_avg / config.gesture_big_baseline, 0.0, 1.0)
    
    # Engagement metric: 60% gesture rate + 40% gesture amplitude
    # Normalize gesture rate using config target
    gesture_rate_norm = clamp(gesture_rate_per_min / config.gesture_rate_target, 0.0, 1.0)
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
    
    # Build result dictionary (audio_results already obtained from parallel execution)
    proc = {
        "frames_total": frames_total,
        "frames_with_face": frames_with_face,
        "fps": fps,
        "duration_sec": duration_sec,
        "dropped_frames_pct": dropped_frames_pct,
        "gesture_events": 0,  # Deprecated, kept for backward compatibility
        "events": confirmed_events[:config.max_events],
        "gesture_stats": gesture_stats,
        "media": {},
        # Derived nonverbal metrics
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
        # Audio metrics from audio processing
        "verbal": audio_results["verbal"],
        "prosody": audio_results["prosody"],
        "audio_available": audio_results.get("audio_available", False),
    }
    
    # Compute dynamic scores based on analysis results
    try:
        proc["scores"] = compute_scores(proc)
        logger.info("Dynamic scores computed successfully")
    except (KeyError, ValueError, TypeError) as e:
        logger.warning("Score computation failed, using defaults: %s", e, exc_info=True)
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
        "gesture_events_returned": min(len(confirmed_events), config.max_events),
        "gesture_buckets_5s": gesture_buckets,
        "coverage_pct": float(coverage_pct),
        "last_frame_ts": float(duration_sec),
        "face_present_ratio": float(frames_with_face) / max(frames_total, 1),
        "frames_with_face": gesture_stats["frames_with_face"],
        "max_events_config": config.max_events,
        "gesture_params": {
            "min_amp": config.gesture_min_amp,
            "min_dur": config.gesture_min_dur,
            "cooldown": config.gesture_cooldown,
            "require_face": config.gesture_require_face,
            "hyst_low": config.gesture_hyst_low,
            "hyst_low_mult": config.gesture_hyst_low_mult,
            "max_seg_s": config.gesture_max_seg
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
        config.gesture_min_amp,
        config.gesture_hyst_low,
        config.gesture_hyst_low_mult,
        config.gesture_min_dur,
        config.gesture_cooldown,
        config.gesture_max_seg,
        config.gesture_require_face
    )
    
    logger.info("Total pipeline time: %.1f ms", (time.time() - t0) * 1000)
    return proc
