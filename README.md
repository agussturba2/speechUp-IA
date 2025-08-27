# SpeechUp-IA — Multimodal Oratory Feedback (API + Pipeline)

SpeechUp-IA analyzes **video + audio** and returns **human-friendly feedback** about speaking performance: rhythm, pauses, filler words, prosody (pitch/energy variation), gaze, posture, head stability, gestures, and more. It also generates **labels** (e.g., "ritmo adecuado", "gestualidad activa") and **actionable advice** in Spanish (es-AR).

## TL;DR
- **Endpoint:** `POST /v1/feedback-oratoria` (multipart or JSON with media reference)
- **Default:** full analysis ON (audio + ASR + prosody + video/gestures)
- **Output:** scores (0–100), verbal & prosody metrics, nonverbal metrics, labels, recommendations, events[]

---

## Project Layout
```
├─ api/
│ ├─ main.py # FastAPI app bootstrap, default-on feature flags, logging
│ ├─ routers/
│ │ └─ oratory.py # /v1/feedback-oratoria endpoint integration
│ ├─ schemas/
│ │ └─ analysis_json.py # Pydantic model for the response (labels[], recommendations[])
│ ├─ core/
│ │ ├─ logging.py # JSON/text logging config
│ │ └─ middleware.py # request/response middleware
│ ├─ services/
│ │ └─ video_processor.py # video processing service layer
│ └─ websockets/ # real-time analysis support
│
├─ video/
│ ├─ pipeline.py # end-to-end pipeline orchestrating audio+video analysis
│ ├─ metrics.py # payload assembly, scores/labels/advice integration
│ ├─ scoring.py # dynamic scoring across 6 dimensions (0–100)
│ ├─ advice_generator.py # human-friendly labels + recommendations (es-AR)
│ ├─ audio_utils.py # audio extraction, VAD, pause metrics
│ ├─ realtime.py # real-time analysis pipeline
│ ├─ extract_frames.py # frame extraction utilities
│ └─ analysis/ # MediaPipe analysis modules
│
├─ audio/
│ ├─ asr.py # Whisper-based ASR + confidence & transcript short/full
│ ├─ prosody.py # pitch/energy/rhythm metrics using Librosa
│ ├─ text_metrics.py # WPM, filler detection, normalizations
│ ├─ audio_utils.py # WebRTC VAD + fallback energy-based segmentation
│ ├─ emotion.py # emotion detection from voice (experimental)
│ ├─ speech.py # speech processing utilities
│ └─ processors.py # audio processing pipeline
│
├─ utils/ # utility modules
├─ core/ # core functionality
├─ config.py # configuration management
├─ app.py # legacy app entry point
│
├─ tests/
│ ├─ demo_asr.py # smoke test ASR on sample video
│ ├─ demo_prosody.py # smoke test prosody on sample video
│ ├─ collect_all_metrics.py # batch metrics dump for test clips
│ ├─ test_labels.py # advice/labels unit tests
│ ├─ test_feature_flags.py # default-on flags coverage
│ ├─ test_prosody_integration.py # prosody pipeline tests
│ └─ sample*.mp4 # test video files
│
├─ model-es/ # Spanish language models (if present)
├─ cache/ # analysis result caching
├─ fixtures/ # test data and examples
├─ README.md
└─ requirements.txt
```

---

## Models & Why We Chose Them

### **🎤 ASR: OpenAI Whisper (`base` model)**
- **What it does**: Transcribes speech to text with high accuracy in Spanish
- **Why we chose it**: 
  - Excellent Spanish language support (es-AR)
  - CPU-optimized (`base` model: 74MB, fast inference)
  - Stable API with confidence scores
  - No internet connection required (offline processing)
- **How we use it**: 
  - Extracts full transcript and `transcript_short` (preview)
  - Computes `stt_confidence` from decoder probabilities as pronunciation proxy
  - Enables WPM calculation and Spanish filler word detection
  - Configurable via `SPEECHUP_ASR_MODEL` environment variable

### **🎵 Prosody: Librosa + Custom DSP Pipeline**
- **What it does**: Analyzes pitch variation, energy dynamics, and speech rhythm
- **Why we chose it**:
  - **Librosa**: Industry-standard audio analysis library with robust F0 estimation
  - **Custom pipeline**: Lightweight, deterministic, no heavy ML dependencies
  - **CPU-friendly**: Optimized for real-time processing on laptops
- **How we use it**:
  - **Pitch tracking**: `librosa.pyin` for robust fundamental frequency (F0) estimation
  - **Energy analysis**: RMS energy variation across audio frames
  - **Rhythm consistency**: Regularity of voiced/pause timing patterns
  - **Pre-filtering**: High-pass filter (80Hz) + spectral gating for noise reduction

### **👁️ Nonverbal: MediaPipe (TensorFlow Lite)**
- **What it does**: Detects face landmarks, body pose, hand positions, and facial expressions
- **Why we chose it**:
  - **Google's solution**: Production-ready, well-maintained, cross-platform
  - **TFLite runtime**: Fast CPU inference, no GPU required
  - **Comprehensive models**: Holistic (pose+hands), FaceMesh (468 landmarks), Face Detection
  - **Real-time capable**: 10+ FPS on typical laptop hardware
- **How we use it**:
  - **Face detection**: Ensures face visibility and positioning
  - **Landmark tracking**: 468 facial points for expression analysis
  - **Pose estimation**: Shoulder span for posture openness
  - **Hand tracking**: Motion magnitude for gesture detection

### **🔊 Voice Activity Detection: WebRTC VAD + Librosa Fallback**
- **What it does**: Separates speech from silence to compute pause metrics
- **Why we chose it**:
  - **WebRTC VAD**: Google's production VAD, highly accurate for speech detection
  - **Librosa fallback**: Energy-based segmentation when WebRTC fails
  - **Robust**: Handles noisy environments and varying audio quality
- **How we use it**:
  - **Primary**: WebRTC VAD with configurable aggressiveness
  - **Fallback**: Librosa energy-based segmentation with spectral analysis
  - **Output**: Speech segments → pause duration, pause rate, long pause detection

### **🎬 Video Processing: OpenCV + NumPy**
- **What it does**: Frame extraction, motion analysis, and gesture windowing
- **Why we chose it**:
  - **OpenCV**: Industry standard for computer vision, excellent Python bindings
  - **NumPy**: Fast numerical operations for motion magnitude calculations
  - **Cross-platform**: Works on Windows, macOS, and Linux
- **How we use it**:
  - **Frame extraction**: Sampled at ~10 FPS for performance
  - **Motion analysis**: Hand/body motion magnitude over time
  - **Gesture detection**: Hysteresis state machine with configurable thresholds

### **🔄 Audio Extraction: FFmpeg**
- **What it does**: Converts video to 16kHz mono WAV for audio analysis
- **Why we chose it**:
  - **Industry standard**: Most reliable audio/video conversion tool
  - **Format support**: Handles virtually any video format
  - **Quality**: Maintains audio fidelity during conversion
- **How we use it**:
  - **Format conversion**: Video → 16kHz mono WAV (optimal for speech analysis)
  - **Segment trimming**: Extracts specific time windows for analysis
  - **Error handling**: Graceful fallback if conversion fails

---

## Feature Flags (Default ON)

Flags are read via `_flag()` (default_on=True). If not set, they resolve to **True**.

- `SPEECHUP_USE_AUDIO` = 1 (extract audio + VAD)
- `SPEECHUP_USE_ASR` = 1 (Whisper transcription, WPM, fillers)
- `SPEECHUP_USE_PROSODY` = 1 (pitch/energy/rhythm)

**Gesture tuning (optional):**
- `SPEECHUP_GESTURE_MIN_AMP` (default `0.18`)
- `SPEECHUP_GESTURE_MIN_DUR` (default `0.08`)
- `SPEECHUP_GESTURE_COOLDOWN` (default `0.25`)
- `SPEECHUP_GESTURE_HYST_LOW_MULT` (default `0.55`)
- `SPEECHUP_GESTURE_MAX_SEG_S` (default `2.5`)
- `SPEECHUP_GESTURE_REQUIRE_FACE` (default `1`)

**Transcript controls (optional):**
- `SPEECHUP_INCLUDE_TRANSCRIPT` (default `0`)
- `SPEECHUP_TRANSCRIPT_PREVIEW_MAX` (default `1200` chars)

---

## End-to-End Flow

1) **API receive** (FastAPI)
   - `POST /v1/feedback-oratoria` → saves media and calls `video.pipeline.run_analysis_pipeline()`.

2) **Video pipeline**
   - Extract frames, detect face/landmarks, compute:
     - gaze %, head stability, posture openness, expression variability
     - gesture windows via **hysteresis state machine** + cooldown; events[] with {t, end_t, duration, amplitude, frame, confidence}
   - Extract audio (FFmpeg → 16k mono WAV)
     - VAD segments → pause metrics
     - Prosody → pitch/energy/rhythm
     - ASR (Whisper base) → transcript, WPM, fillers, stt_confidence

3) **Metrics → Scores → Labels/Advice**
   - `video/scoring.py`: 6 scores (0–100):
     - Fluency, Clarity, Delivery Confidence, Pronunciation, Pace, Engagement
   - `video/advice_generator.py`: labels[] + Spanish tips (balanced, deduplicated).
   - `video/metrics.py`: assemble final JSON, propagate `quality.debug` diagnostics.

---

## Technical Implementation Details

### **🎯 Core Architecture Principles**
- **Modular Design**: Each component (ASR, prosody, nonverbal) can be enabled/disabled independently
- **Graceful Degradation**: If one analysis fails, others continue and return safe defaults
- **Performance First**: Optimized for CPU-only laptops with real-time feedback
- **Spanish-First**: Tailored for es-AR speakers with appropriate thresholds and language support

### **⚡ Performance Optimizations**
- **Frame Sampling**: Video analyzed at ~10 FPS (vs 30+ FPS) for 3x speed improvement
- **Audio Windowing**: ASR limited to first 20s by default (configurable via `SPEECHUP_ASR_MAX_WINDOW_SEC`)
- **Model Selection**: Auto-selects optimal Whisper model based on device (CPU: `base`, GPU: `small`)
- **Caching**: Analysis results cached to avoid re-processing identical videos

### **🔧 Robustness Features**
- **Multi-Fallback VAD**: WebRTC VAD → Librosa energy → safe defaults
- **Gesture Hysteresis**: Prevents false positives with configurable amplitude/duration thresholds
- **Error Isolation**: ASR failures don't crash prosody analysis, prosody failures don't crash nonverbal
- **Input Validation**: Comprehensive Pydantic schemas ensure API response consistency

### **🌍 Language & Cultural Adaptation**
- **Spanish Fillers**: Detects "este", "eh", "o sea", "bueno", "mira" with accent normalization
- **WPM Thresholds**: Optimized for Spanish speaking patterns (120-160 WPM ideal range)
- **Advice Generation**: Argentine Spanish expressions and cultural context
- **Metric Tuning**: Thresholds calibrated for Spanish speech characteristics

---

## Metrics — Exact Definitions

### Verbal (audio/text)
- **WPM**: `words / (speech_duration_sec / 60)`  
  speech_duration_sec = sum of VAD speech segments.
- **Articulation rate (sps)**: `WPM * 2.3 / 60`  
  (2.3 ≈ avg Spanish syllables per word).
- **Fillers per min**: count of Spanish fillers (heuristic) normalized by speech minutes.
- **Avg pause (sec)**: mean duration of non-speech gaps between VAD segments.
- **Pause rate (/min)**: number of pauses ≥ 0.2–0.3s per minute (tuned with VAD).
- **Long pauses**: gaps ≥ `LONG_PAUSE_S` (default 0.8s) as `{start,end,duration}`.
- **Pronunciation score**: `stt_confidence` in `[0,1]` derived from Whisper segment stats.
- **Transcript**: `transcript_short` (preview), `transcript_full` optional by flag.

### Prosody
- **pitch_mean_hz**: average F0 over voiced frames.
- **pitch_range_semitones**: range(F0) mapped to semitones.
- **pitch_cv**: coefficient of variation of F0 (std/mean).
- **energy_cv**: coefficient of variation of short-term energy.
- **rhythm_consistency**: regularity of voiced/pause timing (0–1).

### Nonverbal (video)
- **gaze_screen_pct**: % frames with gaze toward camera/screen (proxy via facial orientation).
- **head_stability**: 1 − normalized landmark jitter; closer to 1 = steadier head.
- **posture_openness**: normalized shoulder/torso openness (0–1).
- **expression_variability**: variance of facial change over time (0–1).
- **gesture_rate_per_min**: `#confirmed_gesture_events / duration_min`  
  Events built by **hysteresis windowing**:
  - open when amplitude ≥ MIN_AMP; close when amplitude < HYST_LOW
  - enforce `MIN_DUR_S` ≤ duration ≤ `MAX_SEG_S`
  - apply time cooldown between starts
  - each event stores `{t, end_t, duration, amplitude, frame, confidence}`.

### Labels & Recommendations
- Deterministic thresholds → labels (e.g., "ritmo adecuado", "pausas regulares", "gestualidad activa", "bajo contraste de energía").
- Advice generator:
  - positive + constructive mix, es-AR phrasing
  - deduplication and area balancing (voz / comunicación / presencia)
  - caps on max labels/tips.

---

## Scores (0–100)

High-level (details in `video/scoring.py`):
- **Fluency**: WPM band + low fillers + regular pauses.
- **Pace**: target 120–160 WPM; penalize slow/fast.
- **Pronunciation**: `stt_confidence * 100`, clamped.
- **Clarity**: low fillers + stable pauses + decent gaze.
- **Engagement**: gestures/min + expression variability (+ small face coverage effect).
- **Delivery Confidence**: head stability + gaze (+ small prosody bonus if rhythm ≥ 0.5).

All scores are clamped to `[0,100]`.

---

## Run Locally

### 1) Install
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
FFmpeg must be available on PATH for audio extraction.

### 2) Start API (defaults ON)
```bash
# Optional: tune gestures while testing
$env:SPEECHUP_GESTURE_MIN_AMP="0.18"
$env:SPEECHUP_GESTURE_COOLDOWN="0.25"

uvicorn api.main:app --reload
```
You should see:
```
FEATURES -> USE_AUDIO=True USE_ASR=True USE_PROSODY=True
```

### 3) Test with a video

Postman → POST http://127.0.0.1:8000/v1/feedback-oratoria

Body: form-data with key `file` = your mp4

### 4) Smoke tests
```bash
python -m tests.demo_asr tests\sample2.mp4
python -m tests.demo_prosody tests\sample2.mp4
python -m tests.collect_all_metrics
```

---

## Troubleshooting

**ASR 0s?** Check logs for "ASR ok" vs "ASR failed". Set `$env:SPEECHUP_DEBUG_ASR="1"`.

**Few gestures?** Lower `SPEECHUP_GESTURE_MIN_AMP` (e.g., 0.16), reduce cooldown (0.20), or temporarily `SPEECHUP_GESTURE_REQUIRE_FACE=0`.

**Prosody zeros?** Confirm audio extracted and VAD segments found.

---

## License & Notes

Internal MVP for demo; thresholds and weights are tuned for Spanish speaking and typical laptop cameras on CPU. Model choices favor easy deployment and stable CPU performance.

---

## Changelog (MVP highlights)

- Default-ON feature gating (audio/ASR/prosody).
- Real ASR in pipeline; scoring driven by metrics (no hardcodes).
- Prosody: pitch/energy/rhythm with safe fallbacks.
- Gesture detection: hysteresis windowing + cooldown + max segment.
- Advice system: labels + diversified tips (es-AR), deduped and balanced.
- Diagnostics: per-stage timings, bucketed gesture coverage, long pauses, etc.

---

## Contact

Team SpeechUp-IA — reach out in the repo issues/PRs with clips and logs for tuning.
