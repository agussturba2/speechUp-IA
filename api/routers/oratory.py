"""API routers for oratory video/audio analysis endpoints."""
import logging, os, time, tempfile, shutil, requests
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from api.schemas.analysis_json import AnalysisJSON
from video.pipeline import run_analysis_pipeline
from video.metrics import build_metrics_response
from video.scoring import compute_scores
# from video.advice_generator import AdviceGenerator  # si lo usás

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Oratoria"])

def _download_to_tmp(url: str, suffix: str = ".mp4") -> str:
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
    except Exception:
        try: os.remove(tmp_path)
        except Exception: pass
        raise
    return tmp_path

@router.post("/v1/feedback-oratoria")
async def feedback_oratoria(
    media_url: Optional[str] = Form(None),
    video_file: Optional[UploadFile] = File(None),
):
    if not media_url and not video_file:
        raise HTTPException(400, "Provide media_url or video_file")

    tmp_path = None
    t0 = time.perf_counter()
    try:
        # 1) obtener archivo local
        if video_file:
            suffix = os.path.splitext(video_file.filename or "in.mp4")[1] or ".mp4"
            fd, tmp_path = tempfile.mkstemp(suffix=suffix); os.close(fd)
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(video_file.file, f)
        else:
            tmp_path = _download_to_tmp(media_url, ".mp4")

        # 2) pipeline
        proc = run_analysis_pipeline(tmp_path)

        # Normalize pipeline output for Pydantic validation
        # 1. Normalize fps to integer
        if isinstance(proc.get("fps"), float):
            proc["fps"] = int(round(proc["fps"]))
        
        # 2. Normalize events list structure
        events = proc.get("events", [])
        normalized_events = []
        for e in events:
            normalized_event = {
                "t": e.get("t") or e.get("time_sec") or e.get("timestamp", 0.0),
                "kind": e.get("kind") or e.get("type", "gesture"),
                "label": e.get("label"),
                "duration": e.get("duration")
            }
            # Optionally keep score and confidence if present
            if "score" in e:
                normalized_event["score"] = e["score"]
            if "confidence" in e:
                normalized_event["confidence"] = e["confidence"]
            normalized_events.append(normalized_event)
        
        # Build media dict with derived nonverbal metrics
        media = {
            "frames_total":       proc.get("frames_total", 0),
            "frames_with_face":   proc.get("frames_with_face", 0),
            "fps":                proc.get("fps", 0.0),
            "duration_sec":       proc.get("duration_sec", 0.0),
            "dropped_frames_pct": proc.get("dropped_frames_pct", 0.0),
            # Use new pipeline metrics directly
            "gaze_screen_pct":    proc.get("gaze_screen_pct", 0.0),
            "head_stability":     proc.get("head_stability", 0.0),
            "gesture_amplitude":  proc.get("gesture_amplitude", 0.0),
            "posture_openness":   proc.get("posture_openness", 0.0),
            "expression_variability": proc.get("expression_variability", 0.0),
            "engagement":         proc.get("engagement", 0.0),
            # eventos
            "gesture_events":     proc.get("gesture_events", 0),
        }
        
        gesture_events = media["gesture_events"]  # <- consistente con 'media'

 

        # 3) armar métricas y scores
        # Get audio availability and verbal/prosody data from pipeline
        pipeline_audio_available = proc.get("audio_available", False)
        pipeline_verbal = proc.get("verbal", {})
        pipeline_prosody = proc.get("prosody", {})
        
        result = build_metrics_response(
            feedbacks=[],
            media=media,                 # <- ¡no lo pises!
            gesture_events=gesture_events,
            events=normalized_events,
            analysis_ms=0,
            verbal=pipeline_verbal, 
            prosody=pipeline_prosody,
            audio_available=pipeline_audio_available,
        )
        # idioma por defecto
        result.setdefault("media", {}).setdefault("lang", "es-AR")

        # Inject pipeline values if present and > 0
        if proc.get("fps", 0) > 0:
            result["media"]["fps"] = proc["fps"]
        if proc.get("duration_sec", 0) > 0:
            result["media"]["duration_sec"] = proc["duration_sec"]
        if proc.get("frames_total", 0) > 0:
            result.setdefault("quality", {})["frames_analyzed"] = proc["frames_total"]

        # 4) analysis_ms SIEMPRE en quality (aquí, con tiempo total)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        q = result.setdefault("quality", {})
        q["analysis_ms"] = elapsed_ms
        q.setdefault("audio_available", pipeline_audio_available)
        
        # Add ASR debug info if enabled and available
        if os.getenv("SPEECHUP_DEBUG_ASR", "0") == "1" and pipeline_verbal:
            asr_debug = pipeline_verbal.get("debug", {})
            if asr_debug:
                q.setdefault("debug", {})["asr"] = asr_debug

        # Ensure all scores are integers (already computed in scoring.py)
        scores = result.get("scores", {})
        for k in ["fluency", "clarity", "delivery_confidence", "pronunciation", "pace", "engagement"]:
            if k in scores:
                scores[k] = int(round(scores[k]))
        result["scores"] = scores

        # INFO logs for propagated values
        logger.info(f"Propagated fps={result['media'].get('fps')}, duration_sec={result['media'].get('duration_sec')}, frames_analyzed={result['quality'].get('frames_analyzed')}, scores={result['scores']}")

        # Final normalization before validation
        if isinstance(result["media"].get("fps"), float):
            result["media"]["fps"] = int(round(result["media"]["fps"]))

        # 5) validar contrato y responder
        validated = AnalysisJSON.model_validate(result)
        return validated.model_dump()
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except Exception: pass
