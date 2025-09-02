# tests/collect_all_metrics.py
# -----------------------------------------------------------------------------
# Recolecta métricas de Audio/ASR/Prosodia para todos los videos en ./tests
# y guarda un CSV "metrics_all.csv" en el root del repo.
#
# Requisitos (ya en el proyecto):
# - video/audio_utils.py : extract_wav_mono_16k, compute_vad_segments, compute_pause_metrics
# - audio/asr.py         : transcribe_wav
# - audio/text_metrics.py: compute_wpm, detect_spanish_fillers, normalize_fillers_per_minute
# - audio/prosody.py     : compute_prosody_metrics_from_path(wav_path, segments)
#
# Uso:
#   .\.venv\Scripts\Activate.ps1
#   $env:SPEECHUP_USE_AUDIO="1"
#   $env:SPEECHUP_USE_PROSODY="1"
#   python -m tests.collect_all_metrics
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import sys
import csv
import contextlib
import wave
from typing import Dict, Any, List, Optional

# --- Imports del proyecto ---
from video.audio_utils import (
    extract_wav_mono_16k,
    compute_vad_segments,
    compute_pause_metrics,
)
from audio.asr import transcribe_wav
from audio.text_metrics import (
    compute_wpm,
    detect_spanish_fillers,
    normalize_fillers_per_minute,
)
from audio.prosody import compute_prosody_metrics_from_path


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS_DIR = os.path.join(ROOT, "tests")
OUT_CSV = os.path.join(ROOT, "metrics_all.csv")


def _get_wav_duration_sec(wav_path: str) -> float:
    """Duración precisa del wav mono 16k."""
    try:
        with contextlib.closing(wave.open(wav_path, "rb")) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 16000
            return float(frames) / float(rate) if rate else 0.0
    except Exception:
        return 0.0


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _sanitize_segments(raw: Any) -> List[Dict[str, float]]:
    """
    Acepta cualquier cosa que venga del VAD y devuelve una lista estandarizada:
    [{"start": float, "end": float}, ...] con end > start.
    """
    out: List[Dict[str, float]] = []
    if not raw:
        return out

    if isinstance(raw, dict):
        raw = raw.get("segments") or raw.get("data") or []

    if not isinstance(raw, (list, tuple)):
        return out

    for item in raw:
        try:
            # Soportar diferentes claves
            st = item.get("start", item.get("t0", item.get("t_start", None)))
            en = item.get("end", item.get("t1", item.get("t_end", None)))
            st = float(st) if st is not None else None
            en = float(en) if en is not None else None
            if st is not None and en is not None and en > st:
                out.append({"start": st, "end": en})
        except Exception:
            # Ignorar ítems corruptos
            continue
    return out


def analyze_video(video_path: str) -> Dict[str, Any]:
    """Corre extracción de audio, VAD/pausas, ASR, prosodia y devuelve un dict de métricas."""
    row: Dict[str, Any] = {
        "video": os.path.basename(video_path),
        # Duración total (wav o ASR), se completa más abajo
        "duration_sec": "",
        # Pausas (VAD)
        "speech_segments": 0,
        "avg_pause_sec": 0.0,
        "pause_rate_per_min": 0.0,
        # ASR
        "wpm": 0.0,
        "fillers_per_min": 0.0,
        "stt_confidence": 0.0,
        "transcript_short": "",
        # Prosodia
        "pitch_mean_hz": 0.0,
        "pitch_range_semitones": 0.0,
        "pitch_cv": 0.0,
        "energy_cv": 0.0,
        "rhythm_consistency": 0.0,
        # Debug ASR device/model (si retorna)
        "asr_device": "",
        "asr_model": "",
    }

    if not os.path.exists(video_path):
        print(f"[WARN] Video no existe: {video_path}")
        return row

    # 1) WAV temporal mono 16k
    wav_path: Optional[str] = None
    try:
        wav_path = extract_wav_mono_16k(video_path)
    except Exception as e:
        print(f"[ERROR] extract_wav_mono_16k falló para {os.path.basename(video_path)}: {e}")

    total_dur = _get_wav_duration_sec(wav_path) if wav_path else 0.0
    if total_dur > 0:
        row["duration_sec"] = round(total_dur, 3)

    # 2) VAD y métricas de pausas (si hay wav)
    segments: List[Dict[str, float]] = []
    if wav_path:
        try:
            segments_raw = compute_vad_segments(wav_path) or []
            segments = _sanitize_segments(segments_raw)
        except Exception as e:
            print(f"[WARN] VAD falló para {os.path.basename(video_path)}: {e}")
            segments = []

    row["speech_segments"] = len(segments)
    
    if len(segments) == 0:
        print(f"[INFO] Empty segments for {os.path.basename(video_path)}, returning default prosody metrics")

    try:
        pause_metrics = compute_pause_metrics(segments, total_dur)
        row["avg_pause_sec"] = _safe_float(pause_metrics.get("avg_pause_sec", 0.0))
        row["pause_rate_per_min"] = _safe_float(pause_metrics.get("pause_rate_per_min", 0.0))
    except Exception as e:
        # No rompas el flujo si hay algún problema con las pausas
        print(f"[WARN] Pause metrics failed en {os.path.basename(video_path)}: {e}")
        row["avg_pause_sec"] = 0.0
        row["pause_rate_per_min"] = 0.0

    # 3) Prosodia (solo si SPEECHUP_USE_PROSODY=1)
    use_prosody = os.getenv("SPEECHUP_USE_PROSODY", "0") == "1"
    if wav_path and use_prosody:
        try:
            # Usar la función wrapper que maneja la carga de audio internamente
            pros = compute_prosody_metrics_from_path(wav_path, segments)
            row["pitch_mean_hz"] = _safe_float(pros.get("pitch_mean_hz", 0.0))
            row["pitch_range_semitones"] = _safe_float(pros.get("pitch_range_semitones", 0.0))
            row["pitch_cv"] = _safe_float(pros.get("pitch_cv", 0.0))
            row["energy_cv"] = _safe_float(pros.get("energy_cv", 0.0))
            row["rhythm_consistency"] = _safe_float(pros.get("rhythm_consistency", 0.0))
        except Exception as e:
            print(f"[WARN] Prosodia falló para {os.path.basename(video_path)}: {e}")

    # 4) ASR (si SPEECHUP_USE_AUDIO=1 o SPEECHUP_USE_ASR=1)
    use_asr = (os.getenv("SPEECHUP_USE_AUDIO", "0") == "1") or (os.getenv("SPEECHUP_USE_ASR", "0") == "1")
    if wav_path and use_asr:
        try:
            asr = transcribe_wav(wav_path, lang=None) or {}
            if asr.get("ok"):
                text = asr.get("text", "") or ""
                # Preferir duración del ASR si existe
                dur_asr = asr.get("duration_sec", None)
                if isinstance(dur_asr, (int, float)) and dur_asr > 0:
                    row["duration_sec"] = round(float(dur_asr), 3)
                    dur_for_wpm = float(dur_asr)
                else:
                    dur_for_wpm = total_dur if total_dur > 0 else 60.0  # fallback defensivo

                # WPM y fillers
                row["wpm"] = _safe_float(compute_wpm(text, dur_for_wpm))
                fillers = detect_spanish_fillers(text) or {"fillers_per_min": 0.0, "filler_counts": {}}
                row["fillers_per_min"] = _safe_float(
                    normalize_fillers_per_minute(fillers.get("fillers_per_min", 0.0), dur_for_wpm)
                )

                # Confianza y transcript corto (si disponibles)
                row["stt_confidence"] = _safe_float(asr.get("stt_confidence", 0.0))
                row["transcript_short"] = (asr.get("transcript_short") or "")[:200]

                # Debug (device/model) si vuelve en "debug"
                dbg = asr.get("debug") or {}
                row["asr_device"] = str(dbg.get("device", "") or "")
                row["asr_model"] = str(dbg.get("model", "") or "")
            else:
                err = asr.get("error", "unknown")
                print(f"[WARN] ASR failed for {os.path.basename(video_path)}: {err}")
        except Exception as e:
            print(f"[WARN] ASR exception en {os.path.basename(video_path)}: {e}")

    # Limpieza del wav temporal
    try:
        if wav_path and os.path.exists(wav_path):
            os.remove(wav_path)
    except Exception:
        pass

    return row


def main() -> None:
    print(f"Scanning videos in: {TESTS_DIR}")
    print(f"Env -> SPEECHUP_USE_AUDIO={os.getenv('SPEECHUP_USE_AUDIO','0')}  SPEECHUP_USE_PROSODY={os.getenv('SPEECHUP_USE_PROSODY','0')}")

    if not os.path.isdir(TESTS_DIR):
        print(f"[ERROR] No existe la carpeta de tests: {TESTS_DIR}")
        sys.exit(1)

    vids = []
    for fn in os.listdir(TESTS_DIR):
        if fn.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpg", ".mpeg")):
            vids.append(os.path.join(TESTS_DIR, fn))

    if not vids:
        print("No se encontraron videos en /tests")
        sys.exit(0)

    rows: List[Dict[str, Any]] = []

    for vp in vids:
        print(f"Processing: {vp} ...")
        rows.append(analyze_video(vp))

    # Columnas fijas del CSV (orden amigable)
    fieldnames = [
        "video",
        "duration_sec",
        "speech_segments",
        "avg_pause_sec",
        "pause_rate_per_min",
        "wpm",
        "fillers_per_min",
        "stt_confidence",
        "transcript_short",
        "pitch_mean_hz",
        "pitch_range_semitones",
        "pitch_cv",
        "energy_cv",
        "rhythm_consistency",
        "asr_device",
        "asr_model",
    ]

    # Guardar CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\n✅ Done! Saved: {OUT_CSV}")
    print("Tip: abrí el CSV y verificá que cada estilo se distinga por varias columnas (ritmo, WPM, fillers, etc.).")


if __name__ == "__main__":
    main()
