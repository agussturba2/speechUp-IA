# tests/collect_metrics.py
import os
import csv
import glob
import traceback

from video.audio_utils import extract_wav_mono_16k, compute_vad_segments
from audio.prosody import compute_prosody_metrics

OUT_CSV = os.path.join("tests", "prosody_metrics.csv")

def compute_for_video(video_path: str) -> dict:
    """Extrae audio, corre VAD y calcula métricas de prosodia para un video."""
    wav_path = None
    try:
        wav_path = extract_wav_mono_16k(video_path)
        if not wav_path or not os.path.exists(wav_path):
            raise RuntimeError("No se pudo extraer audio (WAV).")

        segments = compute_vad_segments(wav_path)
        # Tolerante a firmas distintas de compute_prosody_metrics
        metrics = None
        try:
            metrics = compute_prosody_metrics(wav_path, segments)  # firma A (wav, segments)
        except TypeError:
            # Algunos PRs usan duración total como tercer parámetro
            import wave
            with wave.open(wav_path, "rb") as wf:
                frames = wf.getnframes()
                sr = wf.getframerate()
                duration_sec = frames / float(sr)
            metrics = compute_prosody_metrics(wav_path, segments, duration_sec)  # firma B (wav, segments, dur)

        if not isinstance(metrics, dict):
            raise RuntimeError("compute_prosody_metrics no devolvió un dict.")

        return {
            "video": os.path.basename(video_path),
            "pitch_mean_hz": float(metrics.get("pitch_mean_hz", 0.0)),
            "pitch_range_semitones": float(metrics.get("pitch_range_semitones", 0.0)),
            "pitch_cv": float(metrics.get("pitch_cv", 0.0)),
            "energy_cv": float(metrics.get("energy_cv", 0.0)),
            "rhythm_consistency": float(metrics.get("rhythm_consistency", 0.0)),
        }
    finally:
        # Limpieza del WAV temporal
        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

def main():
    tests_dir = "tests"
    videos = sorted(glob.glob(os.path.join(tests_dir, "*.mp4")))
    print(f"Encontrados {len(videos)} videos")

    rows = []
    for vp in videos:
        print(f"Procesando {vp}...")
        try:
            row = compute_for_video(vp)
            rows.append(row)
        except Exception as e:
            print(f"[WARN] No se pudieron calcular métricas para {vp}: {e}")
            # Descomenta para debug detallado:
            # traceback.print_exc()

    # Crear carpeta tests si no existe
    os.makedirs(tests_dir, exist_ok=True)

    # Escribir CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "video",
                "pitch_mean_hz",
                "pitch_range_semitones",
                "pitch_cv",
                "energy_cv",
                "rhythm_consistency",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n✅ CSV generado: {OUT_CSV}")
    if rows:
        print(f"   Filas: {len(rows)}")
    else:
        print("   (Sin filas; revisá los avisos arriba)")

if __name__ == "__main__":
    # Recomendado: asegurate que estos flags estén activos en la sesión
    # SPEECHUP_USE_AUDIO=1 y SPEECHUP_USE_PROSODY=1
    main()
