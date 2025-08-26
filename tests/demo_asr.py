# tests/demo_asr.py
import sys
import os
from video.audio_utils import extract_wav_mono_16k
from audio.asr import transcribe_wav  # debe devolver stt_confidence y segment stats en 'debug'
from audio.text_metrics import compute_wpm, detect_spanish_fillers, normalize_fillers_per_minute

def _fmt(x, nd=3):
    """Helper to format possibly-None numeric values without crashing."""
    return f"{x:.{nd}f}" if isinstance(x, (int, float)) else "N/A"

def main():
    # -------- Config visible --------
    model_env = os.getenv("SPEECHUP_ASR_MODEL", "(auto)")
    device_env = os.getenv("WHISPER_DEVICE", "(auto)")
    print(f"[CFG] SPEECHUP_ASR_MODEL={model_env}  WHISPER_DEVICE={device_env}")

    # -------- Input --------
    video_path = sys.argv[1] if len(sys.argv) > 1 else "tests/sample2.mp4"
    if not os.path.exists(video_path):
        print(f"[ERROR] File not found: {video_path}")
        sys.exit(1)
    print(f"[INFO] Using video: {video_path}")

    # -------- Extract audio --------
    wav_path = extract_wav_mono_16k(video_path)
    if not wav_path or not os.path.exists(wav_path):
        print("[ERROR] Failed to extract audio.")
        sys.exit(1)

    try:
        # -------- ASR --------
        # lang=None → auto detección; usa "es" si querés forzar español
        asr_result = transcribe_wav(wav_path, lang=None)

        dbg = asr_result.get("debug", {})
        print("\n[DEBUG] segments:", dbg.get("num_segments"))
        print("[DEBUG] avg_logprob_mean:", dbg.get("avg_logprob_mean"))
        print("[DEBUG] no_speech_prob_mean:", dbg.get("no_speech_prob_mean"))
        print("[DEBUG] device/model:", dbg.get("device"), dbg.get("model"))

        if not asr_result.get("ok"):
            print(f"[ERROR] ASR failed: {asr_result.get('error')}")
            sys.exit(1)

        # Campos esperados del nuevo ASR:
        # text, transcript_short, duration_sec, stt_confidence, debug:{avg_logprob_mean, no_speech_prob_min/max/mean, device, model}
        text = asr_result.get("text", "") or ""
        duration_sec = float(asr_result.get("duration_sec", 0.0) or 0.0)
        stt_conf = float(asr_result.get("stt_confidence", 0.0) or 0.0)

        dbg = asr_result.get("debug", {}) or {}
        avg_logprob_mean = dbg.get("avg_logprob_mean", None)
        nsp_min = dbg.get("no_speech_prob_min", None)
        nsp_max = dbg.get("no_speech_prob_max", None)
        used_device = dbg.get("device", device_env)
        used_model = dbg.get("model", model_env)

        # -------- Metrics --------
        wpm = compute_wpm(text, duration_sec)
        fillers = detect_spanish_fillers(text)
        fillers_per_min = normalize_fillers_per_minute(fillers.get("fillers_per_min", 0.0), duration_sec)

        # -------- Output --------
        print()
        print(f"[ASR] device={used_device}  model={used_model}")
        print(f'Transcript (first 120 chars): "{text[:120]}"')
        print()
        print(f"WPM: {wpm:.1f}")
        print(f"Fillers per minute: {fillers_per_min:.2f}")
        print(f"Filler counts: {fillers.get('filler_counts', {})}")
        print(f"STT confidence (real): {stt_conf:.2f}")

        # Componentes de la confianza (para ver por qué sube/baja)
        if avg_logprob_mean is not None:
            print(f"  avg_logprob_mean: {avg_logprob_mean:.3f}")
        if nsp_min is not None and nsp_max is not None:
            print(f"  no_speech_prob min/max: {nsp_min:.3f} / {nsp_max:.3f}")
        
        # Comprehensive debug output if available
        if "debug" in asr_result and asr_result["debug"]:
            dbg = asr_result["debug"]
            print("\n[DEBUG] ASR Confidence Analysis:")
            print(f"  segments: {dbg.get('num_segments', 'N/A')}")
            print(f"  avg_logprob_median: {_fmt(dbg.get('avg_logprob_median'))}")
            print(f"  avg_logprob_mean: {_fmt(dbg.get('avg_logprob_mean'))}")
            print(f"  no_speech_prob_median: {_fmt(dbg.get('no_speech_prob_median'))}")
            print(f"  no_speech_prob_mean: {_fmt(dbg.get('no_speech_prob_mean'))}")
            print(f"  no_speech_prob_min: {_fmt(dbg.get('no_speech_prob_min'))}")
            print(f"  no_speech_prob_max: {_fmt(dbg.get('no_speech_prob_max'))}")
            print(f"  lp_norm: {_fmt(dbg.get('lp_norm'))}")
            print(f"  conf_final: {_fmt(dbg.get('conf_final'))}")
            print(f"  used_topk: {dbg.get('used_topk', 'N/A')}")
            print(f"  k: {dbg.get('k', 'N/A')}")
            print(f"  device: {dbg.get('device', 'N/A')}")
            print(f"  model: {dbg.get('model', 'N/A')}")

    finally:
        try:
            if wav_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass

if __name__ == "__main__":
    main()
