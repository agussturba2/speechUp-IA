def assert_common_analysis_fields(data: dict):
    # Debug útil: si algo falla, querés ver el payload
    def _dbg(msg):
        try:
            import json
            print(msg, json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            print(msg, data)

    # shape mínima
    for k in ["media", "scores", "nonverbal", "quality"]:
        assert k in data, _dbg(f"Missing top-level key: {k}")

    # media
    assert data["media"]["lang"] in ("es-AR", "es"), _dbg("Invalid media.lang")
    assert isinstance(data["media"]["fps"], (int, float)), _dbg("media.fps type")
    assert data["media"]["fps"] >= 0, _dbg("media.fps value")
    assert isinstance(data["media"]["duration_sec"], (int, float)), _dbg("media.duration_sec type")
    assert data["media"]["duration_sec"] > 0, _dbg("media.duration_sec value")

    # nonverbal
    nv = data["nonverbal"]
    for k in ["gaze_screen_pct", "gesture_amplitude", "face_coverage_pct"]:
        assert 0.0 <= float(nv[k]) <= 1.0, _dbg(f"nonverbal.{k} out of [0,1]")

    assert isinstance(nv["gesture_rate_per_min"], (int, float)), _dbg("nonverbal.gesture_rate_per_min type")
    assert nv["gesture_rate_per_min"] >= 0.0, _dbg("nonverbal.gesture_rate_per_min value")

    # quality
    q = data["quality"]
    assert isinstance(q.get("analysis_ms"), (int, float)), _dbg("quality.analysis_ms type/missing")
    assert q["analysis_ms"] >= 0, _dbg("quality.analysis_ms value")
    assert isinstance(q.get("audio_available"), bool), _dbg("quality.audio_available type/missing")
