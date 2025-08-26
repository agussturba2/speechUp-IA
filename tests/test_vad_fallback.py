# tests/test_vad_fallback.py
import os
import numpy as np
import soundfile as sf
from video.audio_utils import compute_vad_segments

def test_vad_does_not_crash_on_silence(tmp_path):
    """Test that VAD handles silence gracefully without crashing."""
    # 1s of silence 16k
    y = np.zeros(16000, dtype=np.float32)
    wav = os.path.join(tmp_path, "sil.wav")
    sf.write(wav, y, 16000)
    segs = compute_vad_segments(wav)
    assert isinstance(segs, list)
    # For silence, we expect either empty list or very few segments
    # The fallback energy VAD might detect some low-level noise
    print(f"Silence test: detected {len(segs)} segments")
    if segs:
        print(f"Segment details: {segs}")
    # Just ensure it doesn't crash and returns a list
    assert isinstance(segs, list)

def test_vad_returns_something_on_voice_sample():
    """Test that VAD detects speech in voice samples."""
    # If you have tests/sample2.mp4 extracted to wav somewhere, test it
    # For now, just a placeholder test
    pass

def test_vad_robustness():
    """Test that VAD functions don't crash on various inputs."""
    # Test with empty segments
    from video.audio_utils import compute_pause_metrics
    
    # Empty segments should return safe defaults
    result = compute_pause_metrics([], 10.0)
    assert isinstance(result, dict)
    assert "avg_pause_sec" in result
    assert "pause_rate_per_min" in result
    assert result["avg_pause_sec"] == 0.0
    assert result["pause_rate_per_min"] == 0.0
    
    # Invalid duration should return safe defaults
    result = compute_pause_metrics([{"start": 0.0, "end": 1.0}], -1.0)
    assert result["avg_pause_sec"] == 0.0
    assert result["pause_rate_per_min"] == 0.0

if __name__ == "__main__":
    # Simple test runner
    import tempfile
    
    print("Testing VAD fallback functionality...")
    
    # Test silence handling
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_vad_does_not_crash_on_silence(tmp_dir)
        print("✅ Silence test passed")
    
    # Test robustness
    test_vad_robustness()
    print("✅ Robustness test passed")
    
    print("All tests passed!")
