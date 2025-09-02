"""
Test ASR integration with the video pipeline.

This test validates that ASR functionality works correctly and
doesn't break the existing API schema.
"""

import os
import pytest
import tempfile
import wave
import numpy as np
from unittest.mock import patch, MagicMock

# Test the ASR module
def test_asr_module_imports():
    """Test that ASR modules can be imported correctly."""
    try:
        from audio.asr import transcribe_wav, is_asr_enabled
        from audio.text_metrics import compute_wpm, detect_spanish_fillers
        assert True
    except ImportError as e:
        pytest.skip(f"ASR modules not available: {e}")

def test_asr_environment_check():
    """Test ASR environment variable checking."""
    from audio.asr import is_asr_enabled
    
    # Test default behavior
    if "SPEECHUP_USE_ASR" in os.environ:
        del os.environ["SPEECHUP_USE_ASR"]
    
    assert not is_asr_enabled()
    
    # Test enabled
    os.environ["SPEECHUP_USE_ASR"] = "1"
    assert is_asr_enabled()
    
    # Test disabled
    os.environ["SPEECHUP_USE_ASR"] = "0"
    assert not is_asr_enabled()

def test_text_metrics_functions():
    """Test text metrics computation functions."""
    from audio.text_metrics import compute_wpm, detect_spanish_fillers, normalize_fillers_per_minute
    
    # Test WPM calculation
    transcript = "hola mundo como estas"
    wpm = compute_wpm(transcript, 10.0)  # 4 words in 10 seconds = 24 WPM
    assert wpm == 24.0
    
    # Test WPM with empty transcript
    wpm_empty = compute_wpm("", 10.0)
    assert wpm_empty == 0.0
    
    # Test filler detection
    transcript_with_fillers = "este es un ejemplo o sea de como usar este tipo de palabras"
    fillers = detect_spanish_fillers(transcript_with_fillers)
    
    assert "fillers_per_min" in fillers
    assert "filler_counts" in fillers
    assert isinstance(fillers["filler_counts"], dict)
    assert fillers["filler_counts"].get("este", 0) > 0
    assert fillers["filler_counts"].get("o sea", 0) > 0
    
    # Test normalization
    normalized = normalize_fillers_per_minute(5.0, 30.0)  # 5 fillers in 30 seconds
    assert normalized == 10.0  # 5 * (60/30) = 10 per minute

def test_asr_transcription_mock():
    """Test ASR transcription with mocked Whisper."""
    from audio.asr import transcribe_wav
    
    # Create a temporary WAV file for testing
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        wav_path = tmp_file.name
        
        # Create a simple WAV file
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(16000)  # 16kHz
            # Generate 1 second of silence
            frames = np.zeros(16000, dtype=np.int16)
            wav_file.writeframes(frames.tobytes())
        
        try:
            # Mock Whisper to avoid actual model loading
            with patch('audio.asr.whisper') as mock_whisper:
                mock_model = MagicMock()
                mock_model.transcribe.return_value = {
                    "text": "hola mundo",
                    "segments": [{"start": 0.0, "end": 1.0, "text": "hola mundo"}]
                }
                mock_whisper.load_model.return_value = mock_model
                
                # Test transcription
                result = transcribe_wav(wav_path, lang="es")
                
                assert "text" in result
                assert "segments" in result
                assert "duration_sec" in result
                assert result["text"] == "hola mundo"
                assert len(result["segments"]) == 1
                
        finally:
            # Cleanup
            if os.path.exists(wav_path):
                os.unlink(wav_path)

def test_asr_fallback_on_error():
    """Test that ASR gracefully handles errors."""
    from audio.asr import transcribe_wav
    
    # Test with non-existent file
    result = transcribe_wav("/non/existent/file.wav", lang="es")
    
    assert result["text"] == ""
    assert result["segments"] == []
    assert result["duration_sec"] == 0.0

import os
import tempfile
import wave
import numpy as np
import pytest

def test_compute_wpm_guards():
    from audio.text_metrics import compute_wpm
    assert compute_wpm("hola mundo", 2.0) == 0.0  # too short
    assert compute_wpm("", 10.0) == 0.0
    assert compute_wpm("uno dos tres cuatro cinco", 10.0) == 30.0
    assert compute_wpm("uno dos tres", 3.0) == 60.0

def test_smoke_transcribe_and_wpm():
    from audio.asr import transcribe_wav
    from audio.text_metrics import compute_wpm
    # Create a short WAV file (5s silence)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        wav_path = tmp_file.name
        with wave.open(wav_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            frames = np.zeros(16000*5, dtype=np.int16)
            wav_file.writeframes(frames.tobytes())
    try:
        result = transcribe_wav(wav_path, lang="es")
        assert "ok" in result
        assert result["ok"] in (True, False)
        wpm = compute_wpm(result["text"], result["duration_sec"])
        assert wpm == 0.0  # silence
    finally:
        os.unlink(wav_path)

if __name__ == "__main__":
    # Run basic tests if executed directly
    print("Testing ASR integration...")
    
    try:
        test_asr_module_imports()
        print("✓ ASR modules import correctly")
        
        test_asr_environment_check()
        print("✓ ASR environment checking works")
        
        test_text_metrics_functions()
        print("✓ Text metrics functions work")
        
        print("\nAll basic tests passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
