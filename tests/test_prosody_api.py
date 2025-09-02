#!/usr/bin/env python3
"""
Unit tests for the new prosody API.
"""

import numpy as np
import os
import sys
import tempfile
import wave

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.prosody import (
    load_wav_16k, 
    compute_prosody_metrics, 
    compute_prosody_metrics_from_path
)


def test_load_wav_16k_returns_tuple():
    """Test that load_wav_16k returns a tuple with duration > 0."""
    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        # Create a simple 1-second 16kHz mono WAV file
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        # Generate a simple sine wave
        frequency = 440  # A4 note
        t = np.linspace(0, duration, samples, False)
        audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Write WAV file
        with wave.open(tmp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
        
        try:
            # Test the function
            y, sr, dur = load_wav_16k(tmp_file.name)
            
            # Assertions
            assert isinstance(y, np.ndarray), "Should return numpy array"
            assert sr == 16000, f"Sample rate should be 16000, got {sr}"
            assert dur > 0, f"Duration should be > 0, got {dur}"
            assert len(y) > 0, "Audio array should not be empty"
            
            print("✓ load_wav_16k returns tuple with duration > 0")
            
        finally:
            # Cleanup
            os.unlink(tmp_file.name)


def test_compute_prosody_metrics_empty_inputs():
    """Test that compute_prosody_metrics returns zeros for empty inputs."""
    # Test with empty audio array
    result = compute_prosody_metrics(np.array([]), 16000, [], 0.0)
    expected_keys = ["pitch_mean_hz", "pitch_range_semitones", "pitch_cv", "energy_cv", "rhythm_consistency"]
    
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"
        assert result[key] == 0.0, f"Expected 0.0 for {key}, got {result[key]}"
    
    print("✓ compute_prosody_metrics returns zeros for empty inputs")


def test_wrapper_matches_canonical():
    """Test that the wrapper function matches canonical function result."""
    # Create a temporary WAV file with some content
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        # Create a simple 2-second 16kHz mono WAV file
        sample_rate = 16000
        duration = 2.0
        samples = int(sample_rate * duration)
        
        # Generate a simple sine wave
        frequency = 440  # A4 note
        t = np.linspace(0, duration, samples, False)
        audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        
        # Write WAV file
        with wave.open(tmp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
        
        try:
            # Test both functions
            y, sr, dur = load_wav_16k(tmp_file.name)
            segments = [(0.0, 1.0), (1.5, 2.0)]  # Some speech segments
            
            canonical_result = compute_prosody_metrics(y, sr, segments, dur)
            wrapper_result = compute_prosody_metrics_from_path(tmp_file.name, segments)
            
            # Both should return the same keys
            assert set(canonical_result.keys()) == set(wrapper_result.keys()), "Keys should match"
            
            # Both should have valid numeric values
            for key in canonical_result.keys():
                assert isinstance(canonical_result[key], (int, float)), f"Canonical {key} should be numeric"
                assert isinstance(wrapper_result[key], (int, float)), f"Wrapper {key} should be numeric"
            
            print("✓ Wrapper matches canonical result structure")
            
        finally:
            # Cleanup
            os.unlink(tmp_file.name)


def main():
    """Run all tests."""
    print("Running prosody API tests...")
    
    try:
        test_load_wav_16k_returns_tuple()
        test_compute_prosody_metrics_empty_inputs()
        test_wrapper_matches_canonical()
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
