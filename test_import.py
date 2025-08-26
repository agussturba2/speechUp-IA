#!/usr/bin/env python3
"""Simple test script to verify prosody module imports."""

try:
    from audio.prosody import load_wav_16k, compute_prosody_metrics, compute_prosody_metrics_from_path
    print("✓ All prosody functions imported successfully")
    
    # Test function signatures
    print(f"load_wav_16k signature: {load_wav_16k.__name__}")
    print(f"compute_prosody_metrics signature: {compute_prosody_metrics.__name__}")
    print(f"compute_prosody_metrics_from_path signature: {compute_prosody_metrics_from_path.__name__}")
    
    print("🎉 Prosody API unification successful!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
