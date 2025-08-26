#!/usr/bin/env python3
"""
Demo script for prosody analysis.

Usage:
    python -m tests.demo_prosody [video_path]
    
Example:
    python -m tests.demo_prosody tests/sample2.mp4
"""

import os
import sys
import json
from video.audio_utils import extract_wav_mono_16k, compute_vad_segments
from audio.prosody import compute_prosody_metrics_from_path


def main():
    """Main function to demonstrate prosody analysis."""
    video = sys.argv[1] if len(sys.argv) > 1 else "tests/sample2.mp4"
    
    # Set environment variables
    os.environ["SPEECHUP_USE_AUDIO"] = "1"
    os.environ["SPEECHUP_USE_PROSODY"] = "1"
    
    print(f"Processing video: {video}")
    print("Environment: SPEECHUP_USE_AUDIO=1, SPEECHUP_USE_PROSODY=1")
    
    # Check prosody prefilter setting
    prosody_prefilter = os.getenv("SPEECHUP_PROSODY_PREFILTER", "1")
    print(f"SPEECHUP_PROSODY_PREFILTER={prosody_prefilter}")
    
    try:
        # Extract audio
        print("Extracting audio...")
        wav = extract_wav_mono_16k(video)
        
        if not wav or not os.path.exists(wav):
            print("Failed to extract audio")
            return
        
        print(f"Audio extracted to: {wav}")
        
        # Compute VAD segments
        print("Computing VAD segments...")
        segs = compute_vad_segments(wav)
        print(f"Found {len(segs)} speech segments")
        
        # Get duration
        dur = 0.0
        try:
            import soundfile as sf
            import librosa
            y, sr = librosa.load(wav, sr=16000, mono=True)
            dur = len(y) / sr
            print(f"Audio duration: {dur:.2f}s")
        except Exception as e:
            print(f"Could not determine duration: {e}")
        
        # Compute prosody metrics using the wrapper
        print("Computing prosody metrics...")
        pros = compute_prosody_metrics_from_path(wav, segs)
        
        print("\nProsody Metrics:")
        print(json.dumps(pros, indent=2))
        
        # Cleanup
        try:
            os.remove(wav)
            print(f"Cleaned up temporary file: {wav}")
        except Exception as e:
            print(f"Warning: Could not remove {wav}: {e}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
