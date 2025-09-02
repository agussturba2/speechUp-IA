#!/usr/bin/env python3
"""
Test prosody integration with the pipeline and metrics system.
"""

import os
import sys
import tempfile
import json
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video.metrics import build_metrics_response
from video.advice_generator import AdviceGenerator


def test_prosody_metrics_integration():
    """Test that prosody metrics are properly integrated."""
    
    # Mock prosody data - use values that will trigger tips
    prosody_data = {
        "pitch_mean_hz": 150.0,
        "pitch_range_semitones": 1.5,  # Low pitch variation (< 2.0)
        "pitch_cv": 0.3,
        "energy_cv": 0.05,  # Low energy variation (< 0.1)
        "rhythm_consistency": 0.3  # Low rhythm consistency (< 0.4)
    }
    
    # Test advice generator with prosody
    advice_gen = AdviceGenerator()
    prosody_tips = advice_gen.generate_prosody_tips(prosody_data)
    
    print(f"Prosody tips generated: {prosody_tips}")
    assert len(prosody_tips) > 0, "Should generate prosody tips"
    
    # Test metrics response with prosody
    media = {
        "duration_sec": 30.0,
        "lang": "es-AR",
        "fps": 30,
        "frames_total": 900,
        "frames_with_face": 800,
        "gesture_events": 5,
        "dropped_frames_pct": 0.0,
        "gaze_screen_pct": 0.8,
        "head_stability": 0.7,
        "posture_openness": 0.6,
        "gesture_amplitude": 0.5,
        "expression_variability": 0.6,
        "engagement": 0.7
    }
    
    response = build_metrics_response(
        feedbacks=[],
        media=media,
        gesture_events=5,
        events=[],
        analysis_ms=1000,
        verbal={},
        prosody=prosody_data,
        audio_available=True
    )
    
    print(f"Response keys: {list(response.keys())}")
    print(f"Prosody in response: {response.get('prosody', {})}")
    
    # Verify prosody is in the response
    assert "prosody" in response, "Prosody should be in response"
    assert response["prosody"]["pitch_mean_hz"] == 150.0, "Pitch mean should match"
    assert response["prosody"]["rhythm_consistency"] == 0.3, "Rhythm consistency should match"
    
    print("✅ Prosody integration test passed!")


def test_prosody_defaults():
    """Test that prosody defaults work when no data is provided."""
    
    media = {
        "duration_sec": 30.0,
        "lang": "es-AR",
        "fps": 30,
        "frames_total": 900,
        "frames_with_face": 800,
        "gesture_events": 5,
        "dropped_frames_pct": 0.0,
        "gaze_screen_pct": 0.8,
        "head_stability": 0.7,
        "posture_openness": 0.6,
        "gesture_amplitude": 0.5,
        "expression_variability": 0.6,
        "engagement": 0.7
    }
    
    response = build_metrics_response(
        feedbacks=[],
        media=media,
        gesture_events=5,
        events=[],
        analysis_ms=1000,
        verbal={},
        prosody=None,  # No prosody data
        audio_available=False
    )
    
    print(f"Default prosody: {response.get('prosody', {})}")
    
    # Verify default prosody structure
    assert "prosody" in response, "Prosody should be present even with defaults"
    assert response["prosody"]["pitch_mean_hz"] == 0.0, "Default pitch should be 0"
    assert response["prosody"]["rhythm_consistency"] == 0.0, "Default rhythm should be 0"
    
    print("✅ Prosody defaults test passed!")


def test_prosody_tips_generation():
    """Test prosody-based tip generation."""
    
    advice_gen = AdviceGenerator()
    
    # Test with low pitch variation
    low_pitch_prosody = {
        "pitch_mean_hz": 120.0,
        "pitch_range_semitones": 1.0,  # Low variation
        "pitch_cv": 0.1,
        "energy_cv": 0.3,
        "rhythm_consistency": 0.6
    }
    
    tips = advice_gen.generate_prosody_tips(low_pitch_prosody)
    print(f"Low pitch variation tips: {tips}")
    assert any("tono" in tip.lower() for tip in tips), "Should suggest pitch variation"
    
    # Test with low energy dynamics
    low_energy_prosody = {
        "pitch_mean_hz": 150.0,
        "pitch_range_semitones": 4.0,
        "pitch_cv": 0.4,
        "energy_cv": 0.05,  # Low energy variation
        "rhythm_consistency": 0.7
    }
    
    tips = advice_gen.generate_prosody_tips(low_energy_prosody)
    print(f"Low energy variation tips: {tips}")
    assert any("volumen" in tip.lower() for tip in tips), "Should suggest energy variation"
    
    print("✅ Prosody tips generation test passed!")


if __name__ == "__main__":
    print("Running prosody integration tests...")
    
    test_prosody_metrics_integration()
    test_prosody_defaults()
    test_prosody_tips_generation()
    
    print("\n🎉 All prosody integration tests passed!")
