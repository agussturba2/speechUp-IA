"""
Tests for the labeling and feedback generation system.
"""

import pytest
from video.advice_generator import AdviceGenerator


def test_labels_and_tips_plana_pausas_regulares_pocos_gestos_wpm_ideal():
    """Test case: flat intonation + regular pauses + few gestures + ideal WPM"""
    proc = {
        "verbal": {
            "wpm": 140,  # ideal range (120-160)
            "fillers_per_min": 2,  # low
            "avg_pause_sec": 0.6,  # not too long
            "pause_rate_per_min": 25,  # regular
        },
        "prosody": {
            "pitch_range_semitones": 3.5,  # flat (< 4.0)
            "energy_cv": 0.15,  # medium
        },
        "nonverbal": {
            "gesture_rate_per_min": 2.0,  # low (< 3.0)
        }
    }
    
    advice_gen = AdviceGenerator()
    labels, tips = advice_gen.generate_labels_and_tips(proc)
    
    # Check labels
    assert "entonación plana" in labels
    assert "pausas regulares" in labels
    assert "pocos gestos" in labels
    assert "ritmo adecuado" in labels
    
    # Check tips
    assert len(tips) >= 1
    assert any("Variá la entonación" in tip["tip"] for tip in tips)
    assert any("Buen ritmo" in tip["tip"] for tip in tips)
    
    # Verify no contradictory labels
    assert "pausas caóticas" not in labels
    assert "ritmo lento" not in labels
    assert "ritmo acelerado" not in labels


def test_labels_and_tips_expresiva_caotica_acelerado_muchas_muletillas():
    """Test case: expressive + chaotic + fast + many fillers"""
    proc = {
        "verbal": {
            "wpm": 175,  # fast (> 170)
            "fillers_per_min": 10,  # high (>= 8)
            "avg_pause_sec": 0.4,  # not too long
            "pause_rate_per_min": 45,  # chaotic (>= 40)
        },
        "prosody": {
            "pitch_range_semitones": 9.0,  # expressive (>= 8.0)
            "energy_cv": 0.25,  # good contrast (>= 0.22)
        },
        "nonverbal": {
            "gesture_rate_per_min": 2.0,  # low
        }
    }
    
    advice_gen = AdviceGenerator()
    labels, tips = advice_gen.generate_labels_and_tips(proc)
    
    # Check labels
    assert "entonación expresiva" in labels
    assert "pausas caóticas" in labels
    assert "ritmo acelerado" in labels
    assert "muchas muletillas" in labels
    assert "buen contraste de energía" in labels
    
    # Check tips
    assert len(tips) >= 1
    assert any("Vas rápido" in tip["tip"] for tip in tips)
    assert any("Hay varias muletillas" in tip["tip"] for tip in tips)
    assert any("Pausas irregulares" in tip["tip"] for tip in tips)
    
    # Verify no contradictory labels
    assert "entonación plana" not in labels
    assert "pausas regulares" not in labels
    assert "ritmo adecuado" not in labels


def test_labels_and_tips_mixed_positive_and_improvement():
    """Test case: mixed positive and improvement opportunities"""
    proc = {
        "verbal": {
            "wpm": 130,  # ideal
            "fillers_per_min": 1,  # very low
            "avg_pause_sec": 0.3,  # good
            "pause_rate_per_min": 20,  # very regular
        },
        "prosody": {
            "pitch_range_semitones": 6.0,  # medium (not flat, not expressive)
            "energy_cv": 0.08,  # low contrast (< 0.12)
        },
        "nonverbal": {
            "gesture_rate_per_min": 7.0,  # good (>= 6.0)
        }
    }
    
    advice_gen = AdviceGenerator()
    labels, tips = advice_gen.generate_labels_and_tips(proc)
    
    # Check labels
    assert "ritmo adecuado" in labels
    assert "pocas muletillas" in labels
    assert "pausas regulares" in labels
    assert "bajo contraste de energía" in labels
    assert "gestualidad activa" in labels
    
    # Check tips
    assert len(tips) >= 1
    assert any("Buen ritmo" in tip["tip"] for tip in tips)
    assert any("¡Casi sin muletillas" in tip["tip"] for tip in tips)
    assert any("Buen uso de pausas" in tip["tip"] for tip in tips)
    assert any("Sumá contrastes de energía" in tip["tip"] for tip in tips)
    assert any("Buena gestualidad" in tip["tip"] for tip in tips)


def test_labels_and_tips_empty_data():
    """Test case: empty or missing data should not crash"""
    proc = {}
    
    advice_gen = AdviceGenerator()
    labels, tips = advice_gen.generate_labels_and_tips(proc)
    
    # Should return empty lists, not crash
    assert isinstance(labels, list)
    assert isinstance(tips, list)
    assert len(labels) == 0
    assert len(tips) == 0


def test_labels_and_tips_partial_data():
    """Test case: partial data should still generate some labels"""
    proc = {
        "verbal": {
            "wpm": 180,  # fast
        },
        # Missing prosody and nonverbal
    }
    
    advice_gen = AdviceGenerator()
    labels, tips = advice_gen.generate_labels_and_tips(proc)
    
    # Should still generate WPM-related label and tip
    assert "ritmo acelerado" in labels
    assert any("Vas rápido" in tip["tip"] for tip in tips)
    
    # Should not have prosody or gesture labels
    assert "entonación plana" not in labels
    assert "pocos gestos" not in labels


def test_labels_and_tips_maximum_limits():
    """Test case: verify maximum limits are respected"""
    # Create a case that would generate many labels
    proc = {
        "verbal": {
            "wpm": 100,  # slow
            "fillers_per_min": 12,  # high
            "avg_pause_sec": 1.2,  # long
            "pause_rate_per_min": 50,  # chaotic
        },
        "prosody": {
            "pitch_range_semitones": 2.0,  # very flat
            "energy_cv": 0.05,  # very low
        },
        "nonverbal": {
            "gesture_rate_per_min": 1.0,  # very low
        }
    }
    
    advice_gen = AdviceGenerator()
    labels, tips = advice_gen.generate_labels_and_tips(proc)
    
    # Should respect maximum limits
    assert len(labels) <= 5
    assert len(tips) <= 6


def test_labels_and_tips_spanish_prioritization():
    """Test case: verify Spanish (es-AR) text in recommendations"""
    proc = {
        "verbal": {
            "wpm": 150,  # ideal
        },
        "prosody": {
            "pitch_range_semitones": 3.0,  # flat
        },
        "nonverbal": {
            "gesture_rate_per_min": 2.0,  # low
        }
    }
    
    advice_gen = AdviceGenerator()
    labels, tips = advice_gen.generate_labels_and_tips(proc)
    
    # Check that tips are in Spanish
    for tip in tips:
        assert tip["tip"].startswith(("Tu", "Buen", "Variá", "Acompañá", "Probá", "Ensayá", "Reducí", "Sumá"))
        # These are Spanish verb forms (imperative)


if __name__ == "__main__":
    # Run tests manually if needed
    test_labels_and_tips_plana_pausas_regulares_pocos_gestos_wpm_ideal()
    test_labels_and_tips_expresiva_caotica_acelerado_muchas_muletillas()
    test_labels_and_tips_mixed_positive_and_improvement()
    test_labels_and_tips_empty_data()
    test_labels_and_tips_partial_data()
    test_labels_and_tips_maximum_limits()
    test_labels_and_tips_spanish_prioritization()
    print("All tests passed!")
