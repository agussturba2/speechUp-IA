import os
import json
import pytest
from fastapi.testclient import TestClient
from api.main import app
from api.schemas.analysis_json import AnalysisJSON

client = TestClient(app)

def test_nonverbal_static_sample():
    """Test nonverbal metrics on static face-only sample video."""
    sample_path = os.path.join(os.path.dirname(__file__), "sample.mp4")
    
    with open(sample_path, "rb") as f:
        response = client.post(
            "/v1/feedback-oratoria",
            files={"video_file": ("sample.mp4", f, "video/mp4")},
        )
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate schema
    AnalysisJSON.model_validate(data)
    
    # Validate common fields
    from _helpers import assert_common_analysis_fields
    assert_common_analysis_fields(data)
    
    # Test nonverbal metrics
    nonverbal = data.get("nonverbal", {})
    
    # Assert expected values for static face-only video
    assert "posture_openness" in nonverbal
    assert "expression_variability" in nonverbal
    assert "gaze_screen_pct" in nonverbal
    assert "gesture_rate_per_min" in nonverbal
    assert "gesture_amplitude" in nonverbal
    assert "engagement" in nonverbal
    
    # For a static face-only video, we expect:
    # - head_stability should be high (~1.0) since minimal head motion
    # - gesture_rate_per_min should be 0 since no hand motion
    # - posture_openness should be > 0 since pose is detected
    # - posture_openness should NEVER be 1.0 (clamped to 0.95)
    # - engagement should be 0 when no gestures
    assert nonverbal["head_stability"] > 0.5, f"Expected high head stability, got {nonverbal['head_stability']}"
    assert nonverbal["gesture_rate_per_min"] == 0, f"Expected no gestures for static video, got {nonverbal['gesture_rate_per_min']}"
    assert nonverbal["posture_openness"] > 0, f"Expected positive posture openness, got {nonverbal['posture_openness']}"
    assert nonverbal["posture_openness"] < 1.0, f"Expected posture_openness < 1.0 (clamped), got {nonverbal['posture_openness']}"
    assert nonverbal["engagement"] == 0, f"Expected engagement = 0 for static video, got {nonverbal['engagement']}"
    
    # Test scores are integers
    scores = data.get("scores", {})
    assert "delivery_confidence" in scores
    assert "engagement" in scores
    
    # Ensure scores are integers
    assert isinstance(scores["delivery_confidence"], int), f"Expected int, got {type(scores['delivery_confidence'])}"
    assert isinstance(scores["engagement"], int), f"Expected int, got {type(scores['engagement'])}"
    
    # Test score ranges
    assert 0 <= scores["delivery_confidence"] <= 100, f"Score out of range: {scores['delivery_confidence']}"
    assert 0 <= scores["engagement"] <= 100, f"Score out of range: {scores['engagement']}"
    
    # Test nonverbal metric ranges
    for key in ["posture_openness", "expression_variability", "gaze_screen_pct", "head_stability", "gesture_amplitude", "engagement"]:
        if key in nonverbal:
            value = nonverbal[key]
            assert 0.0 <= value <= 1.0, f"{key} out of [0,1] range: {value}"
    
    # Test gesture metrics
    assert nonverbal["gesture_rate_per_min"] >= 0, f"Gesture rate should be non-negative: {nonverbal['gesture_rate_per_min']}"
    assert nonverbal["gesture_amplitude"] >= 0, f"Gesture amplitude should be non-negative: {nonverbal['gesture_amplitude']}"
    
    # Test events (should be empty for static video)
    events = data.get("events", [])
    assert isinstance(events, list), f"Events should be a list, got {type(events)}"
    assert len(events) == 0, f"Expected no events for static video, got {len(events)}"
    
    print(f"✅ Nonverbal metrics test passed:")
    print(f"   - posture_openness: {nonverbal.get('posture_openness', 'N/A')}")
    print(f"   - expression_variability: {nonverbal.get('expression_variability', 'N/A')}")
    print(f"   - gaze_screen_pct: {nonverbal.get('gaze_screen_pct', 'N/A')}")
    print(f"   - head_stability: {nonverbal.get('head_stability', 'N/A')}")
    print(f"   - gesture_rate_per_min: {nonverbal.get('gesture_rate_per_min', 'N/A')}")
    print(f"   - gesture_amplitude: {nonverbal.get('gesture_amplitude', 'N/A')}")
    print(f"   - engagement: {nonverbal.get('engagement', 'N/A')}")
    print(f"   - delivery_confidence: {scores.get('delivery_confidence', 'N/A')}")
    print(f"   - engagement score: {scores.get('engagement', 'N/A')}")
    print(f"   - events count: {len(events)}")

def test_nonverbal_metrics_structure():
    """Test that nonverbal metrics have the expected structure."""
    sample_path = os.path.join(os.path.dirname(__file__), "sample.mp4")
    
    with open(sample_path, "rb") as f:
        response = client.post(
            "/v1/feedback-oratoria",
            files={"video_file": ("sample.mp4", f, "video/mp4")},
        )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check nonverbal section exists
    assert "nonverbal" in data, "Response missing 'nonverbal' section"
    
    nonverbal = data["nonverbal"]
    expected_keys = [
        "face_coverage_pct",
        "gaze_screen_pct", 
        "head_stability",
        "posture_openness",
        "gesture_rate_per_min",
        "gesture_amplitude",
        "expression_variability",
        "engagement"
    ]
    
    for key in expected_keys:
        assert key in nonverbal, f"Missing nonverbal metric: {key}"
        assert nonverbal[key] is not None, f"Nonverbal metric {key} is None"
    
    # Check that all nonverbal values are numeric
    for key, value in nonverbal.items():
        assert isinstance(value, (int, float)), f"Nonverbal metric {key} is not numeric: {type(value)}"
    
    print(f"✅ Nonverbal metrics structure test passed - all {len(expected_keys)} metrics present")

def test_gesture_amplitude_normalization():
    """Test that gesture_amplitude is properly normalized to [0,1] range."""
    sample_path = os.path.join(os.path.dirname(__file__), "sample.mp4")
    
    with open(sample_path, "rb") as f:
        response = client.post(
            "/v1/feedback-oratoria",
            files={"video_file": ("sample.mp4", f, "video/mp4")},
        )
    
    assert response.status_code == 200
    data = response.json()
    
    nonverbal = data.get("nonverbal", {})
    gesture_amplitude = nonverbal.get("gesture_amplitude", 0.0)
    
    # gesture_amplitude should be in [0,1] range
    assert 0.0 <= gesture_amplitude <= 1.0, f"gesture_amplitude out of [0,1] range: {gesture_amplitude}"
    
    print(f"✅ Gesture amplitude normalization test passed: {gesture_amplitude}")

def test_posture_openness_clamping():
    """Test that posture_openness is clamped to maximum 0.95."""
    sample_path = os.path.join(os.path.dirname(__file__), "sample.mp4")
    
    with open(sample_path, "rb") as f:
        response = client.post(
            "/v1/feedback-oratoria",
            files={"video_file": ("sample.mp4", f, "video/mp4")},
        )
    
    assert response.status_code == 200
    data = response.json()
    
    nonverbal = data.get("nonverbal", {})
    posture_openness = nonverbal.get("posture_openness", 0.0)
    
    # posture_openness should never be 1.0 (clamped to 0.95)
    assert posture_openness < 1.0, f"posture_openness should be < 1.0 (clamped), got {posture_openness}"
    assert 0.0 <= posture_openness <= 0.95, f"posture_openness out of [0, 0.95] range: {posture_openness}"
    
    print(f"✅ Posture openness clamping test passed: {posture_openness}") 