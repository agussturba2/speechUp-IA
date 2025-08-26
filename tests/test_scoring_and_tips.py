from video.scoring import compute_scores
from video.advice_generator import AdviceGenerator

def test_scoring_and_tips_low_metrics():
    nonverbal = {
        "gaze_screen_pct": 0.2,
        "gesture_amplitude": 0.1,
        "head_stability": 0.3,
        "posture_openness": 0.5,
        "gesture_rate_per_min": 0.5,
        "expression_variability": 0.2,
        "face_coverage_pct": 0.5,
    }
    scores = compute_scores({}, {}, nonverbal)
    tips = AdviceGenerator().generate_nonverbal_tips(nonverbal)
    assert any("cámara" in tip for tip in tips)
    assert any("gestos" in tip for tip in tips)
    assert any("cabeza" in tip for tip in tips)
    assert scores["delivery_confidence"] < 60
    for k in scores:
        assert 0 <= scores[k] <= 100
