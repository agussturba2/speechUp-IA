import os
import json
from fastapi.testclient import TestClient
from api.main import app
from api.schemas.analysis_json import AnalysisJSON

client = TestClient(app)

def test_nonverbal_min_sample():
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
    from _helpers import assert_common_analysis_fields
    assert_common_analysis_fields(data)
