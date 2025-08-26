import os
import tempfile
import requests
from fastapi.testclient import TestClient
from api.main import app
from api.schemas.analysis_json import AnalysisJSON

client = TestClient(app)

def get_sample_video_path():
    # Use a tiny sample video (should exist in repo or generate)
    return os.path.join(os.path.dirname(__file__), "sample.mp4")

def test_feedback_oratoria_file():
    video_path = get_sample_video_path()
    with open(video_path, "rb") as f:
        response = client.post(
            "/v1/feedback-oratoria",
            files={"video_file": ("sample.mp4", f, "video/mp4")},
        )
    assert response.status_code == 200
    data = response.json()
    AnalysisJSON.model_validate(data)
    from _helpers import assert_common_analysis_fields
    assert_common_analysis_fields(data)

def test_feedback_oratoria_url():
    video_path = get_sample_video_path()
    # Serve file via local HTTP (simulate static server)
    import http.server
    import threading
    import socketserver
    PORT = 8001
    Handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", PORT), Handler)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.daemon = True
    thread.start()
    try:
        url = f"http://localhost:{PORT}/tests/sample.mp4"
        response = client.post(
            "/v1/feedback-oratoria",
            data={"media_url": url},
        )
        assert response.status_code == 200
        data = response.json()
        AnalysisJSON.model_validate(data)
        from _helpers import assert_common_analysis_fields
        assert_common_analysis_fields(data)
    finally:
        httpd.shutdown()
        thread.join()
