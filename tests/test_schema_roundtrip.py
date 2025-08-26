import json
from api.schemas.analysis_json import AnalysisJSON

def test_analysisjson_roundtrip():
    with open('fixtures/analysis_v1_example.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Validate from dict
    model = AnalysisJSON.model_validate(data)
    # Dump to dict
    dumped = model.model_dump()
    # Validate again
    model2 = AnalysisJSON.model_validate(dumped)
    assert model2 == model
