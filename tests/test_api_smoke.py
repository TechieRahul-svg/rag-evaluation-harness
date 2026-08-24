from fastapi.testclient import TestClient

from src.app import main


def test_app_import_does_not_initialize_pipeline():
    assert main.pipeline is None
    assert main.evaluator is None


def test_results_endpoint_returns_json():
    client = TestClient(main.app)

    response = client.get("/api/results")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
