
from fastapi.testclient import TestClient
from api.main import app

SAMPLE = {
    "Store": 1,
    "Holiday_Flag": 0,
    "Temperature": 42.31,
    "Fuel_Price": 2.572,
    "CPI": 211.096358,
    "Unemployment": 8.106,
    "Year": 2010,
    "Month": 2,
    "Week": 5,
    "Quarter": 1,
}


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200


def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True


def test_metadata_endpoint():
    with TestClient(app) as client:
        response = client.get("/metadata")
        assert response.status_code == 200
        assert response.json()["target"] == "Weekly_Sales"


def test_single_prediction():
    with TestClient(app) as client:
        response = client.post("/predict", json=SAMPLE)
        assert response.status_code == 200
        result = response.json()
        assert isinstance(result["prediction"], (int, float))
        assert result["latency_ms"] >= 0


def test_batch_prediction():
    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={"records": [SAMPLE, SAMPLE]},
        )
        assert response.status_code == 200
        result = response.json()
        assert result["count"] == 2
        assert len(result["predictions"]) == 2


def test_invalid_input():
    with TestClient(app) as client:
        response = client.post("/predict", json={"Store": 1})
        assert response.status_code == 422
