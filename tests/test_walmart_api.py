import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Walmart ML API is running" in response.json()["message"]

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True

def test_prediction(client):
    payload = {
        "Store": 1,
        "Date": "05-02-2010",
        "Holiday_Flag": 0,
        "Temperature": 42.31,
        "Fuel_Price": 2.572,
        "CPI": 211.096358,
        "Unemployment": 8.106
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_weekly_sales" in response.json()

def test_invalid_date(client):
    payload = {
        "Store": 1,
        "Date": "invalid-date",
        "Holiday_Flag": 0,
        "Temperature": 42.31,
        "Fuel_Price": 2.572,
        "CPI": 211.096358,
        "Unemployment": 8.106
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
