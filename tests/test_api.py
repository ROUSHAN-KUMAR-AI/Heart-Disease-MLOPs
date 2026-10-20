"""Tests for the FastAPI serving layer (offline, uses the trained_model_path fixture)."""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app

VALID = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}


@pytest.fixture()
def client(trained_model_path):
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["model_loaded"] is True


def test_predict_returns_valid_schema(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"]


def test_predict_rejects_out_of_range(client):
    bad = dict(VALID, sex=5)  # sex must be 0/1
    r = client.post("/predict", json=bad)
    assert r.status_code == 422


def test_metrics_endpoint_exposes_prometheus(client):
    client.post("/predict", json=VALID)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "model_predictions_total" in r.text
