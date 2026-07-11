"""Tests for the FastAPI app in api/app.py.

These tests exercise the API layer against whatever real trained model(s)
happen to exist under models/ (produced by `python -m src.models.train`).
If no model is trained yet, the health/model-info endpoints must degrade
gracefully rather than crash — that behavior itself is what's tested.
"""

from __future__ import annotations

from api.app import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint_returns_valid_schema():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "model_loaded" in body
    assert body["status"] in {"ok", "degraded"}


def test_predict_rejects_invalid_store_id():
    response = client.post("/predict", json={"store": -1, "item": 1, "date": "2018-01-15"})
    assert response.status_code == 422  # pydantic validation error (store must be >= 1)


def test_predict_rejects_malformed_date():
    response = client.post("/predict", json={"store": 1, "item": 1, "date": "not-a-date"})
    assert response.status_code == 422


def test_forecast_rejects_horizon_over_90_days():
    response = client.post("/forecast", json={"store": 1, "item": 1, "horizon_days": 200})
    assert response.status_code == 422
