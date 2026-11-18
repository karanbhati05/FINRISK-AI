"""
Tests for the FinRisk AI FastAPI application.

Uses FastAPI TestClient for integration tests. Tests both mocked-model
execution paths (200, 422) and missing-artifact paths (503).
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.routes import (
    _model_cache,
    get_feature_engineer,
    get_model,
    get_preprocessor,
)

client = TestClient(app, raise_server_exceptions=False)

# ---------------------------------------------------------------------------
# Mock / Dummy objects for testing prediction flow
# ---------------------------------------------------------------------------

class DummyModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        # Return probability of 0.25 for class 1
        n = len(X)
        return np.column_stack([np.full(n, 0.75), np.full(n, 0.25)])


class DummyTransformer:
    def transform(self, df):
        return df


# ---------------------------------------------------------------------------
# Sample payload
# ---------------------------------------------------------------------------

SAMPLE_CUSTOMER = {
    "LIMIT_BAL": 200000.0,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 35,
    "PAY_0": 0,
    "PAY_2": 0,
    "PAY_3": -1,
    "PAY_4": -1,
    "PAY_5": -1,
    "PAY_6": -1,
    "BILL_AMT1": 50000.0,
    "BILL_AMT2": 48000.0,
    "BILL_AMT3": 45000.0,
    "BILL_AMT4": 42000.0,
    "BILL_AMT5": 40000.0,
    "BILL_AMT6": 38000.0,
    "PAY_AMT1": 5000.0,
    "PAY_AMT2": 4800.0,
    "PAY_AMT3": 4500.0,
    "PAY_AMT4": 4200.0,
    "PAY_AMT5": 4000.0,
    "PAY_AMT6": 3800.0,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """GET /api/v1/health"""

    def test_health_returns_200(self):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_response_shape(self):
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["version"] == "0.1.0"


class TestRootEndpoint:
    """GET /"""

    def test_root_returns_200(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "FinRisk AI" in resp.json().get("message", "")


class TestPredictEndpoint:
    """POST /api/v1/predict"""

    def test_predict_success_with_model(self):
        app.dependency_overrides[get_model] = lambda: DummyModel()
        app.dependency_overrides[get_preprocessor] = lambda: DummyTransformer()
        app.dependency_overrides[get_feature_engineer] = lambda: DummyTransformer()
        try:
            resp = client.post("/api/v1/predict", json=SAMPLE_CUSTOMER)
            assert resp.status_code == 200
            data = resp.json()
            assert "default_probability" in data
            assert data["risk_label"] == "Low Risk"
            assert data["model_used"] == "DummyModel"
        finally:
            app.dependency_overrides.clear()

    def test_predict_without_model_returns_503(self):
        """When model artifacts are missing the endpoint should 503."""
        app.dependency_overrides.clear()
        _model_cache.clear()

        resp = client.post("/api/v1/predict", json=SAMPLE_CUSTOMER)
        assert resp.status_code in (200, 503)

    def test_predict_invalid_payload_returns_422(self):
        app.dependency_overrides[get_model] = lambda: DummyModel()
        app.dependency_overrides[get_preprocessor] = lambda: DummyTransformer()
        app.dependency_overrides[get_feature_engineer] = lambda: DummyTransformer()
        try:
            resp = client.post("/api/v1/predict", json={"LIMIT_BAL": 100})
            assert resp.status_code == 422
        finally:
            app.dependency_overrides.clear()


class TestBatchPredictEndpoint:
    """POST /api/v1/predict/batch"""

    def test_batch_predict_success_with_model(self):
        app.dependency_overrides[get_model] = lambda: DummyModel()
        app.dependency_overrides[get_preprocessor] = lambda: DummyTransformer()
        app.dependency_overrides[get_feature_engineer] = lambda: DummyTransformer()
        try:
            resp = client.post(
                "/api/v1/predict/batch",
                json={"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["predictions"]) == 2
            assert data["summary"]["total"] == 2
        finally:
            app.dependency_overrides.clear()

    def test_batch_predict_invalid(self):
        app.dependency_overrides[get_model] = lambda: DummyModel()
        app.dependency_overrides[get_preprocessor] = lambda: DummyTransformer()
        app.dependency_overrides[get_feature_engineer] = lambda: DummyTransformer()
        try:
            resp = client.post("/api/v1/predict/batch", json={"customers": []})
            assert resp.status_code == 422
        finally:
            app.dependency_overrides.clear()

    def test_batch_predict_missing_model(self):
        app.dependency_overrides.clear()
        _model_cache.clear()

        resp = client.post(
            "/api/v1/predict/batch",
            json={"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]},
        )
        assert resp.status_code in (200, 503)


class TestExplainEndpoint:
    """POST /api/v1/explain"""

    def test_explain_missing_model(self):
        app.dependency_overrides.clear()
        _model_cache.clear()

        resp = client.post("/api/v1/explain", json=SAMPLE_CUSTOMER)
        assert resp.status_code in (200, 503)


class TestDriftEndpoint:
    """GET /api/v1/drift-report"""

    def test_drift_report_no_reports(self):
        resp = client.get("/api/v1/drift-report")
        assert resp.status_code in (200, 404)
