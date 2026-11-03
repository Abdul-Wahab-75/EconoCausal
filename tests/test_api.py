import pytest
from fastapi.testclient import TestClient

from src.api import main as api


@pytest.fixture
def client():
    api.estimator = None
    api.trained_feature_names = []
    api.training_observations = 0

    with TestClient(api.app) as test_client:
        yield test_client

    api.estimator = None
    api.trained_feature_names = []
    api.training_observations = 0


def sample_fit_payload():
    return {
        "feature_names": [
            "Age",
            "PriorSpend",
            "VisitFrequency",
            "LoyaltyScore",
        ],
        "features": [
            [22, 1000, 2, 3],
            [25, 1500, 3, 4],
            [30, 2000, 4, 5],
            [35, 2500, 5, 6],
            [40, 3000, 6, 7],
            [45, 3500, 7, 8],
            [24, 1200, 2, 4],
            [28, 1800, 3, 5],
            [32, 2200, 4, 6],
            [38, 2800, 5, 7],
            [42, 3200, 6, 8],
            [48, 4000, 8, 9],
        ],
        "treatment": [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        "outcome": [120, 150, 180, 210, 240, 270, 90, 110, 130, 150, 170, 190],
        "n_splits": 3,
        "nuisance_type": "random_forest",
        "random_state": 42,
    }


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_fitted"] is False


def test_predict_before_fit_returns_conflict(client):
    response = client.post(
        "/predict",
        json={
            "feature_names": ["Age"],
            "features": [[25]],
        },
    )

    assert response.status_code == 409


def test_fit_predict_and_model_info(client):
    fit_response = client.post("/fit", json=sample_fit_payload())

    assert fit_response.status_code == 200
    assert fit_response.json()["status"] == "success"
    assert fit_response.json()["observations"] == 12

    predict_response = client.post(
        "/predict",
        json={
            "feature_names": [
                "Age",
                "PriorSpend",
                "VisitFrequency",
                "LoyaltyScore",
            ],
            "features": [
                [26, 1600, 3, 4],
                [37, 2700, 5, 7],
            ],
        },
    )

    assert predict_response.status_code == 200
    assert predict_response.json()["count"] == 2
    assert len(predict_response.json()["predictions"]) == 2

    info_response = client.get("/model-info")

    assert info_response.status_code == 200
    assert info_response.json()["fitted"] is True
    assert info_response.json()["observations"] == 12


def test_drift_endpoint(client):
    reference = [
        {"Age": 20 + i, "PriorSpend": 1000 + i * 100}
        for i in range(20)
    ]

    current = [
        {"Age": 50 + i, "PriorSpend": 4000 + i * 100}
        for i in range(20)
    ]

    response = client.post(
        "/drift",
        json={
            "reference": reference,
            "current": current,
            "psi_threshold": 0.20,
            "significance_level": 0.05,
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["drift_detected"] is True
    assert "Age" in response.json()["drifted_features"]


def test_drift_endpoint_rejects_invalid_threshold(client):
    response = client.post(
        "/drift",
        json={
            "reference": [{"Age": 20}, {"Age": 30}],
            "current": [{"Age": 40}, {"Age": 50}],
            "significance_level": 1.5,
        },
    )

    assert response.status_code == 422
