"""
Integration Tests for FastAPI Endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["classes_count"] == 35


def test_classes_endpoint(client):
    response = client.get("/api/classes")
    assert response.status_code == 200
    data = response.json()
    assert data["total_classes"] == 35
    assert "HELLO" in data["common_phrases"]
    assert "HELLO" in data["ui_shortcut_phrases"]
    assert "A" in data["validated_classes"]
    assert "B" in data["validated_classes"]
    assert "C" in data["validated_classes"]


def test_capabilities_endpoint(client):
    response = client.get("/api/system/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert "empirically_validated_classes" in data
    assert "ui_shortcut_phrases" in data
    assert "A" in data["empirically_validated_classes"]
    assert "B" in data["empirically_validated_classes"]
    assert "C" in data["empirically_validated_classes"]
    assert data["dynamic_gesture_sequence_support"] is False


def test_predict_landmarks_endpoint(client):
    dummy_landmarks = [[float(i), float(i * 2)] for i in range(21)]
    response = client.post(
        "/api/predict/landmarks",
        json={"landmarks": dummy_landmarks, "confidence_threshold": 0.5}
    )
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert "top_candidates" in data
    assert "is_empirically_validated" in data


def test_phrase_action_and_state(client):
    # Clear
    client.post("/api/phrase/action", json={"action": "clear"})

    # Add characters
    client.post("/api/phrase/action", json={"action": "add_char", "value": "A"})
    client.post("/api/phrase/action", json={"action": "add_char", "value": "B"})

    # Verify state
    state = client.get("/api/phrase/state").json()
    assert state["text"] == "AB"

    # Space
    client.post("/api/phrase/action", json={"action": "space"})
    client.post("/api/phrase/action", json={"action": "add_phrase", "value": "WATER"})

    state2 = client.get("/api/phrase/state").json()
    assert state2["text"] == "AB WATER"


def test_reference_endpoints(client):
    # Lookup sign 'A'
    res = client.get("/api/reference/sign/A")
    assert res.status_code == 200
    assert res.json()["sign"] == "A"
    assert res.json()["image_url"] == "/static/reference/A.png"

    # Lookup gallery
    gallery_res = client.get("/api/reference/gallery")
    assert gallery_res.status_code == 200
    assert gallery_res.json()["total_signs"] == 35
