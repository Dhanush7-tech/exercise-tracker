"""
API endpoint tests using FastAPI's TestClient.

Deliberately NOT tested here: POST /api/retrain. That endpoint writes to the
real, shipped model files in models/ -- exercising it through the live `app`
object would overwrite the production model with one retrained on test data.
The retraining logic itself is fully covered in test_retrain.py using an
isolated model directory instead.
"""
import pytest
from fastapi.testclient import TestClient

import synthetic_data as sd
from api import app

client = TestClient(app)


def test_health_check():
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_dashboard_serves_html():
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]


def test_exercises_endpoint_returns_known_classes():
    res = client.get("/api/exercises")
    assert res.status_code == 200
    exercises = res.json()["exercises"]
    assert set(exercises) == set(sd.EXERCISE_PROFILES.keys())


def test_create_session_returns_id_and_label():
    res = client.post("/api/sessions", json={"label": "api test session"})
    assert res.status_code == 200
    data = res.json()
    assert "session_id" in data
    assert data["label"] == "api test session"


def test_generate_and_log_end_to_end():
    session_id = client.post("/api/sessions", json={"label": "generate test"}).json()["session_id"]

    res = client.post("/api/generate-and-log", json={
        "session_id": session_id,
        "exercise": "squat",
        "num_sets": 2,
        "reps_per_set": 8,
    })
    assert res.status_code == 200
    data = res.json()
    assert data["session_id"] == session_id
    assert len(data["generated"]) == 2
    for s in data["generated"]:
        assert s["predicted_exercise"] in sd.EXERCISE_PROFILES
        assert 0.0 <= s["confidence"] <= 1.0


def test_generate_and_log_rejects_unknown_exercise():
    session_id = client.post("/api/sessions", json={"label": "bad exercise test"}).json()["session_id"]
    res = client.post("/api/generate-and-log", json={
        "session_id": session_id,
        "exercise": "not_a_real_exercise",
        "num_sets": 1,
        "reps_per_set": 8,
    })
    assert res.status_code == 400


def test_generate_and_log_rejects_absurd_set_count():
    session_id = client.post("/api/sessions", json={"label": "too many sets test"}).json()["session_id"]
    res = client.post("/api/generate-and-log", json={
        "session_id": session_id,
        "exercise": "squat",
        "num_sets": 999,
        "reps_per_set": 8,
    })
    assert res.status_code == 400


def test_history_reflects_generated_sets():
    session_id = client.post("/api/sessions", json={"label": "history test"}).json()["session_id"]
    client.post("/api/generate-and-log", json={
        "session_id": session_id, "exercise": "bench", "num_sets": 1, "reps_per_set": 8,
    })
    res = client.get("/api/history")
    assert res.status_code == 200
    sets = res.json()["sets"]
    assert any(s["session_id"] == session_id for s in sets)


def test_analytics_endpoint_shape():
    res = client.get("/api/analytics")
    assert res.status_code == 200
    data = res.json()
    for key in ["total_sets", "total_reps", "total_sessions", "by_exercise",
                "intended_vs_predicted_accuracy", "by_day"]:
        assert key in data


def test_retrain_status_endpoint_shape():
    res = client.get("/api/retrain/status")
    assert res.status_code == 200
    data = res.json()
    for key in ["total_labeled_sets", "sets_per_exercise", "ready_to_retrain", "history"]:
        assert key in data
