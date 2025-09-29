import os
from fastapi.testclient import TestClient

# Speed up imports/startup in tests: skip training; avoid on-demand predictions.
os.environ.setdefault("ML_SKIP_STARTUP_TRAIN", "1")
os.environ.setdefault("ALLOW_ON_DEMAND_PREDICTIONS", "0")

from app.main import app  # noqa: E402


client = TestClient(app)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") in {"healthy", "degraded"}


def test_teams_endpoint():
    r = client.get("/api/teams")
    assert r.status_code == 200
    data = r.json()
    assert "teams" in data
    assert isinstance(data["teams"], list)


def test_matches_endpoint():
    r = client.get("/api/matches?limit=5")
    assert r.status_code == 200
    data = r.json()
    assert "matches" in data
    assert isinstance(data["matches"], list)


def test_top_scorers_endpoint():
    r = client.get("/api/players/top-scorers?limit=5")
    assert r.status_code == 200
    data = r.json()
    assert "top_scorers" in data
    assert isinstance(data["top_scorers"], list)


def test_week_odds_endpoint():
    r = client.get("/api/betting/odds/week/5?limit=3")
    assert r.status_code == 200
    data = r.json()
    assert data.get("week") == 5
    assert "matches" in data
    assert isinstance(data["matches"], list)


def test_week_edges_endpoint():
    r = client.get(
        "/api/betting/edges/week/5?limit=3&threshold=0.01&allow_on_demand=true"
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("week") == 5
    assert "matches" in data
    assert isinstance(data["matches"], list)
