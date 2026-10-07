import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from gtosolver.api import _solve_lock, app, metal_status


client = TestClient(app)
BASE = Path(__file__).resolve().parents[1]


def example(name="heads-up-river"):
    payload = json.loads((BASE / "examples" / f"{name}.json").read_text())
    payload.update(iterations=10, samples=32, backend="cpu")
    return payload


def test_health_and_openapi():
    status = client.get("/health")
    assert status.status_code == 200
    assert status.json()["cpu"]["available"]
    schema = client.get("/openapi.json").json()
    assert "/v1/solve" in schema["paths"]
    assert schema["components"]["schemas"]["SolveRequest"]["properties"]["backend"]["default"] == "metal"


@pytest.mark.parametrize("change", [
    {"board": ["As", "AS", "Qd"]},
    {"board": ["As", "Kd"]},
    {"pot": 0},
    {"effective_stack": -1},
    {"pot": True},
    {"iterations": True},
    {"backend": "cuda"},
    {"iterations": 10001},
    {"bet_sizes": [float("inf")]},
    {"vision": True},
    {"players": [{"name": "Only", "range": "AA"}]},
    {"players": [{"name": "Same", "range": "AA"}, {"name": "Same", "range": "KK"}]},
])
def test_invalid_input(change):
    payload = example()
    payload.update(change)
    # JSON itself excludes Infinity; test the finite schema with a string instead.
    if change.get("bet_sizes") == [float("inf")]:
        payload["bet_sizes"] = ["Infinity"]
    assert client.post("/v1/solve", json=payload).status_code == 422


def test_invalid_range_and_impossible_matchup():
    payload = example()
    payload["players"][0]["range"] = "not-a-hand"
    assert client.post("/v1/solve", json=payload).status_code == 422
    payload["players"] = [{"name": "A", "range": "AsAh"}, {"name": "B", "range": "AsAd"}]
    assert client.post("/v1/solve", json=payload).status_code == 422


def test_single_solver_worker():
    _solve_lock.acquire()
    try:
        response = client.post("/v1/solve", json=example())
        assert response.status_code == 429
    finally:
        _solve_lock.release()


def test_unavailable_metal_never_falls_back(monkeypatch):
    monkeypatch.setattr("gtosolver.api.metal_status", lambda: {"available": False, "reason": "test unavailable"})
    payload = example()
    payload["backend"] = "metal"
    response = client.post("/v1/solve", json=payload)
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "metal_unavailable"


@pytest.mark.parametrize("name", ["heads-up-river", "multiplayer-river"])
def test_api_solve(name):
    response = client.post("/v1/solve", json=example(name))
    assert response.status_code == 200, response.text
    result = response.json()
    assert len(result["players"]) == len(example(name)["players"])
    assert result["request"]["backend"] == "cpu"
    assert result["diagnostics"]["api_elapsed_ms"] > 0
    assert isinstance(result["limitations"], list)


@pytest.mark.metal
def test_api_multiplayer_metal():
    if not metal_status()["available"]:
        pytest.skip("Metal unavailable")
    payload = example("multiplayer-river")
    payload["backend"] = "metal"
    response = client.post("/v1/solve", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["backend"]["actual"] == "metal"
