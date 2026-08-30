from __future__ import annotations


def test_health_is_lightweight(client, app):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
    registry = app.extensions["model_registry"]
    assert not registry.is_loaded("text")
    assert not registry.is_loaded("image")


def test_ready_loads_both_models(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "ready",
        "models": {"text": True, "image": True},
        "device": "cpu",
        "version": "1.0.0",
    }


def test_info_exposes_only_allowlisted_runtime_data(client, monkeypatch):
    monkeypatch.setenv("PRIVATE_TEST_VALUE", "must-not-leak")
    response = client.get("/info")

    assert response.status_code == 200
    body = response.get_json()
    assert body["framework"] == "flask"
    assert body["profile"] == "heavy"
    assert body["device"] == "cpu"
    assert body["models"]["text"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert "must-not-leak" not in response.get_data(as_text=True)
