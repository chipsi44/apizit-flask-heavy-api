from __future__ import annotations


def test_http_errors_use_homogeneous_json(client):
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.is_json
    assert set(response.get_json()["error"]) == {"code", "message"}


def test_unexpected_errors_do_not_expose_stack_traces(app):
    def raise_unexpected_error():
        raise RuntimeError("sensitive-internal-detail")

    original_health = app.view_functions["api.health"]
    app.view_functions["api.health"] = raise_unexpected_error
    try:
        response = app.test_client().get("/health")
    finally:
        app.view_functions["api.health"] = original_health

    assert response.status_code == 500
    assert response.get_json()["error"]["code"] == "INTERNAL_ERROR"
    assert "sensitive-internal-detail" not in response.get_data(as_text=True)
