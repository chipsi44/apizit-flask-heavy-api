from __future__ import annotations


def test_http_errors_use_homogeneous_json(client):
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.is_json
    assert set(response.get_json()["error"]) == {"code", "message"}


def test_unexpected_errors_do_not_expose_stack_traces(app):
    @app.get("/test-error")
    def test_error():
        raise RuntimeError("sensitive-internal-detail")

    response = app.test_client().get("/test-error")

    assert response.status_code == 500
    assert response.get_json()["error"]["code"] == "INTERNAL_ERROR"
    assert "sensitive-internal-detail" not in response.get_data(as_text=True)
