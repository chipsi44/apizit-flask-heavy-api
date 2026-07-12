from __future__ import annotations

import numpy as np
import pytest

from src.services.text_service import TextService


def test_text_embedding_contract(client):
    response = client.post("/text/embedding", json={"text": "Deploy with APIZIT"})

    assert response.status_code == 200
    assert response.get_json() == {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 3,
        "embedding": [0.25, 0.5, 0.75],
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "A JSON request body is required."),
        ({"data": "{", "content_type": "application/json"}, "valid JSON object"),
        ({"json": {}}, "field 'text' is required"),
        ({"json": {"text": 123}}, "must be a string"),
        ({"json": {"text": "   "}}, "must not be empty"),
        ({"json": {"text": "x" * 33}}, "32 character limit"),
    ],
)
def test_text_embedding_validation(client, kwargs, message):
    response = client.post("/text/embedding", **kwargs)

    assert response.status_code == 400
    error = response.get_json()["error"]
    assert set(error) == {"code", "message"}
    assert message in error["message"]


def test_text_similarity_contract(client):
    response = client.post(
        "/text/similarity",
        json={"left": "Deploy a Flask API", "right": "Publish a Python service"},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "similarity": 0.8125,
    }


def test_similarity_uses_cosine_similarity(monkeypatch):
    service = object.__new__(TextService)
    embeddings = np.asarray([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    monkeypatch.setattr(service, "embed", lambda _texts: embeddings)

    assert service.similarity("left", "right") == pytest.approx(2**-0.5)
