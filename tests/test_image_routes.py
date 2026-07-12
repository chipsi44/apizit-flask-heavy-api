from __future__ import annotations

from io import BytesIO

import pytest


def test_image_analyze_contract(client, png_bytes):
    response = client.post(
        "/image/analyze",
        data={"file": (BytesIO(png_bytes), "sample.png", "image/png")},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["model"] == "resnet50"
    assert body["image"] == {"width": 24, "height": 16, "format": "PNG"}
    assert len(body["predictions"]) == 5
    assert body["predictions"][0] == {"label": "orange", "score": 0.8}


def test_image_embedding_contract(client, png_bytes):
    response = client.post(
        "/image/embedding",
        data={"file": (BytesIO(png_bytes), "sample.png", "image/png")},
    )

    assert response.status_code == 200
    body = response.get_json()
    assert body["model"] == "resnet50"
    assert body["dimension"] == 2048
    assert len(body["embedding"]) == 2048
    assert body["embedding"][:3] == [0.0, 1.0, 2.0]


def test_image_requires_file(client):
    response = client.post("/image/analyze")

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "INVALID_REQUEST"


@pytest.mark.parametrize(
    ("content", "filename", "mime_type", "status", "code"),
    [
        (b"not-an-image", "sample.txt", "text/plain", 415, "UNSUPPORTED_MEDIA_TYPE"),
        (b"not-an-image", "sample.png", "image/png", 400, "INVALID_IMAGE"),
    ],
)
def test_image_rejects_invalid_files(client, content, filename, mime_type, status, code):
    response = client.post(
        "/image/analyze",
        data={"file": (BytesIO(content), filename, mime_type)},
    )

    assert response.status_code == status
    assert response.get_json()["error"]["code"] == code
