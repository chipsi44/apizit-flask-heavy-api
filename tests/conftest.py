from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app import create_app
from app.config import Settings


class StubTextService:
    model_id = "sentence-transformers/all-MiniLM-L6-v2"
    dimension = 384

    def embed(self, texts: list[str]) -> np.ndarray:
        base = np.asarray([0.25, 0.5, 0.75], dtype=np.float32)
        return np.stack([base + index for index, _text in enumerate(texts)])

    def similarity(self, left: str, right: str) -> float:
        return 0.8125


class StubImageService:
    model_id = "resnet50"
    weights_id = "resnet50-imagenet1k-v2"
    dimension = 2048

    def classify(self, _decoded, top_k: int = 5):
        predictions = [
            {"label": "orange", "score": 0.8},
            {"label": "lemon", "score": 0.1},
            {"label": "banana", "score": 0.05},
            {"label": "pineapple", "score": 0.03},
            {"label": "strawberry", "score": 0.02},
        ]
        return predictions[:top_k]

    def embed(self, _decoded) -> np.ndarray:
        return np.arange(self.dimension, dtype=np.float32)


@pytest.fixture()
def settings(tmp_path):
    return Settings(
        max_upload_bytes=1024 * 1024,
        max_text_length=32,
        model_cache_dir=str(tmp_path / "models"),
    )


@pytest.fixture()
def app(settings):
    application = create_app(settings)
    application.config.update(TESTING=True)
    registry = application.extensions["model_registry"]
    registry.register("text", StubTextService)
    registry.register("image", StubImageService)
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (24, 16), color=(210, 120, 40)).save(buffer, format="PNG")
    return buffer.getvalue()
