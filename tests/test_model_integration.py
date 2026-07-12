from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from src.config import Settings
from src.services.image_service import ImageService, decode_image
from src.services.text_service import TextService


@pytest.mark.model
def test_real_models_load_and_execute():
    settings = Settings()
    if not Path(settings.text_model_path).is_dir() or not Path(settings.image_model_path).is_file():
        pytest.skip("Run scripts/download_models.py before the real model integration test.")

    text_vector = TextService(settings).embed(["APIZIT deploys a Flask API."])[0]

    buffer = BytesIO()
    Image.new("RGB", (64, 48), color=(210, 120, 40)).save(buffer, format="PNG")
    decoded = decode_image(buffer.getvalue(), "image/png", settings)
    image_service = ImageService(settings)
    image_vector = image_service.embed(decoded)
    predictions = image_service.classify(decoded)

    assert text_vector.shape == (384,)
    assert image_vector.shape == (2048,)
    assert len(predictions) == 5
    assert 0.0 <= predictions[0]["score"] <= 1.0
