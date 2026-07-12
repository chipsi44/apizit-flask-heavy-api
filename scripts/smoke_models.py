"""Run one real inference through each downloaded production model."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from src.config import Settings
from src.services.image_service import ImageService, decode_image
from src.services.text_service import TextService


def main() -> None:
    settings = Settings()

    text_service = TextService(settings)
    text_vector = text_service.embed(["APIZIT deploys Python APIs."])[0]
    if text_vector.shape != (384,):
        raise RuntimeError(f"Unexpected text shape: {text_vector.shape}")

    buffer = BytesIO()
    Image.new("RGB", (64, 48), color=(210, 120, 40)).save(buffer, format="PNG")
    decoded = decode_image(buffer.getvalue(), "image/png", settings)
    image_service = ImageService(settings)
    image_vector = image_service.embed(decoded)
    predictions = image_service.classify(decoded)
    if image_vector.shape != (2048,) or len(predictions) != 5:
        raise RuntimeError(
            f"Unexpected image result: shape={image_vector.shape}, predictions={len(predictions)}"
        )

    print(
        {
            "text_embedding_dimension": int(text_vector.shape[0]),
            "image_embedding_dimension": int(image_vector.shape[0]),
            "top_image_prediction": predictions[0],
        }
    )


if __name__ == "__main__":
    main()
