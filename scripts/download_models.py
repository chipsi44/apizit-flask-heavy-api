"""Download, verify, and persist the two production models during image build."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer
from torchvision.models import ResNet50_Weights, resnet50

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def download_text_model(model_id: str, destination: Path) -> None:
    logger.info("Downloading text model %s", model_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_id, device="cpu")
    model.eval()
    dimension = model.get_sentence_embedding_dimension()
    if dimension != 384:
        raise RuntimeError(f"Unexpected text embedding dimension: {dimension}")
    model.save_pretrained(str(destination))
    logger.info("Saved text model to %s", destination)


def download_image_model(destination: Path) -> None:
    logger.info("Downloading ResNet50 ImageNet V2 weights")
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    torch.save(model.state_dict(), destination)
    logger.info("Saved image model to %s", destination)


def main() -> None:
    text_model_id = os.getenv("TEXT_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
    text_model_path = Path(os.getenv("TEXT_MODEL_PATH", "models/text/all-MiniLM-L6-v2"))
    image_model_path = Path(
        os.getenv(
            "IMAGE_MODEL_PATH",
            "models/torchvision/resnet50-imagenet1k-v2.pth",
        )
    )

    download_text_model(text_model_id, text_model_path)
    download_image_model(image_model_path)


if __name__ == "__main__":
    main()
