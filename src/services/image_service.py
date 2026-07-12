"""ResNet50 image classification and feature extraction on CPU."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy.special import softmax
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from src.config import Settings
from src.errors import APIError, ModelUnavailableError

ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG", "WEBP"}


@dataclass(frozen=True, slots=True)
class DecodedImage:
    """Validated image plus metadata retained before model preprocessing."""

    image: Image.Image
    width: int
    height: int
    format: str


def decode_image(data: bytes, mime_type: str, settings: Settings) -> DecodedImage:
    """Validate encoded image bytes and apply deterministic OpenCV enhancement."""

    if mime_type not in ALLOWED_IMAGE_MIME_TYPES:
        raise APIError(
            "UNSUPPORTED_MEDIA_TYPE",
            "Only JPEG, PNG, and WebP images are supported.",
            415,
        )
    if not data:
        raise APIError("INVALID_IMAGE", "The uploaded image is empty.", 400)
    if len(data) > settings.max_upload_bytes:
        raise APIError(
            "PAYLOAD_TOO_LARGE",
            f"The image exceeds the {settings.max_upload_bytes} byte limit.",
            413,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as candidate:
                image_format = candidate.format or ""
                width, height = candidate.size
                if width * height > settings.max_image_pixels:
                    raise APIError(
                        "IMAGE_TOO_LARGE",
                        f"The image exceeds the {settings.max_image_pixels} pixel limit.",
                        413,
                    )
                candidate.verify()

            with Image.open(BytesIO(data)) as candidate:
                image = ImageOps.exif_transpose(candidate).convert("RGB")
    except APIError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise APIError("IMAGE_TOO_LARGE", "The image dimensions are unsafe.", 413) from error
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise APIError("INVALID_IMAGE", "The file is not a valid supported image.", 400) from error

    if image_format not in ALLOWED_IMAGE_FORMATS:
        raise APIError(
            "UNSUPPORTED_MEDIA_TYPE",
            "The encoded image must be JPEG, PNG, or WebP.",
            415,
        )

    rgb = np.asarray(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab)
    enhanced_lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    enhanced_lab = cv2.merge((enhanced_lightness, green_red, blue_yellow))
    enhanced_rgb = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    enhanced_image = Image.fromarray(enhanced_rgb, mode="RGB")

    return DecodedImage(
        image=enhanced_image,
        width=image.width,
        height=image.height,
        format=image_format,
    )


class ImageService:
    """Own one pretrained ResNet50 and reuse it across warm requests."""

    def __init__(self, settings: Settings) -> None:
        self.model_id = "resnet50"
        self.weights_id = settings.image_model_id
        weights = ResNet50_Weights.IMAGENET1K_V2
        weights_path = Path(settings.image_model_path)
        offline = os.getenv("TORCH_MODEL_OFFLINE", "0") == "1"

        if weights_path.is_file():
            model = resnet50(weights=None)
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
        elif offline:
            raise ModelUnavailableError(
                f"The image model file is missing at '{settings.image_model_path}'."
            )
        else:
            model = resnet50(weights=weights)

        self._model = model.eval()
        self._feature_extractor = nn.Sequential(*list(model.children())[:-1]).eval()
        self._transform = weights.transforms()
        self._categories: list[str] = list(weights.meta["categories"])
        self.dimension = 2048

    def _tensor(self, decoded: DecodedImage) -> torch.Tensor:
        return self._transform(decoded.image).unsqueeze(0)

    def classify(self, decoded: DecodedImage, top_k: int = 5) -> list[dict[str, float | str]]:
        """Return the highest-scoring real ImageNet predictions."""

        with torch.inference_mode():
            logits = self._model(self._tensor(decoded))[0].cpu().numpy()
        probabilities = softmax(logits)
        indices = np.argsort(probabilities)[::-1][:top_k]
        return [
            {"label": self._categories[int(index)], "score": float(probabilities[index])}
            for index in indices
        ]

    def embed(self, decoded: DecodedImage) -> np.ndarray:
        """Return the 2048-value activation from ResNet50's average-pool layer."""

        with torch.inference_mode():
            vector = self._feature_extractor(self._tensor(decoded)).flatten(1)[0]
        return vector.cpu().numpy().astype(np.float32, copy=False)
