"""Centralized, environment-driven application configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    """Runtime settings with safe local defaults and no secret material."""

    service_name: str = "apizit-heavy-ml-api"
    version: str = "1.0.0"
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = int(os.getenv("APP_PORT", "5000"))
    debug: bool = _as_bool(os.getenv("APP_DEBUG", "false"))
    log_level: str = os.getenv("APP_LOG_LEVEL", "INFO").upper()
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(8 * 1024 * 1024)))
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
    max_image_pixels: int = int(os.getenv("MAX_IMAGE_PIXELS", "20000000"))
    text_model_id: str = os.getenv("TEXT_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
    text_model_path: str = os.getenv("TEXT_MODEL_PATH", "models/text/all-MiniLM-L6-v2")
    image_model_id: str = "resnet50-imagenet1k-v2"
    image_model_path: str = os.getenv(
        "IMAGE_MODEL_PATH", "models/torchvision/resnet50-imagenet1k-v2.pth"
    )
    device: str = "cpu"

    @property
    def execution_mode(self) -> str:
        return "lambda" if os.getenv("AWS_LAMBDA_FUNCTION_NAME") else "local"
