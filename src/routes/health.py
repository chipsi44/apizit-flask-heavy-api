"""Lightweight health and non-sensitive runtime information endpoints."""

from __future__ import annotations

import platform
from importlib.metadata import version

import cv2
import numpy
import PIL
import scipy
import sklearn
import torch
import torchvision
import transformers
from flask import Blueprint, current_app, jsonify

from src import __version__
from src.errors import ModelUnavailableError
from src.services.model_registry import ModelRegistry

health_blueprint = Blueprint("health", __name__)


@health_blueprint.get("/ready")
def ready():
    settings = current_app.config["SETTINGS"]
    registry: ModelRegistry = current_app.extensions["model_registry"]
    model_status: dict[str, bool] = {}
    failures: list[str] = []

    for name in ("text", "image"):
        try:
            registry.get(name)
            model_status[name] = True
        except ModelUnavailableError as error:
            model_status[name] = False
            failures.append(error.message)

    payload = {
        "status": "ready" if not failures else "unavailable",
        "models": model_status,
        "device": settings.device,
        "version": __version__,
    }
    if failures:
        payload["error"] = {
            "code": "MODEL_UNAVAILABLE",
            "message": " ".join(failures),
        }
        return jsonify(payload), 503
    return jsonify(payload)


@health_blueprint.get("/health")
def health():
    settings = current_app.config["SETTINGS"]
    return jsonify(status="ok", service=settings.service_name)


@health_blueprint.get("/info")
def info():
    settings = current_app.config["SETTINGS"]
    return jsonify(
        version=__version__,
        python=platform.python_version(),
        libraries={
            "flask": version("flask"),
            "numpy": numpy.__version__,
            "opencv": cv2.__version__,
            "pillow": PIL.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "sentence_transformers": version("sentence-transformers"),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "transformers": transformers.__version__,
        },
        models={"text": settings.text_model_id, "image": settings.image_model_id},
        device=settings.device,
        execution_mode=settings.execution_mode,
    )
