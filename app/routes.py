"""All HTTP routes for the standalone Flask API."""

from __future__ import annotations

import platform
from importlib.metadata import version
from typing import Any

import cv2
import numpy
import PIL
import scipy
import sklearn
import torch
import torchvision
import transformers
from flask import Blueprint, current_app, jsonify, request

from app.errors import APIError, ModelUnavailableError
from app.ml import DecodedImage, ImageService, ModelRegistry, TextService, decode_image

api = Blueprint("api", __name__)


def _registry() -> ModelRegistry:
    return current_app.extensions["model_registry"]


def _json_object() -> dict[str, Any]:
    if not request.is_json:
        raise APIError("INVALID_JSON", "A JSON request body is required.", 400)
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise APIError("INVALID_JSON", "The request body must be a valid JSON object.", 400)
    return payload


def _text_field(payload: dict[str, Any], field: str) -> str:
    if field not in payload:
        raise APIError("INVALID_REQUEST", f"The field '{field}' is required.", 400)
    value = payload[field]
    if not isinstance(value, str):
        raise APIError("INVALID_REQUEST", f"The field '{field}' must be a string.", 400)
    value = value.strip()
    if not value:
        raise APIError("INVALID_REQUEST", f"The field '{field}' must not be empty.", 400)
    max_length = current_app.config["SETTINGS"].max_text_length
    if len(value) > max_length:
        raise APIError(
            "TEXT_TOO_LONG",
            f"The field '{field}' exceeds the {max_length} character limit.",
            400,
        )
    return value


def _uploaded_image() -> DecodedImage:
    upload = request.files.get("file")
    if upload is None:
        raise APIError("INVALID_REQUEST", "The multipart field 'file' is required.", 400)
    settings = current_app.config["SETTINGS"]
    data = upload.stream.read(settings.max_upload_bytes + 1)
    return decode_image(data, upload.mimetype, settings)


@api.get("/")
def index():
    return jsonify(
        name="APIZIT Heavy ML API",
        endpoints=[
            "GET /health",
            "GET /ready",
            "GET /info",
            "POST /text/embedding",
            "POST /text/similarity",
            "POST /image/analyze",
            "POST /image/embedding",
        ],
    )


@api.get("/health")
def health():
    return jsonify(status="ok", service=current_app.config["SETTINGS"].service_name)


@api.get("/ready")
def ready():
    model_status: dict[str, bool] = {}
    failures: list[str] = []
    for name in ("text", "image"):
        try:
            _registry().get(name)
            model_status[name] = True
        except ModelUnavailableError as error:
            model_status[name] = False
            failures.append(error.message)
    payload = {
        "status": "ready" if not failures else "unavailable",
        "models": model_status,
        "device": "cpu",
        "version": current_app.config["SETTINGS"].version,
    }
    if failures:
        payload["error"] = {"code": "MODEL_UNAVAILABLE", "message": " ".join(failures)}
        return jsonify(payload), 503
    return jsonify(payload)


@api.get("/info")
def info():
    settings = current_app.config["SETTINGS"]
    return jsonify(
        version=settings.version,
        runtime="flask",
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
        device="cpu",
    )


@api.post("/text/embedding")
def text_embedding():
    text = _text_field(_json_object(), "text")
    service: TextService = _registry().get("text")
    vector = service.embed([text])[0]
    return jsonify(
        model=service.model_id,
        dimension=int(vector.shape[0]),
        embedding=vector.tolist(),
    )


@api.post("/text/similarity")
def text_similarity():
    payload = _json_object()
    left = _text_field(payload, "left")
    right = _text_field(payload, "right")
    service: TextService = _registry().get("text")
    return jsonify(model=service.model_id, similarity=service.similarity(left, right))


@api.post("/image/analyze")
def image_analyze():
    decoded = _uploaded_image()
    service: ImageService = _registry().get("image")
    return jsonify(
        model=service.model_id,
        image={"width": decoded.width, "height": decoded.height, "format": decoded.format},
        predictions=service.classify(decoded),
    )


@api.post("/image/embedding")
def image_embedding():
    decoded = _uploaded_image()
    service: ImageService = _registry().get("image")
    vector = service.embed(decoded)
    return jsonify(
        model=service.model_id,
        dimension=int(vector.shape[0]),
        embedding=vector.tolist(),
    )
