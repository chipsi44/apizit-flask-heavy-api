"""Image classification and embedding HTTP routes."""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from src.errors import APIError
from src.services.image_service import DecodedImage, ImageService, decode_image
from src.services.model_registry import ModelRegistry

image_blueprint = Blueprint("image", __name__, url_prefix="/image")


def _uploaded_image() -> DecodedImage:
    upload = request.files.get("file")
    if upload is None:
        raise APIError("INVALID_REQUEST", "The multipart field 'file' is required.", 400)

    settings = current_app.config["SETTINGS"]
    data = upload.stream.read(settings.max_upload_bytes + 1)
    return decode_image(data, upload.mimetype, settings)


def _service() -> ImageService:
    registry: ModelRegistry = current_app.extensions["model_registry"]
    return registry.get("image")


@image_blueprint.post("/analyze")
def analyze():
    decoded = _uploaded_image()
    service = _service()
    return jsonify(
        model=service.model_id,
        image={
            "width": decoded.width,
            "height": decoded.height,
            "format": decoded.format,
        },
        predictions=service.classify(decoded),
    )


@image_blueprint.post("/embedding")
def embedding():
    decoded = _uploaded_image()
    service = _service()
    vector = service.embed(decoded)
    return jsonify(
        model=service.model_id,
        dimension=int(vector.shape[0]),
        embedding=vector.tolist(),
    )
