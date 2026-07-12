"""Text embedding and semantic similarity HTTP routes."""

from __future__ import annotations

from typing import Any

from flask import Blueprint, current_app, jsonify, request

from src.errors import APIError
from src.services.model_registry import ModelRegistry
from src.services.text_service import TextService

text_blueprint = Blueprint("text", __name__, url_prefix="/text")


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


def _service() -> TextService:
    registry: ModelRegistry = current_app.extensions["model_registry"]
    return registry.get("text")


@text_blueprint.post("/embedding")
def embedding():
    text = _text_field(_json_object(), "text")
    service = _service()
    vector = service.embed([text])[0]
    return jsonify(
        model=service.model_id,
        dimension=int(vector.shape[0]),
        embedding=vector.tolist(),
    )


@text_blueprint.post("/similarity")
def similarity():
    payload = _json_object()
    left = _text_field(payload, "left")
    right = _text_field(payload, "right")
    service = _service()
    return jsonify(model=service.model_id, similarity=service.similarity(left, right))
