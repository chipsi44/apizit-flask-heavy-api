"""JSON API errors and Flask error-handler registration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from flask import Flask, jsonify
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class APIError(Exception):
    """An expected client-facing error with a stable machine-readable code."""

    code: str
    message: str
    status_code: int


class ModelUnavailableError(APIError):
    """Raised when a configured ML model cannot be loaded."""

    def __init__(self, message: str) -> None:
        super().__init__("MODEL_UNAVAILABLE", message, 503)


def error_payload(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def register_error_handlers(app: Flask) -> None:
    """Register homogeneous JSON responses without leaking stack traces."""

    @app.errorhandler(APIError)
    def handle_api_error(error: APIError):
        return jsonify(error_payload(error.code, error.message)), error.status_code

    @app.errorhandler(413)
    def handle_too_large(_error):
        return (
            jsonify(
                error_payload(
                    "PAYLOAD_TOO_LARGE",
                    f"The request exceeds the {app.config['MAX_CONTENT_LENGTH']} byte limit.",
                )
            ),
            413,
        )

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException):
        return (
            jsonify(error_payload("HTTP_ERROR", error.description or "HTTP request failed.")),
            error.code or 500,
        )

    @app.errorhandler(Exception)
    def handle_unexpected_error(error: Exception):
        logger.exception("Unhandled request error", exc_info=error)
        return (
            jsonify(error_payload("INTERNAL_ERROR", "An unexpected error occurred.")),
            500,
        )
