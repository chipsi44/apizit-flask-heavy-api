"""Flask application factory."""

from __future__ import annotations

import logging

from flask import Flask

from src.config import Settings
from src.errors import register_error_handlers
from src.routes.health import health_blueprint
from src.routes.text import text_blueprint
from src.services.model_registry import ModelRegistry
from src.services.text_service import TextService


def create_app(settings: Settings | None = None) -> Flask:
    """Build the Flask application without eagerly loading ML models."""

    active_settings = settings or Settings()
    logging.basicConfig(
        level=getattr(logging, active_settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    app = Flask(__name__)
    app.config.update(
        MAX_CONTENT_LENGTH=active_settings.max_upload_bytes,
        SETTINGS=active_settings,
    )

    registry = ModelRegistry()
    registry.register("text", lambda: TextService(active_settings))
    app.extensions["model_registry"] = registry

    app.register_blueprint(health_blueprint)
    app.register_blueprint(text_blueprint)
    register_error_handlers(app)
    return app


app = create_app()
