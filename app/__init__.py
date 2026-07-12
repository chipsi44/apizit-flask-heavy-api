"""Flask application factory for the heavy multimodal API."""

from __future__ import annotations

import logging

from flask import Flask

from app.config import Settings
from app.errors import register_error_handlers
from app.ml import ImageService, ModelRegistry, TextService
from app.routes import api

__version__ = "1.0.0"


def create_app(settings: Settings | None = None) -> Flask:
    """Create the Flask app without loading either model eagerly."""

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
    registry.register("image", lambda: ImageService(active_settings))
    app.extensions["model_registry"] = registry

    app.register_blueprint(api)
    register_error_handlers(app)
    return app


app = create_app()
