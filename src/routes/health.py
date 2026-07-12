"""Lightweight health and non-sensitive runtime information endpoints."""

from __future__ import annotations

import platform

import cv2
import flask
import numpy
import PIL
import scipy
import sklearn
import torch
import torchvision
import transformers
from flask import Blueprint, current_app, jsonify
from sentence_transformers import __version__ as sentence_transformers_version

from src import __version__

health_blueprint = Blueprint("health", __name__)


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
            "flask": flask.__version__,
            "numpy": numpy.__version__,
            "opencv": cv2.__version__,
            "pillow": PIL.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "sentence_transformers": sentence_transformers_version,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "transformers": transformers.__version__,
        },
        models={"text": settings.text_model_id, "image": settings.image_model_id},
        device=settings.device,
        execution_mode=settings.execution_mode,
    )
