from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

from werkzeug.test import EnvironBuilder

import lambda_handler
from tests.conftest import StubImageService, StubTextService


def _event(method: str, path: str, *, body: str | None = None, headers=None, encoded=False):
    return {
        "version": "2.0",
        "routeKey": f"{method} {path}",
        "rawPath": path,
        "rawQueryString": "",
        "headers": headers or {"host": "localhost"},
        "requestContext": {
            "http": {
                "method": method,
                "path": path,
                "protocol": "HTTP/1.1",
                "sourceIp": "127.0.0.1",
                "userAgent": "pytest",
            },
            "requestId": "pytest",
            "routeKey": f"{method} {path}",
            "stage": "$default",
            "time": "01/Jan/2026:00:00:00 +0000",
            "timeEpoch": 1767225600000,
        },
        "body": body,
        "isBase64Encoded": encoded,
    }


def test_lambda_handler_accepts_api_gateway_v2_health_event():
    event = json.loads(Path("examples/events/health.json").read_text(encoding="utf-8"))

    response = lambda_handler.handler(event, None)

    assert response["statusCode"] == 200
    assert json.loads(response["body"])["status"] == "ok"


def test_lambda_handler_decodes_base64_multipart_upload(png_bytes):
    registry = lambda_handler.app.extensions["model_registry"]
    registry.clear()
    registry.register("text", StubTextService)
    registry.register("image", StubImageService)

    builder = EnvironBuilder(
        method="POST",
        path="/image/analyze",
        data={"file": (BytesIO(png_bytes), "sample.png", "image/png")},
    )
    environ = builder.get_environ()
    raw_body = environ["wsgi.input"].read()
    content_type = environ["CONTENT_TYPE"]
    event = _event(
        "POST",
        "/image/analyze",
        body=base64.b64encode(raw_body).decode("ascii"),
        headers={"host": "localhost", "content-type": content_type},
        encoded=True,
    )

    response = lambda_handler.handler(event, None)

    assert response["statusCode"] == 200
    assert json.loads(response["body"])["image"]["format"] == "PNG"
