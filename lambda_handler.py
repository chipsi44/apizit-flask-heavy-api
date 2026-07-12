"""AWS Lambda entry point translating API Gateway events to Flask WSGI."""

from __future__ import annotations

from typing import Any

import serverless_wsgi

from src.app import app


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Handle API Gateway v1/v2 or ALB proxy events through Flask."""

    return serverless_wsgi.handle_request(app, event, context)
