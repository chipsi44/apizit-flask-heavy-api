"""Thread-safe, process-local registry for expensive model services."""

from __future__ import annotations

import logging
from collections.abc import Callable
from threading import RLock
from typing import Any

from src.errors import ModelUnavailableError

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Load each registered service at most once per warm process."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], Any]] = {}
        self._instances: dict[str, Any] = {}
        self._lock = RLock()

    def register(self, name: str, factory: Callable[[], Any]) -> None:
        with self._lock:
            if name in self._instances:
                raise RuntimeError(f"Cannot replace loaded model service: {name}")
            self._factories[name] = factory

    def get(self, name: str) -> Any:
        instance = self._instances.get(name)
        if instance is not None:
            return instance

        with self._lock:
            instance = self._instances.get(name)
            if instance is not None:
                return instance

            factory = self._factories.get(name)
            if factory is None:
                raise ModelUnavailableError(f"The '{name}' model service is not configured.")

            try:
                logger.info("Loading %s model service", name)
                instance = factory()
            except ModelUnavailableError:
                raise
            except Exception as error:
                logger.exception("Failed to load %s model service", name)
                raise ModelUnavailableError(
                    f"The '{name}' model could not be loaded. Check the server logs."
                ) from error

            self._instances[name] = instance
            logger.info("Loaded %s model service", name)
            return instance

    def is_loaded(self, name: str) -> bool:
        return name in self._instances

    def clear(self) -> None:
        """Drop cached services; intended for isolated tests only."""

        with self._lock:
            self._instances.clear()
