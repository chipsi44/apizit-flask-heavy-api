"""Sentence Transformer inference for text embeddings and similarity."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import Settings


class TextService:
    """Own one CPU Sentence Transformer instance for the process lifetime."""

    def __init__(self, settings: Settings) -> None:
        self.model_id = settings.text_model_id
        configured_path = Path(settings.text_model_path)
        offline = (
            os.getenv("HF_HUB_OFFLINE", "0") == "1" or os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
        )
        source = str(configured_path) if configured_path.is_dir() else self.model_id

        self._model = SentenceTransformer(
            source,
            device="cpu",
            local_files_only=offline,
        )
        self._model.eval()
        self.dimension = int(self._model.get_sentence_embedding_dimension() or 0)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return normalized, real embeddings for one or more input texts."""

        with torch.inference_mode():
            embeddings = self._model.encode(
                texts,
                batch_size=min(len(texts), 8),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return np.asarray(embeddings, dtype=np.float32)

    def similarity(self, left: str, right: str) -> float:
        """Compute cosine similarity from two generated embeddings."""

        embeddings = self.embed([left, right])
        return float(cosine_similarity(embeddings[0:1], embeddings[1:2])[0, 0])
