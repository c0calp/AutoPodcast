# autopodcast/embeddings.py

from __future__ import annotations
from typing import List
import numpy as np

from .config import CONFIG

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class EmbeddingModel:
    def __init__(self):
        self.cfg = CONFIG.embedding
        self._model: SentenceTransformer | None = None

    def load_model(self):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install -U sentence-transformers"
            )
        if self._model is None:
            self._model = SentenceTransformer(self.cfg.model_name)

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Return array of shape (n, dim)
        """
        if self._model is None:
            self.load_model()

        if not texts:
            return np.zeros((0, self.cfg.dim), dtype="float32")

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embeddings.astype("float32")
