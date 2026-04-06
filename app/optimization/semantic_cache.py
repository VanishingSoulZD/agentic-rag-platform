from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class CacheEntry:
    query: str
    vector: np.ndarray
    payload: dict[str, Any]


class SemanticCache:
    def __init__(self, sim_threshold: float = 0.85, max_entries: int = 500):
        self.sim_threshold = sim_threshold
        self.max_entries = max_entries
        self.entries: list[CacheEntry] = []
        self.hits = 0
        self.lookups = 0
        self._model: SentenceTransformer | None = None
        self._model_failed = False

    def _fallback_embed(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        dim = 64
        vec = np.zeros((dim,), dtype=np.float32)
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = digest[0] % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    def _embed(self, text: str) -> np.ndarray:
        if not self._model_failed:
            try:
                if self._model is None:
                    self._model = SentenceTransformer("all-MiniLM-L6-v2")
                vec = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
                return vec.astype(np.float32)
            except Exception:
                self._model_failed = True
        return self._fallback_embed(text)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)

    def lookup(self, query: str) -> tuple[dict[str, Any] | None, float]:
        self.lookups += 1
        if not self.entries:
            return None, 0.0

        q_vec = self._embed(query)
        best_entry = None
        best_sim = -math.inf
        for entry in self.entries:
            sim = self._cosine(q_vec, entry.vector)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry and best_sim >= self.sim_threshold:
            self.hits += 1
            return best_entry.payload, best_sim

        return None, best_sim if best_sim > -math.inf else 0.0

    def store(self, query: str, payload: dict[str, Any]) -> None:
        vector = self._embed(query)
        self.entries.append(CacheEntry(query=query, vector=vector, payload=payload))
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0
