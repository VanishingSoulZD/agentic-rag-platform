from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


@dataclass
class SemanticEntry:
    key: str
    vector: np.ndarray
    payload: Any


class EmbeddingProvider:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._model_failed = False

    def _fallback_embed(self, text: str) -> np.ndarray:
        tokens = normalize_text(text).split()
        dim = 64
        vec = np.zeros((dim,), dtype=np.float32)
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = digest[0] % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        return vec if norm == 0 else vec / norm

    def embed(self, text: str) -> np.ndarray:
        if not self._model_failed:
            try:
                if self._model is None:
                    self._model = SentenceTransformer(self.model_name)
                vec = self._model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
                return vec.astype(np.float32)
            except Exception:
                self._model_failed = True
        return self._fallback_embed(text)


class EmbeddingCache:
    def __init__(self, provider: EmbeddingProvider, max_entries: int = 5000):
        self.provider = provider
        self.max_entries = max_entries
        self._cache: dict[str, np.ndarray] = {}
        self._order: list[str] = []
        self.lookups = 0
        self.hits = 0

    def get(self, text: str) -> tuple[np.ndarray, bool]:
        key = normalize_text(text)
        self.lookups += 1
        if key in self._cache:
            self.hits += 1
            return self._cache[key], True

        vec = self.provider.embed(text)
        self._cache[key] = vec
        self._order.append(key)
        if len(self._order) > self.max_entries:
            old = self._order.pop(0)
            self._cache.pop(old, None)
        return vec, False


class ResponseCache:
    def __init__(self, embedding_cache: EmbeddingCache, sim_threshold: float = 0.85, max_entries: int = 500):
        self.embedding_cache = embedding_cache
        self.sim_threshold = sim_threshold
        self.max_entries = max_entries
        self.exact: dict[str, dict[str, Any]] = {}
        self.semantic_entries: list[SemanticEntry] = []

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)

    def lookup(self, query: str) -> tuple[dict[str, Any] | None, float, str]:
        exact_key = normalize_text(query)
        if exact_key in self.exact:
            return self.exact[exact_key], 1.0, "exact"

        if not self.semantic_entries:
            return None, 0.0, "miss"

        q_vec, _ = self.embedding_cache.get(query)
        best_entry = None
        best_sim = -math.inf
        for entry in self.semantic_entries:
            sim = self._cosine(q_vec, entry.vector)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry and best_sim >= self.sim_threshold:
            return best_entry.payload, best_sim, "semantic"
        return None, (best_sim if best_sim > -math.inf else 0.0), "miss"

    def store(self, query: str, payload: dict[str, Any]) -> None:
        exact_key = normalize_text(query)
        self.exact[exact_key] = payload
        vec, _ = self.embedding_cache.get(query)
        self.semantic_entries.append(SemanticEntry(key=exact_key, vector=vec, payload=payload))
        while len(self.semantic_entries) > self.max_entries:
            evicted = self.semantic_entries.pop(0)
            if not any(entry.key == evicted.key for entry in self.semantic_entries):
                self.exact.pop(evicted.key, None)


class RetrievalCache:
    def __init__(self, embedding_cache: EmbeddingCache, sim_threshold: float = 0.9, max_entries: int = 500):
        self.embedding_cache = embedding_cache
        self.sim_threshold = sim_threshold
        self.max_entries = max_entries
        self.entries: list[SemanticEntry] = []

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)

    def lookup(self, query: str) -> tuple[dict[str, Any] | None, float, str]:
        if not self.entries:
            return None, 0.0, "miss"
        q_vec, _ = self.embedding_cache.get(query)
        best_entry = None
        best_sim = -math.inf
        for entry in self.entries:
            sim = self._cosine(q_vec, entry.vector)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
        if best_entry and best_sim >= self.sim_threshold:
            return best_entry.payload, best_sim, "semantic"
        return None, (best_sim if best_sim > -math.inf else 0.0), "miss"

    def store(self, query: str, payload: dict[str, Any]) -> None:
        vec, _ = self.embedding_cache.get(query)
        self.entries.append(SemanticEntry(key=normalize_text(query), vector=vec, payload=payload))
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]


class ToolCache:
    def __init__(self, embedding_cache: EmbeddingCache, sim_threshold: float = 0.92, max_entries: int = 500):
        self.embedding_cache = embedding_cache
        self.sim_threshold = sim_threshold
        self.max_entries = max_entries
        self.exact: dict[str, str] = {}
        self.semantic_entries: list[SemanticEntry] = []

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _tool_key(tool_name: str, tool_input: str) -> str:
        return f"{tool_name}::{normalize_text(tool_input)}"

    def lookup(self, tool_name: str, tool_input: str) -> tuple[str | None, float, str]:
        key = self._tool_key(tool_name, tool_input)
        if key in self.exact:
            return self.exact[key], 1.0, "exact"

        q_vec, _ = self.embedding_cache.get(key)
        best_entry = None
        best_sim = -math.inf
        for entry in self.semantic_entries:
            if not entry.key.startswith(f"{tool_name}::"):
                continue
            sim = self._cosine(q_vec, entry.vector)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
        if best_entry and best_sim >= self.sim_threshold:
            return str(best_entry.payload), best_sim, "semantic"
        return None, (best_sim if best_sim > -math.inf else 0.0), "miss"

    def store(self, tool_name: str, tool_input: str, result: str) -> None:
        key = self._tool_key(tool_name, tool_input)
        self.exact[key] = result
        vec, _ = self.embedding_cache.get(key)
        self.semantic_entries.append(SemanticEntry(key=key, vector=vec, payload=result))
        while len(self.semantic_entries) > self.max_entries:
            evicted = self.semantic_entries.pop(0)
            if not any(entry.key == evicted.key for entry in self.semantic_entries):
                self.exact.pop(evicted.key, None)


class CacheManager:
    def __init__(self):
        provider = EmbeddingProvider()
        self.embedding_cache = EmbeddingCache(provider=provider)
        self.response_cache = ResponseCache(embedding_cache=self.embedding_cache)
        self.retrieval_cache = RetrievalCache(embedding_cache=self.embedding_cache)
        self.tool_cache = ToolCache(embedding_cache=self.embedding_cache)

    def snapshot_metrics(self) -> dict[str, dict[str, int | float]]:
        return {
            "embedding": {
                "lookups": self.embedding_cache.lookups,
                "hits": self.embedding_cache.hits,
                "hit_rate": (self.embedding_cache.hits / self.embedding_cache.lookups)
                if self.embedding_cache.lookups
                else 0.0,
            },
            "response": {
                "exact_entries": len(self.response_cache.exact),
                "semantic_entries": len(self.response_cache.semantic_entries),
            },
            "retrieval": {"entries": len(self.retrieval_cache.entries)},
            "tool": {
                "exact_entries": len(self.tool_cache.exact),
                "semantic_entries": len(self.tool_cache.semantic_entries),
            },
        }
