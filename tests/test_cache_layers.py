from __future__ import annotations

import numpy as np

from app.optimization.cache_layers import ResponseCache, ToolCache


class StubEmbeddingCache:
    def get(self, text: str) -> tuple[np.ndarray, bool]:
        return np.ones((4,), dtype=np.float32), False


def test_response_cache_evicts_exact_with_semantic_limit() -> None:
    cache = ResponseCache(embedding_cache=StubEmbeddingCache(), max_entries=2)

    cache.store("q1", {"v": 1})
    cache.store("q2", {"v": 2})
    cache.store("q3", {"v": 3})

    assert len(cache.semantic_entries) == 2
    assert len(cache.exact) == 2
    assert "q1" not in cache.exact


def test_response_cache_keeps_exact_for_replaced_key() -> None:
    cache = ResponseCache(embedding_cache=StubEmbeddingCache(), max_entries=2)

    cache.store("same", {"v": 1})
    cache.store("other", {"v": 2})
    cache.store("same", {"v": 3})

    assert len(cache.semantic_entries) == 2
    assert cache.exact["same"] == {"v": 3}


def test_tool_cache_evicts_exact_with_semantic_limit() -> None:
    cache = ToolCache(embedding_cache=StubEmbeddingCache(), max_entries=2)

    cache.store("search", "i1", "r1")
    cache.store("search", "i2", "r2")
    cache.store("search", "i3", "r3")

    assert len(cache.semantic_entries) == 2
    assert len(cache.exact) == 2
    assert "search::i1" not in cache.exact


def test_tool_cache_keeps_exact_for_replaced_key() -> None:
    cache = ToolCache(embedding_cache=StubEmbeddingCache(), max_entries=2)

    cache.store("search", "same", "r1")
    cache.store("search", "other", "r2")
    cache.store("search", "same", "r3")

    assert len(cache.semantic_entries) == 2
    assert cache.exact["search::same"] == "r3"
