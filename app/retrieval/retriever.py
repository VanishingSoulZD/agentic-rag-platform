from __future__ import annotations

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
ARTIFACT_DIR = Path("app/retrieval/artifacts")
INDEX_PATH = ARTIFACT_DIR / "index.faiss"
CHUNKS_PATH = ARTIFACT_DIR / "chunks.json"


@lru_cache(maxsize=1)
def _get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def _load_index_and_chunks() -> tuple[faiss.Index, list[dict[str, Any]]]:
    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Retrieval artifacts not found. Run `python -m app.retrieval.build_index` first."
        )

    index = faiss.read_index(str(INDEX_PATH))
    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    return index, chunks


def _encode(texts: list[str]) -> np.ndarray:
    model = _get_embedding_model()
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


def retrieve(query: str, k: int = 12, rerank_k: int = 6, top_docs: int = 3) -> list[dict[str, Any]]:
    index, chunks = _load_index_and_chunks()

    q_emb = _encode([query])
    faiss_scores, indices = index.search(q_emb, k)
    candidates = [chunks[i] for i in indices[0] if i >= 0]

    if not candidates:
        return []

    candidate_embs = _encode([candidate["text"] for candidate in candidates])
    rerank_scores = (q_emb @ candidate_embs.T)[0]

    reranked_chunks = sorted(
        [
            {
                **candidate,
                "faiss_score": float(faiss_scores[0][idx]),
                "rerank_score": float(rerank_scores[idx]),
            }
            for idx, candidate in enumerate(candidates)
        ],
        key=lambda item: item["rerank_score"],
        reverse=True,
    )[:rerank_k]

    per_doc: dict[str, dict[str, Any]] = defaultdict(dict)
    for item in reranked_chunks:
        current = per_doc.get(item["doc_id"])
        if not current or item["rerank_score"] > current["rerank_score"]:
            per_doc[item["doc_id"]] = item

    return sorted(per_doc.values(), key=lambda item: item["rerank_score"], reverse=True)[:top_docs]


def rag_search(query: str, k: int = 5) -> dict[str, Any]:
    """Pure retrieval for RAG pipeline: retrieve + rerank, no LLM generation."""
    docs = retrieve(query=query, k=max(k * 2, 10), rerank_k=max(k, 5), top_docs=k)
    return {
        "query": query,
        "docs": docs,
        "doc_ids": [doc["doc_id"] for doc in docs],
    }
