from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.llm_client import AsyncLLMClient
from app.memory import ChatStoreConfig, HybridChatStore

MODEL_NAME = "all-MiniLM-L6-v2"
ARTIFACT_DIR = Path("app/retrieval/artifacts")
INDEX_PATH = ARTIFACT_DIR / "index.faiss"
CHUNKS_PATH = ARTIFACT_DIR / "chunks.json"

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(str(INDEX_PATH))
chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
llm_client = AsyncLLMClient()
chat_store = HybridChatStore(
    ChatStoreConfig(
        redis_url=os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0"),
        key_prefix="retrieval:chat:memory:",
        ttl_seconds=24 * 60 * 60,
    )
)


def _encode(texts: list[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


def retrieve(query: str, k: int = 12, rerank_k: int = 6, top_docs: int = 3) -> list[dict[str, Any]]:
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


async def ask(session_id: str, query: str) -> dict[str, Any]:
    top_docs = retrieve(query)
    context = "\n\n".join([f"[Doc {d['doc_id']}] {d['text']}" for d in top_docs])

    history = chat_store.get_memory(session_id)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a QA assistant. Use only provided context. "
                "If answer is not in context, respond with 'I don't know.'"
            ),
        },
        *history,
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    llm_result = await llm_client.chat(messages)
    chat_store.append_message(session_id, {"role": "user", "content": query})
    chat_store.append_message(session_id, {"role": "assistant", "content": llm_result.answer})

    return {
        "answer": llm_result.answer,
        "docs": top_docs,
        "use": {
            "model": llm_result.model,
            "mock": llm_result.mock,
            "prompt_tokens": llm_result.prompt_tokens,
            "completion_tokens": llm_result.completion_tokens,
            "total_tokens": llm_result.total_tokens,
        },
    }


def ask_sync(session_id: str, query: str) -> dict[str, Any]:
    return asyncio.run(ask(session_id=session_id, query=query))
