from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from myllm import MyLLM

llm = MyLLM()

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = Path("myfaiss/index.faiss")
CHUNKS_PATH = Path("myfaiss/chunks.json")

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(str(INDEX_PATH))
chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

memory_store: dict[str, list[dict[str, str]]] = {}


def _chunk_embeddings(texts: list[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)


def retrieve(query: str, k: int = 12, rerank_k: int = 6, top_docs: int = 3) -> list[dict]:
    """Retrieve candidates with FAISS top-k and rerank to doc-level top-3."""
    q_emb = _chunk_embeddings([query])
    scores, indices = index.search(q_emb, k)

    candidates = [chunks[i] for i in indices[0] if i >= 0]
    if not candidates:
        return []

    c_embs = _chunk_embeddings([c["text"] for c in candidates])
    rerank_scores = (q_emb @ c_embs.T)[0]

    reranked = sorted(
        [
            {
                **candidate,
                "faiss_score": float(scores[0][i]),
                "rerank_score": float(rerank_scores[i]),
            }
            for i, candidate in enumerate(candidates)
        ],
        key=lambda x: x["rerank_score"],
        reverse=True,
    )[:rerank_k]

    # aggregate to document-level ranking (best chunk wins)
    per_doc: dict[str, dict] = defaultdict(dict)
    for item in reranked:
        doc_id = item["doc_id"]
        current = per_doc.get(doc_id)
        if not current or item["rerank_score"] > current["rerank_score"]:
            per_doc[doc_id] = item

    doc_ranked = sorted(per_doc.values(), key=lambda x: x["rerank_score"], reverse=True)
    return doc_ranked[:top_docs]


def ask(session_id: str, query: str) -> dict:
    top_docs = retrieve(query)

    context = "\n\n".join([f"[Doc {d['doc_id']}] {d['text']}" for d in top_docs])

    history = memory_store.get(session_id, [])
    history_text = "\n".join([f"Q: {h['query']}\nA: {h['answer']}" for h in history])

    prompt = f"""
You are a QA assistant.

Answer the question using ONLY the provided context.

If the answer is NOT explicitly stated in the context, say:
"I don't know."

Do NOT make up information.
Do NOT use prior knowledge.

Cite sources like [doc3.txt] when possible.

Context:
{context}

Conversation history:
{history_text}

User question:
{query}
"""

    answer = llm.llm(prompt)
    memory_store.setdefault(session_id, []).append({"query": query, "answer": answer})
    return {"answer": answer, "docs": top_docs}


if __name__ == "__main__":
    demo_queries = [
        "What does FAISS do?",
        "How does semantic search differ from keyword search?",
        "What is FastAPI used for?",
    ]
    for query in demo_queries:
        results = retrieve(query)
        print(f"\nQ: {query}")
        print([r["doc_id"] for r in results])
