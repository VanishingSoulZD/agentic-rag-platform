"""Build FAISS vector index from documents.

Usage:
    python -m app.retrieval.build_index
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

DOC_DIR = Path("data/docs")
ARTIFACT_DIR = Path("app/retrieval/artifacts")
INDEX_PATH = ARTIFACT_DIR / "index.faiss"
CHUNKS_PATH = ARTIFACT_DIR / "chunks.json"
MODEL_NAME = "all-MiniLM-L6-v2"


def get_encoder():
    return tiktoken.get_encoding("cl100k_base")


def load_docs(doc_dir: Path = DOC_DIR) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    for file_path in sorted(doc_dir.glob("*.txt")):
        docs.append(
            {"doc_id": file_path.name, "text": file_path.read_text(encoding="utf-8")}
        )
    return docs


def chunk_by_token(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    enc = get_encoder()
    tokens = enc.encode(text)
    step = chunk_size - overlap
    chunks: list[str] = []

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        if chunk_tokens:
            chunks.append(enc.decode(chunk_tokens).strip())

    return [chunk for chunk in chunks if chunk]


def build_chunk_records(docs: list[dict[str, str]]) -> list[dict[str, str | int]]:
    records: list[dict[str, str | int]] = []
    for doc in docs:
        for idx, chunk in enumerate(chunk_by_token(doc["text"])):
            records.append({"doc_id": doc["doc_id"], "chunk_id": idx, "text": chunk})
    return records


def build_index() -> tuple[int, int, int]:
    docs = load_docs()
    chunks = build_chunk_records(docs)

    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        [str(chunk["text"]) for chunk in chunks],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    dim = vectors.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    ids = np.arange(len(chunks), dtype=np.int64)
    index.add_with_ids(vectors, ids)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return len(docs), len(chunks), dim


if __name__ == "__main__":
    doc_count, chunk_count, dim = build_index()
    print(f"Index built: docs={doc_count}, chunks={chunk_count}, dim={dim}")
