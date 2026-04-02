"""Build a FAISS index from the 50 docs in data/docs.

Usage:
    python -m myfaiss.create_index
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

DOC_DIR = Path("data/docs")
INDEX_PATH = Path("myfaiss/index.faiss")
CHUNKS_PATH = Path("myfaiss/chunks.json")
MODEL_NAME = "all-MiniLM-L6-v2"

def get_encoder():
    return tiktoken.get_encoding("cl100k_base")


def load_docs(doc_dir: Path = DOC_DIR) -> list[dict[str, str]]:
    """Load docs sorted by file name for deterministic indexing."""
    docs: list[dict[str, str]] = []
    for file_path in sorted(doc_dir.glob("*.txt")):
        docs.append({"doc_id": file_path.name, "text": file_path.read_text(encoding="utf-8")})
    return docs


def chunk_by_token(text: str, chunk_size: int = 120, overlap: int = 30) -> list[str]:
    """Split text by token count using cl100k_base."""
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    enc = get_encoder()
    tokens = enc.encode(text)
    chunks: list[str] = []
    step = chunk_size - overlap

    for i in range(0, len(tokens), step):
        chunk_tokens = tokens[i : i + chunk_size]
        if not chunk_tokens:
            continue
        chunks.append(enc.decode(chunk_tokens).strip())

    return [chunk for chunk in chunks if chunk]


def build_chunk_records(docs: list[dict[str, str]]) -> list[dict[str, str | int]]:
    all_chunks: list[dict[str, str | int]] = []
    for doc in docs:
        chunks = chunk_by_token(doc["text"])
        for idx, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "doc_id": doc["doc_id"],
                    "chunk_id": idx,
                    "text": chunk,
                }
            )
    return all_chunks


def build_index() -> tuple[int, int]:
    docs = load_docs()
    if len(docs) != 50:
        raise ValueError(f"Expected 50 docs in {DOC_DIR}, got {len(docs)}")

    all_chunks = build_chunk_records(docs)
    model = SentenceTransformer(MODEL_NAME)

    texts = [str(c["text"]) for c in all_chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    dim = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    ids = np.arange(len(all_chunks), dtype=np.int64)
    index.add_with_ids(embeddings, ids)

    faiss.write_index(index, str(INDEX_PATH))
    CHUNKS_PATH.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")

    return len(all_chunks), dim


if __name__ == "__main__":
    chunk_count, dim = build_index()
    print(f"Index built: {chunk_count} chunks, dimension {dim}")
