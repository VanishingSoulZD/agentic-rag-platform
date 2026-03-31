# build_index.py
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DOC_DIR = "../data/docs"

model = SentenceTransformer("all-MiniLM-L6-v2")

def load_docs():
    docs = []
    for file in os.listdir(DOC_DIR):
        with open(os.path.join(DOC_DIR, file)) as f:
            docs.append({"doc_id": file, "text": f.read()})
    return docs

# chunk + overlap
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

import tiktoken

enc = tiktoken.get_encoding("cl100k_base")

def chunk_by_token(text, chunk_size=200, overlap=50):
    tokens = enc.encode(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk_tokens = tokens[i:i+chunk_size]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)

    return chunks

# 文档切块 + metadata
docs = load_docs()
all_chunks = []
for doc in docs:
    chunks = chunk_by_token(doc["text"])
    for idx, chunk in enumerate(chunks):
        all_chunks.append({
            "doc_id": doc["doc_id"],
            "chunk_id": idx,
            "text": chunk
        })

# embedding
texts = [c["text"] for c in all_chunks]
embeddings = model.encode(texts, convert_to_numpy=True)

# FAISS index + IDMap
dim = embeddings.shape[1]
index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
ids = np.arange(len(all_chunks))
index.add_with_ids(embeddings, ids)

# 保存
faiss.write_index(index, "index.faiss")
with open("chunks.json", "w") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)

print(f"Index built: {len(all_chunks)} chunks, dimension {dim}")