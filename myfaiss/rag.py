import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from myllm import MyLLM
llm = MyLLM()
# 1. 初始化
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("myfaiss/index.faiss")
with open("myfaiss/chunks.json") as f:
    chunks = json.load(f)



# 2. 多轮 memory
memory_store = {}  # session_id -> list of query/answer pairs

# 3. top-k + rerank
def retrieve(query, k=10, rerank_k=3):
    q_emb = model.encode([query])
    distances, indices = index.search(np.array(q_emb), k)
    top_chunks = [chunks[i] for i in indices[0]]

    # rerank 按 cosine similarity 精细排序
    q_norm = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    e_norm = np.array([model.encode(c["text"]) for c in top_chunks])
    e_norm = e_norm / np.linalg.norm(e_norm, axis=1, keepdims=True)
    sims = (q_norm @ e_norm.T)[0]
    sorted_idx = np.argsort(-sims)[:rerank_k]
    return [top_chunks[i] for i in sorted_idx]

def ask(session_id, query):
    top_chunks = retrieve(query)

    context = "\n\n".join([
        f"[Doc {c['doc_id']}] {c['text']}"
        for c in top_chunks
    ])

    # 追加多轮历史
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

    # 保存 memory
    memory_store.setdefault(session_id, []).append({"query": query, "answer": answer})
    return {
        "answer": answer,
        "docs": top_chunks
    }