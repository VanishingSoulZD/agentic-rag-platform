"""Simple retrieval evaluation for acceptance.

Run:
    python -m app.retrieval.evaluate_retrieval
"""

from __future__ import annotations

from app.retrieval.retriever import retrieve

EVAL_SET = [
    {"query": "What is FAISS?", "expected_doc": "doc11.txt"},
    {"query": "What is semantic search?", "expected_doc": "doc22.txt"},
    {"query": "What is FastAPI good for?", "expected_doc": "doc1.txt"},
    {"query": "What is Docker used for?", "expected_doc": "doc3.txt"},
    {"query": "How does Redis store data?", "expected_doc": "doc2.txt"},
    {"query": "What is a knowledge graph?", "expected_doc": "doc49.txt"},
    {"query": "What is prompt engineering?", "expected_doc": "doc18.txt"},
    {"query": "What is Kubernetes for?", "expected_doc": "doc4.txt"},
    {"query": "What are sentence transformers used for?", "expected_doc": "doc21.txt"},
    {"query": "What is RAG in LLM systems?", "expected_doc": "doc7.txt"},
]


def evaluate() -> float:
    hit = 0
    for sample in EVAL_SET:
        pred = retrieve(sample["query"], k=12, rerank_k=6, top_docs=3)
        top3 = [r["doc_id"] for r in pred]
        ok = sample["expected_doc"] in top3
        hit += int(ok)
        print(f"Q: {sample['query']}")
        print(f"  expected: {sample['expected_doc']} | top3: {top3} | hit: {ok}")

    acc = hit / len(EVAL_SET)
    print(f"\nTop-3 accuracy: {acc:.2%} ({hit}/{len(EVAL_SET)})")
    return acc


if __name__ == "__main__":
    evaluate()
