from app.retrieval.evaluate_rag_quality import _bm25_scores, _f1


def test_f1_returns_high_score_for_similar_answers():
    score = _f1(
        "FAISS is for efficient vector similarity search",
        "FAISS is an open-source library for efficient similarity search",
    )
    assert score > 0.5


def test_bm25_ranks_relevant_doc_first():
    corpus = {
        "doc_a": "redis is in memory data store for caching",
        "doc_b": "kubernetes manages containers at scale",
    }
    ranked = _bm25_scores("what is redis caching", corpus)
    assert ranked[0][0] == "doc_a"
