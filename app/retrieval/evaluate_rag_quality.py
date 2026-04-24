"""RAG quality evaluation (retrieval + answer quality + baseline).

Run:
    python -m app.retrieval.evaluate_rag_quality
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.llm_client import AsyncLLMClient
from app.retrieval.retriever import rag_search
from app.utils import ensure_output_parent

DOC_DIR = Path("data/docs")
GOLD_PATH = Path("app/retrieval/eval/gold_qa.json")
REPORT_JSON = Path("reports/rag_eval_report.json")
REPORT_MD = Path("reports/rag_eval_report.md")


@dataclass
class EvalItem:
    question: str
    expected_doc: str
    gold_answer: str


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text)


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def _f1(pred: str, gold: str) -> float:
    """计算预测答案与标准答案的 token 级 F1 分数。

    这里不是按“句子是否完全相等”来打分，而是把文本先标准化并切词，
    再比较两个 token 列表的重叠程度：
    - Precision：预测 token 里有多少比例是“命中”的；
    - Recall：标准答案 token 里有多少比例被预测覆盖；
    - F1：Precision 与 Recall 的调和平均。

    这样做比 exact match 更宽松，能容忍同义表达、语序变化或部分冗余。
    """
    p_tokens = _tokenize(pred)
    g_tokens = _tokenize(gold)
    # 任何一侧为空都无法形成有效重叠，直接记 0 分。
    if not p_tokens or not g_tokens:
        return 0.0

    # 统计预测答案中每个 token 的出现次数（multiset / 词频袋模型）。
    # 之所以要计数而不是 set，是为了正确处理重复词。
    common = {}
    for token in p_tokens:
        common[token] = common.get(token, 0) + 1

    # 遍历 gold token，按“可消费词频”计算命中数 hit。
    # 命中一次就将对应词频减 1，避免一个预测 token 被重复匹配多次。
    hit = 0
    for token in g_tokens:
        if common.get(token, 0) > 0:
            hit += 1
            common[token] -= 1

    # 没有任何重叠词，Precision/Recall 都为 0，F1 也为 0。
    if hit == 0:
        return 0.0

    # precision = 命中词数 / 预测词总数
    # recall    = 命中词数 / 标准词总数
    precision = hit / len(p_tokens)
    recall = hit / len(g_tokens)
    # F1 = 2PR / (P+R)
    return 2 * precision * recall / (precision + recall)


def _load_gold() -> list[EvalItem]:
    rows = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    return [EvalItem(**row) for row in rows]


def _load_corpus() -> dict[str, str]:
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(DOC_DIR.glob("*.txt"))
    }


def _bm25_scores(
    query: str, corpus: dict[str, str], k1: float = 1.5, b: float = 0.75
) -> list[tuple[str, float]]:
    """为 query 计算语料中每个文档的 BM25 分数并降序返回。

    参数说明：
    - k1：控制词频饱和速度。越大，term frequency 增长带来的收益越明显。
    - b：控制文档长度归一化强度。0 表示不做长度归一化，1 表示完全归一化。

    返回值：
    - [(doc_id, score), ...]，按 score 从高到低排序。
    """
    # 将 corpus 转成列表，后续需要多次遍历。
    docs = list(corpus.items())
    # 预先分词，避免在打分循环里重复 tokenize。
    tokenized_docs = {doc_id: _tokenize(text) for doc_id, text in docs}
    q_tokens = _tokenize(query)

    # 文档长度（以 token 数量计）与平均文档长度 avgdl，用于 BM25 的长度归一化。
    doc_lens = {doc_id: len(tokens) for doc_id, tokens in tokenized_docs.items()}
    avgdl = sum(doc_lens.values()) / max(len(doc_lens), 1)

    # 计算 document frequency: 每个词出现在多少篇文档中（不是总出现次数）。
    # 因此这里对每篇文档使用 set(tokens) 去重。
    df: dict[str, int] = {}
    for tokens in tokenized_docs.values():
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1

    # N 为文档总数，后续用于 IDF 计算。
    N = len(tokenized_docs)
    scores: list[tuple[str, float]] = []

    for doc_id, tokens in tokenized_docs.items():
        # 计算当前文档的 term frequency。
        tf: dict[str, int] = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        score = 0.0
        dl = doc_lens[doc_id]
        for term in q_tokens:
            # 查询词不在文档中，对当前文档贡献为 0。
            if term not in tf:
                continue
            # IDF：词越“稀有”（df 小）权重越高；越“常见”权重越低。
            # 这里采用常见的平滑形式，避免极端值与除零问题。
            idf = math.log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
            term_tf = tf[term]
            # BM25 的 tf 饱和 + 长度归一化分母：
            # - term_tf 增大时，分数增益会逐步饱和（非线性）；
            # - 文档越长（dl/avgdl 越大），归一化后分数会被抑制。
            denom = term_tf + k1 * (1 - b + b * (dl / (avgdl or 1)))
            score += idf * (term_tf * (k1 + 1)) / (denom or 1)

        scores.append((doc_id, score))

    # 返回按分数降序排列的结果，便于直接截取 top-k。
    return sorted(scores, key=lambda x: x[1], reverse=True)


async def _generate_answer(
    question: str, docs: list[dict[str, Any]], llm_client: AsyncLLMClient
) -> str:
    context = "\n\n".join([f"[{d['doc_id']}] {d['text']}" for d in docs])
    messages = [
        {
            "role": "system",
            "content": (
                "Answer only from retrieved context. "
                "If the answer is not present, say: I don't know."
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    result = await llm_client.chat(messages)

    # In mock mode, fallback to extractive answer for meaningful offline evaluation.
    if result.mock:
        return docs[0]["text"] if docs else "I don't know."
    return result.answer


async def evaluate(k: int = 3) -> dict[str, Any]:
    gold = _load_gold()
    corpus = _load_corpus()
    llm_client = AsyncLLMClient()

    emb_retrieval_hits = 0
    emb_answer_hits = 0
    bm25_retrieval_hits = 0
    embedding_available = True

    samples: list[dict[str, Any]] = []

    for item in gold:
        bm25_ranked = _bm25_scores(item.question, corpus)
        bm25_doc_ids = [doc_id for doc_id, _ in bm25_ranked[:k]]

        try:
            emb_result = rag_search(item.question, k=k)
            emb_doc_ids = emb_result["doc_ids"]
            emb_docs = emb_result["docs"]
        except Exception:
            embedding_available = False
            emb_doc_ids = bm25_doc_ids
            emb_docs = [
                {"doc_id": doc_id, "text": corpus.get(doc_id, "")}
                for doc_id in bm25_doc_ids
            ]

        emb_retrieval_ok = item.expected_doc in emb_doc_ids
        bm25_retrieval_ok = item.expected_doc in bm25_doc_ids

        predicted_answer = await _generate_answer(item.question, emb_docs, llm_client)
        answer_ok = _f1(predicted_answer, item.gold_answer) >= 0.6

        emb_retrieval_hits += int(emb_retrieval_ok)
        bm25_retrieval_hits += int(bm25_retrieval_ok)
        emb_answer_hits += int(answer_ok)

        samples.append({
            "question": item.question,
            "expected_doc": item.expected_doc,
            "embedding_topk": emb_doc_ids,
            "bm25_topk": bm25_doc_ids,
            "retrieval_ok": emb_retrieval_ok,
            "bm25_retrieval_ok": bm25_retrieval_ok,
            "answer_ok": answer_ok,
        })

    total = len(gold)
    report = {
        "k": k,
        "total": total,
        "embedding_pipeline_available": embedding_available,
        "retrieval_precision": emb_retrieval_hits / total,
        "answer_accuracy": emb_answer_hits / total,
        "bm25_retrieval_precision": bm25_retrieval_hits / total,
        "notes": [
            "Retrieval precision is Hit@k using expected_doc in top-k.",
            "Answer accuracy uses token-F1>=0.6 against gold answers.",
            "In MOCK_LLM mode, answer generation falls back to top-1 extractive text.",
            "If embedding artifacts are missing, evaluation falls back to BM25 retrieval.",
        ],
        "samples": samples,
    }
    return report


def _build_suggestions(report: dict[str, Any]) -> list[str]:
    suggestions = []
    if report["retrieval_precision"] < 0.8:
        suggestions.append(
            "提高 chunk 语义完整性（更小 overlap + 动态 chunk size）并调参 top-k/rerank-k。"
        )
    if report["bm25_retrieval_precision"] > report["retrieval_precision"]:
        suggestions.append(
            "引入 Hybrid Retrieval（BM25 + Embedding 融合）替代单路向量检索。"
        )
    if report["answer_accuracy"] < 0.75:
        suggestions.append("优化 prompt：增加引用约束、拒答策略和 structured output。")
    if not suggestions:
        suggestions.append(
            "当前基线可用，下一步建议进行 hard-negative mining 提升长尾问题效果。"
        )
    return suggestions


def write_report(report: dict[str, Any]) -> None:
    ensure_output_parent(REPORT_JSON).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    suggestions = _build_suggestions(report)
    md = [
        "# RAG 质量评估报告",
        "",
        f"- retrieval_precision: **{report['retrieval_precision']:.2%}**",
        f"- answer_accuracy: **{report['answer_accuracy']:.2%}**",
        f"- bm25_retrieval_precision: **{report['bm25_retrieval_precision']:.2%}**",
        "",
        "## 改进建议",
    ]
    md.extend([f"- {s}" for s in suggestions])
    ensure_output_parent(REPORT_MD).write_text("\n".join(md), encoding="utf-8")


async def _main() -> None:
    report = await evaluate(k=3)
    write_report(report)
    print(
        json.dumps(
            {
                "retrieval_precision": report["retrieval_precision"],
                "answer_accuracy": report["answer_accuracy"],
                "bm25_retrieval_precision": report["bm25_retrieval_precision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Report written: {REPORT_MD}")


if __name__ == "__main__":
    asyncio.run(_main())
