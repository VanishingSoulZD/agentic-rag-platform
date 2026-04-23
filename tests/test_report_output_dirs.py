import json
from pathlib import Path

from app.retrieval import evaluate_rag_quality
from scripts.ttft_benchmark import write_result_file


def test_ttft_write_result_creates_parent_dir(tmp_path):
    output_path = tmp_path / "reports" / "ttft_results.json"
    assert not output_path.parent.exists()

    written = write_result_file({"summary": {"short": {"runs": 1}}}, output_path)

    assert written == output_path
    assert output_path.exists()
    assert json.loads(output_path.read_text(encoding="utf-8"))["summary"]["short"][
        "runs"
    ] == 1


def test_rag_write_report_creates_parent_dir(tmp_path, monkeypatch):
    report_dir = tmp_path / "reports"
    report_json = report_dir / "rag_eval_report.json"
    report_md = report_dir / "rag_eval_report.md"
    assert not report_dir.exists()

    monkeypatch.setattr(evaluate_rag_quality, "REPORT_JSON", report_json)
    monkeypatch.setattr(evaluate_rag_quality, "REPORT_MD", report_md)
    evaluate_rag_quality.write_report(
        {
            "retrieval_precision": 0.9,
            "answer_accuracy": 0.8,
            "bm25_retrieval_precision": 0.7,
        }
    )

    assert report_json.exists()
    assert report_md.exists()
