import csv
from pathlib import Path

from app.metrics import MetricsStore


def test_metrics_store_migrates_legacy_csv_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics_events.csv"
    csv_path.write_text(
        "\n".join(
            [
                "ts,method,path,status_code,success,response_time_ms,ttft_ms,prompt_tokens,completion_tokens,cache_hit",
                "2026-04-01T00:00:00+00:00,POST,/chat,200,1,120.0,,10,5,1",
            ]
        ),
        encoding="utf-8",
    )

    store = MetricsStore(csv_path=str(csv_path))
    store.record_request(
        method="POST",
        path="/chat",
        status_code=200,
        response_time_ms=100.0,
        success=True,
        cache_hit=True,
        cache_layers={"response": "exact"},
    )

    with csv_path.open("r", encoding="utf-8", newline="") as fp:
        rows = list(csv.DictReader(fp))

    assert rows[0]["response_cache_hit"] == "0"
    assert rows[0]["retrieval_cache_hit"] == "0"
    assert rows[0]["embedding_cache_hit"] == "0"
    assert rows[0]["tool_cache_hit"] == "0"
    assert rows[1]["response_cache_hit"] == "1"
