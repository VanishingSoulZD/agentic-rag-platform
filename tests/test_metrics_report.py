import csv
from pathlib import Path

from scripts.weekly_metrics_report import generate_weekly_report


def test_generate_weekly_report(tmp_path: Path) -> None:
    input_csv = tmp_path / 'metrics_events.csv'
    output_csv = tmp_path / 'weekly_report.csv'

    input_csv.write_text(
        '\n'.join([
            'ts,method,path,status_code,success,response_time_ms,ttft_ms,prompt_tokens,completion_tokens',
            '2026-03-30T08:00:00+00:00,POST,/chat,200,1,100.0,,10,20',
            '2026-03-31T08:00:00+00:00,POST,/chat/stream,200,1,220.0,60.0,15,30',
            '2026-04-01T08:00:00+00:00,POST,/chat,500,0,300.0,,5,5',
        ]),
        encoding='utf-8',
    )

    generate_weekly_report(input_csv, output_csv)

    with output_csv.open('r', encoding='utf-8', newline='') as fp:
        rows = list(csv.DictReader(fp))

    assert len(rows) == 1
    row = rows[0]
    assert row['week_start'] == '2026-03-30'
    assert row['request_count'] == '3'
    assert row['success_rate'] == '0.666667'
    assert float(row['latency_p50_ms']) > 0
    assert float(row['latency_p95_ms']) >= float(row['latency_p50_ms'])
    assert row['avg_ttft_ms'] == '60.0000'
    assert row['prompt_tokens_total'] == '30'
    assert row['completion_tokens_total'] == '55'
