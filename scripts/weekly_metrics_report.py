import argparse
import csv
import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    if len(arr) == 1:
        return float(arr[0])
    rank = (len(arr) - 1) * q
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(arr[lo])
    w = rank - lo
    return arr[lo] * (1 - w) + arr[hi] * w


def parse_week_start(ts: str) -> str:
    dt = datetime.fromisoformat(ts.replace('Z', '+00:00')).astimezone(timezone.utc)
    week_start = (dt - timedelta(days=dt.weekday())).date()
    return week_start.isoformat()


def generate_weekly_report(input_csv: Path, output_csv: Path) -> None:
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    with input_csv.open('r', encoding='utf-8', newline='') as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            if not row.get('ts'):
                continue
            buckets[parse_week_start(row['ts'])].append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow([
            'week_start',
            'request_count',
            'success_rate',
            'cache_hit_rate',
            'latency_p50_ms',
            'latency_p95_ms',
            'latency_p99_ms',
            'latency_cache_hit_p95_ms',
            'latency_cache_miss_p95_ms',
            'avg_ttft_ms',
            'prompt_tokens_total',
            'completion_tokens_total',
        ])

        for week_start in sorted(buckets):
            rows = buckets[week_start]
            latencies = [float(r.get('response_time_ms') or 0) for r in rows]
            ttfts = [float(r['ttft_ms']) for r in rows if r.get('ttft_ms')]
            success_count = sum(int(r.get('success') or 0) for r in rows)
            cache_hits = sum(int(r.get('cache_hit') or 0) for r in rows)
            hit_latencies = [float(r.get('response_time_ms') or 0) for r in rows if int(r.get('cache_hit') or 0) == 1]
            miss_latencies = [float(r.get('response_time_ms') or 0) for r in rows if int(r.get('cache_hit') or 0) == 0]
            prompt_tokens_total = sum(int(r.get('prompt_tokens') or 0) for r in rows)
            completion_tokens_total = sum(int(r.get('completion_tokens') or 0) for r in rows)
            count = len(rows)

            writer.writerow([
                week_start,
                count,
                f'{(success_count / count) if count else 0:.6f}',
                f'{(cache_hits / count) if count else 0:.6f}',
                f'{percentile(latencies, 0.5):.4f}',
                f'{percentile(latencies, 0.95):.4f}',
                f'{percentile(latencies, 0.99):.4f}',
                f'{percentile(hit_latencies, 0.95):.4f}',
                f'{percentile(miss_latencies, 0.95):.4f}',
                f'{(sum(ttfts) / len(ttfts)) if ttfts else 0:.4f}',
                prompt_tokens_total,
                completion_tokens_total,
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate weekly metrics report from request-level CSV logs.')
    parser.add_argument('--input', default='reports/metrics_events.csv', help='Input request-level CSV file path')
    parser.add_argument('--output', default='reports/weekly_metrics_report.csv', help='Output report CSV file path')
    args = parser.parse_args()

    generate_weekly_report(Path(args.input), Path(args.output))


if __name__ == '__main__':
    main()
