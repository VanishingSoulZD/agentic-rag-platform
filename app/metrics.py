import csv
import math
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class RequestMetric:
    ts: str
    method: str
    path: str
    status_code: int
    success: int
    response_time_ms: float
    ttft_ms: float | None
    prompt_tokens: int
    completion_tokens: int


class MetricsStore:
    def __init__(self, csv_path: str | None = None):
        self._lock = threading.Lock()
        self._request_count = 0
        self._success_count = 0
        self._response_times: list[float] = []
        self._ttft_times: list[float] = []
        self._prompt_tokens_total = 0
        self._completion_tokens_total = 0

        self.csv_path = Path(csv_path or os.getenv('METRICS_CSV_PATH', 'reports/metrics_events.csv'))
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            self.csv_path.write_text(
                'ts,method,path,status_code,success,response_time_ms,ttft_ms,prompt_tokens,completion_tokens\n',
                encoding='utf-8',
            )

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return float(sorted_values[0])

        rank = (len(sorted_values) - 1) * percentile
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return float(sorted_values[lower])
        weight = rank - lower
        return float(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight)

    def record_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        response_time_ms: float,
        success: bool,
        ttft_ms: float | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        success_as_int = 1 if success else 0

        with self._lock:
            self._request_count += 1
            self._success_count += success_as_int
            self._response_times.append(response_time_ms)
            if ttft_ms is not None:
                self._ttft_times.append(ttft_ms)
            self._prompt_tokens_total += max(0, int(prompt_tokens))
            self._completion_tokens_total += max(0, int(completion_tokens))

            with self.csv_path.open('a', newline='', encoding='utf-8') as fp:
                writer = csv.writer(fp)
                writer.writerow(
                    [
                        ts,
                        method,
                        path,
                        status_code,
                        success_as_int,
                        f'{response_time_ms:.4f}',
                        '' if ttft_ms is None else f'{ttft_ms:.4f}',
                        int(prompt_tokens),
                        int(completion_tokens),
                    ]
                )

    def render_prometheus(self) -> str:
        with self._lock:
            success_rate = (self._success_count / self._request_count) if self._request_count else 0.0
            p50 = self._percentile(self._response_times, 0.5)
            p95 = self._percentile(self._response_times, 0.95)
            p99 = self._percentile(self._response_times, 0.99)
            ttft_p95 = self._percentile(self._ttft_times, 0.95)
            last_updated = int(time.time())

            lines = [
                '# HELP response_time_ms API response latency in milliseconds.',
                '# TYPE response_time_ms summary',
                f'response_time_ms{{quantile="0.5"}} {p50:.4f}',
                f'response_time_ms{{quantile="0.95"}} {p95:.4f}',
                f'response_time_ms{{quantile="0.99"}} {p99:.4f}',
                '# HELP ttft_ms Time to first token in milliseconds.',
                '# TYPE ttft_ms gauge',
                f'ttft_ms {ttft_p95:.4f}',
                '# HELP prompt_tokens Total prompt tokens.',
                '# TYPE prompt_tokens counter',
                f'prompt_tokens {self._prompt_tokens_total}',
                '# HELP completion_tokens Total completion tokens.',
                '# TYPE completion_tokens counter',
                f'completion_tokens {self._completion_tokens_total}',
                '# HELP success_rate Successful request ratio.',
                '# TYPE success_rate gauge',
                f'success_rate {success_rate:.6f}',
                '# HELP requests_total Number of API requests.',
                '# TYPE requests_total counter',
                f'requests_total {self._request_count}',
                '# HELP metrics_last_updated_unix Last updated unix timestamp.',
                '# TYPE metrics_last_updated_unix gauge',
                f'metrics_last_updated_unix {last_updated}',
            ]
        return '\n'.join(lines) + '\n'


metrics_store = MetricsStore()
