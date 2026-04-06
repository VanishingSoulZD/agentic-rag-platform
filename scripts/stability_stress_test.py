#!/usr/bin/env python3
"""并发稳定性与压力测试（100 并发）

This script simulates two versions of an inference service:
1) baseline (no backpressure) -> overload prone
2) improved (bounded queue + worker pool) -> stable under 100 concurrency

It outputs error rate, P95 RT, QPS, and CPU/GPU observations.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class StressMetrics:
    mode: str
    concurrency: int
    total_requests: int
    success: int
    errors: int
    error_rate: float
    p50_ms: float
    p95_ms: float
    qps: float
    success_qps: float
    wall_time_s: float
    cpu_usage_pct_est: float
    gpu_usage_note: str


class BaselineServer:
    """No queue, rejects immediately when in-flight exceeds soft limit."""

    def __init__(self, soft_limit: int = 24):
        self.soft_limit = soft_limit
        self.in_flight = 0
        self._lock = asyncio.Lock()

    async def handle(self) -> None:
        async with self._lock:
            self.in_flight += 1
            current = self.in_flight
        try:
            if current > self.soft_limit:
                raise RuntimeError("overloaded")
            service = random.uniform(0.05, 0.11)
            await asyncio.sleep(service)
        finally:
            async with self._lock:
                self.in_flight -= 1


class BackpressureServer:
    """Bounded queue + worker pool to smooth bursts."""

    def __init__(self, workers: int = 20, queue_size: int = 300):
        self.workers = workers
        self.queue: asyncio.Queue[tuple[asyncio.Future[None], float]] = asyncio.Queue(maxsize=queue_size)
        self.worker_tasks: list[asyncio.Task] = []
        self.stopping = False

    async def start(self) -> None:
        for _ in range(self.workers):
            self.worker_tasks.append(asyncio.create_task(self._worker()))

    async def stop(self) -> None:
        self.stopping = True
        for _ in range(self.workers):
            fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            await self.queue.put((fut, -1.0))
        await asyncio.gather(*self.worker_tasks)

    async def _worker(self) -> None:
        while True:
            fut, marker = await self.queue.get()
            if marker < 0:
                if not fut.done():
                    fut.set_result(None)
                self.queue.task_done()
                break

            # simulate inference service time
            await asyncio.sleep(random.uniform(0.05, 0.11))
            if not fut.done():
                fut.set_result(None)
            self.queue.task_done()

    async def handle(self) -> None:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()

        try:
            await asyncio.wait_for(self.queue.put((fut, time.perf_counter())), timeout=0.25)
        except TimeoutError:
            raise RuntimeError("queue_full")

        try:
            await asyncio.wait_for(fut, timeout=2.0)
        except TimeoutError:
            raise RuntimeError("processing_timeout")


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    arr = sorted(vals)
    k = (len(arr) - 1) * p
    f = int(k)
    c = min(f + 1, len(arr) - 1)
    if f == c:
        return arr[f]
    return arr[f] + (arr[c] - arr[f]) * (k - f)


async def run_stress(mode: str, server, concurrency: int, total_requests: int) -> StressMetrics:
    latencies: list[float] = []
    errors = 0
    cpu_start = time.process_time()
    wall_start = time.perf_counter()

    if isinstance(server, BackpressureServer):
        await server.start()

    req_index = 0
    req_lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal req_index, errors
        while True:
            async with req_lock:
                if req_index >= total_requests:
                    return
                req_index += 1
            t0 = time.perf_counter()
            try:
                await server.handle()
                latencies.append((time.perf_counter() - t0) * 1000)
            except Exception:
                errors += 1

    await asyncio.gather(*(worker() for _ in range(concurrency)))

    if isinstance(server, BackpressureServer):
        await server.stop()

    wall = time.perf_counter() - wall_start
    cpu = time.process_time() - cpu_start
    success = len(latencies)

    p50 = percentile(latencies, 0.5)
    p95 = percentile(latencies, 0.95)
    qps = total_requests / wall
    success_qps = success / wall
    err_rate = errors / total_requests
    cpu_pct_est = min(100.0, (cpu / wall) * 100)

    return StressMetrics(
        mode=mode,
        concurrency=concurrency,
        total_requests=total_requests,
        success=success,
        errors=errors,
        error_rate=round(err_rate, 4),
        p50_ms=round(p50, 2),
        p95_ms=round(p95, 2),
        qps=round(qps, 2),
        success_qps=round(success_qps, 2),
        wall_time_s=round(wall, 3),
        cpu_usage_pct_est=round(cpu_pct_est, 2),
        gpu_usage_note="N/A (CPU-only simulation)",
    )


async def main_async(args) -> None:
    random.seed(args.seed)

    baseline = await run_stress(
        mode="baseline_no_backpressure",
        server=BaselineServer(soft_limit=args.baseline_soft_limit),
        concurrency=args.concurrency,
        total_requests=args.total_requests,
    )

    improved = await run_stress(
        mode="improved_queue_backpressure",
        server=BackpressureServer(workers=args.workers, queue_size=args.queue_size),
        concurrency=args.concurrency,
        total_requests=args.total_requests,
    )

    threshold = {
        "target_error_rate_max": 0.02,
        "target_p95_ms_max": args.p95_threshold_ms,
    }

    result = {
        "threshold": threshold,
        "runs": [asdict(baseline), asdict(improved)],
        "verdict": {
            "baseline_pass": baseline.error_rate < threshold["target_error_rate_max"]
                             and baseline.p95_ms <= threshold["target_p95_ms_max"],
            "improved_pass": improved.error_rate < threshold["target_error_rate_max"]
                             and improved.p95_ms <= threshold["target_p95_ms_max"],
        },
        "delta": {
            "error_rate_drop": round(baseline.error_rate - improved.error_rate, 4),
            "p95_ms_drop": round(baseline.p95_ms - improved.p95_ms, 2),
            "success_qps_change": round(improved.success_qps - baseline.success_qps, 2),
        },
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--total-requests", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--queue-size", type=int, default=300)
    parser.add_argument("--baseline-soft-limit", type=int, default=24)
    parser.add_argument("--p95-threshold-ms", type=float, default=1500.0)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--out", default="reports/stability_stress_results.json")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
