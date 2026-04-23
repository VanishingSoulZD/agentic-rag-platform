#!/usr/bin/env python3
"""并发 & vLLM batching 实验（吞吐 / cost）

Use asyncio to simulate 50 concurrent users and compare three backends:
1) cloud API
2) single-process HF transformer
3) vLLM batching

Outputs P50 / P95 / QPS and rough cost estimation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class RequestSpec:
    id: int
    prompt_tokens: int
    output_tokens: int


@dataclass
class BackendMetrics:
    name: str
    p50_ms: float
    p95_ms: float
    qps: float
    wall_time_s: float
    cost_usd_total: float
    cost_usd_per_1k_req: float


class CloudAPISim:
    """Independent requests with network overhead and per-token billing."""

    in_price_per_m = 0.15
    out_price_per_m = 0.60

    async def infer(self, req: RequestSpec) -> float:
        base_net = random.uniform(0.18, 0.28)
        prefill = 0.0007 * req.prompt_tokens
        decode = 0.0018 * req.output_tokens
        jitter = random.uniform(0.0, 0.06)
        latency = base_net + prefill + decode + jitter
        await asyncio.sleep(latency)
        return latency

    def estimate_cost(
        self, requests: list[RequestSpec], wall_time_s: float
    ) -> tuple[float, float]:
        in_tokens = sum(r.prompt_tokens for r in requests)
        out_tokens = sum(r.output_tokens for r in requests)
        cost = (in_tokens / 1_000_000) * self.in_price_per_m + (
            out_tokens / 1_000_000
        ) * self.out_price_per_m
        cost_per_1k = cost / (len(requests) / 1000)
        return cost, cost_per_1k


class SingleProcessHFSim:
    """Single-worker model server (no batching), requests are serialized."""

    gpu_hourly_usd = 1.20

    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    async def infer(self, req: RequestSpec) -> float:
        async with self._lock:
            prefill = 0.0010 * req.prompt_tokens
            decode = 0.0021 * req.output_tokens
            kernel = 0.040
            latency = kernel + prefill + decode
            await asyncio.sleep(latency)
            return latency

    def estimate_cost(
        self, requests: list[RequestSpec], wall_time_s: float
    ) -> tuple[float, float]:
        cost = (wall_time_s / 3600) * self.gpu_hourly_usd
        cost_per_1k = cost / (len(requests) / 1000)
        return cost, cost_per_1k


class VLLMBatchingSim:
    """Queue + periodic scheduler to emulate dynamic batching."""

    gpu_hourly_usd = 1.35

    def __init__(self, batch_window_ms: int = 40, max_batch_size: int = 16) -> None:
        self.batch_window_s = batch_window_ms / 1000
        self.max_batch_size = max_batch_size
        self.queue: asyncio.Queue[tuple[RequestSpec, asyncio.Future[float], float]] = (
            asyncio.Queue()
        )
        self._runner_task: asyncio.Task | None = None
        self._stop = False

    async def start(self) -> None:
        self._runner_task = asyncio.create_task(self._runner())

    async def stop(self) -> None:
        self._stop = True
        if self._runner_task:
            await self._runner_task

    async def infer(self, req: RequestSpec) -> float:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[float] = loop.create_future()
        enqueue_at = time.perf_counter()
        await self.queue.put((req, fut, enqueue_at))
        return await fut

    async def _runner(self) -> None:
        while not self._stop or not self.queue.empty():
            if self.queue.empty():
                await asyncio.sleep(0.002)
                continue

            await asyncio.sleep(self.batch_window_s)
            batch: list[tuple[RequestSpec, asyncio.Future[float], float]] = []
            while len(batch) < self.max_batch_size and not self.queue.empty():
                batch.append(self.queue.get_nowait())

            reqs = [b[0] for b in batch]
            bs = len(reqs)
            max_prompt = max(r.prompt_tokens for r in reqs)
            avg_output = sum(r.output_tokens for r in reqs) / bs

            # dynamic batching effect: better per-token efficiency when batch is larger
            prefill = 0.0005 * max_prompt + 0.020
            decode = (0.0012 * avg_output * bs) * (0.55 + 0.45 / bs)
            engine_overhead = 0.012
            batch_latency = prefill + decode + engine_overhead

            await asyncio.sleep(batch_latency)
            now = time.perf_counter()
            for _, fut, enqueue_at in batch:
                if not fut.done():
                    fut.set_result(now - enqueue_at)

    def estimate_cost(
        self, requests: list[RequestSpec], wall_time_s: float
    ) -> tuple[float, float]:
        cost = (wall_time_s / 3600) * self.gpu_hourly_usd
        cost_per_1k = cost / (len(requests) / 1000)
        return cost, cost_per_1k


def percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return 0.0
    sorted_vals = sorted(vals)
    k = (len(sorted_vals) - 1) * pct
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


async def run_backend(
    name: str, backend, requests: list[RequestSpec]
) -> BackendMetrics:
    if isinstance(backend, VLLMBatchingSim):
        await backend.start()

    async def one(req: RequestSpec) -> float:
        t0 = time.perf_counter()
        await backend.infer(req)
        return time.perf_counter() - t0

    t_start = time.perf_counter()
    latencies = await asyncio.gather(*(one(r) for r in requests))
    wall_time = time.perf_counter() - t_start

    if isinstance(backend, VLLMBatchingSim):
        await backend.stop()

    p50 = percentile(latencies, 0.5) * 1000
    p95 = percentile(latencies, 0.95) * 1000
    qps = len(requests) / wall_time
    cost, cost_per_1k = backend.estimate_cost(requests, wall_time)

    return BackendMetrics(
        name=name,
        p50_ms=round(p50, 2),
        p95_ms=round(p95, 2),
        qps=round(qps, 2),
        wall_time_s=round(wall_time, 3),
        cost_usd_total=round(cost, 4),
        cost_usd_per_1k_req=round(cost_per_1k, 4),
    )


def make_requests(n: int, seed: int) -> list[RequestSpec]:
    rng = random.Random(seed)
    reqs = []
    for i in range(1, n + 1):
        prompt_tokens = rng.randint(300, 1600)
        output_tokens = rng.randint(120, 320)
        reqs.append(
            RequestSpec(id=i, prompt_tokens=prompt_tokens, output_tokens=output_tokens)
        )
    return reqs


async def main_async(args) -> None:
    requests = make_requests(args.users, args.seed)

    cloud = await run_backend("cloud_api", CloudAPISim(), requests)
    hf = await run_backend("hf_single_process", SingleProcessHFSim(), requests)
    vllm = await run_backend("vllm_batching", VLLMBatchingSim(), requests)

    result = {
        "users": args.users,
        "seed": args.seed,
        "request_mix": {
            "prompt_tokens_min": min(r.prompt_tokens for r in requests),
            "prompt_tokens_max": max(r.prompt_tokens for r in requests),
            "prompt_tokens_avg": round(
                statistics.mean(r.prompt_tokens for r in requests), 2
            ),
            "output_tokens_min": min(r.output_tokens for r in requests),
            "output_tokens_max": max(r.output_tokens for r in requests),
            "output_tokens_avg": round(
                statistics.mean(r.output_tokens for r in requests), 2
            ),
        },
        "metrics": [asdict(cloud), asdict(hf), asdict(vllm)],
    }

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=50)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--out", default="reports/concurrency_batching_results.json")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
