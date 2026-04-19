#!/usr/bin/env python3
"""真实 API 验证 + 100 并发压测（Day21 + Day29）。"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EndpointCheck:
    endpoint: str
    method: str
    status_code: int
    ok: bool
    note: str = ""


@dataclass
class LoadMetrics:
    endpoint: str
    concurrency: int
    total_requests: int
    success: int
    errors: int
    error_rate: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    qps: float
    wall_time_s: float
    accepted: bool
    total_prompt_tokens: int
    total_completion_tokens: int
    estimated_cost_usd: float


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    arr = sorted(values)
    k = (len(arr) - 1) * p
    f = int(k)
    c = min(f + 1, len(arr) - 1)
    if f == c:
        return arr[f]
    return arr[f] + (arr[c] - arr[f]) * (k - f)


async def guarded_check(coro, fallback_endpoint: str, method: str) -> EndpointCheck:
    try:
        return await coro
    except Exception as exc:
        return EndpointCheck(fallback_endpoint, method, 0, False, note=f"exception={exc!r}")


async def check_ping(session: aiohttp.ClientSession, base_url: str) -> EndpointCheck:
    async with session.get(f"{base_url}/ping") as r:
        data = await r.json()
        ok = r.status == 200 and data.get("status") == "ok"
        return EndpointCheck("/ping", "GET", r.status, ok, note=f"body={data}")


async def check_chat(session: aiohttp.ClientSession, base_url: str) -> EndpointCheck:
    payload = {"message": "hello", "session_id": "api-check-chat"}
    async with session.post(f"{base_url}/chat", json=payload) as r:
        data = await r.json()
        ok = r.status == 200 and "answer" in data and "history" in data
        return EndpointCheck("/chat", "POST", r.status, ok, note=f"keys={list(data.keys())}")


async def check_chat_stream(session: aiohttp.ClientSession, base_url: str) -> EndpointCheck:
    payload = {"message": "stream hello", "session_id": "api-check-stream"}
    token_seen = False
    usage_seen = False
    done_seen = False

    async with session.post(f"{base_url}/chat/stream", json=payload) as r:
        if r.status != 200:
            text = await r.text()
            return EndpointCheck("/chat/stream", "POST", r.status, False, note=text[:200])

        async for raw in r.content:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                done_seen = True
                break
            try:
                ev = json.loads(body)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "token":
                token_seen = True
            if ev.get("type") == "usage":
                usage_seen = True

    ok = token_seen and usage_seen and done_seen
    return EndpointCheck(
        "/chat/stream",
        "POST",
        200,
        ok,
        note=f"token={token_seen},usage={usage_seen},done={done_seen}",
    )


async def check_rag_query(session: aiohttp.ClientSession, base_url: str) -> EndpointCheck:
    payload = {
        "query": "What is FAISS?",
        "session_id": "api-check-rag",
        "k": 3,
        "rewrite_query": True,
    }
    async with session.post(f"{base_url}/rag/query", json=payload) as r:
        data = await r.json()
        ok = r.status == 200 and "answer" in data and "sources" in data and "doc_ids" in data
        if ok:
            note = f"doc_ids={len(data.get('doc_ids', []))}"
        else:
            note = f"error={data.get('error')} code={data.get('code')}"
        return EndpointCheck("/rag/query", "POST", r.status, ok, note=note)


async def check_agent_trace_chain(session: aiohttp.ClientSession, base_url: str) -> list[EndpointCheck]:
    checks: list[EndpointCheck] = []

    payload = {"question": "请查 users 并给出 Alice 城市天气"}
    async with session.post(f"{base_url}/agent/trace", json=payload) as r:
        data = await r.json()
        ok = r.status == 200 and "trace_id" in data and "graph" in data
        checks.append(EndpointCheck("/agent/trace", "POST", r.status, ok, note=f"trace_id={data.get('trace_id')}"))
        if not ok:
            return checks
        trace_id = data["trace_id"]

    async with session.get(f"{base_url}/agent/trace/{trace_id}") as r:
        data = await r.json()
        ok = r.status == 200 and isinstance(data, dict)
        checks.append(EndpointCheck(f"/agent/trace/{trace_id}", "GET", r.status, ok, note=f"keys={list(data)[:4]}"))

    async with session.get(f"{base_url}/agent/trace/{trace_id}/view") as r:
        html = await r.text()
        ok = r.status == 200 and "mermaid" in html.lower()
        checks.append(EndpointCheck(f"/agent/trace/{trace_id}/view", "GET", r.status, ok, note=f"html_len={len(html)}"))

    return checks


async def run_chat_load(
    session: aiohttp.ClientSession,
    base_url: str,
    concurrency: int,
    total_requests: int,
    p95_threshold_ms: float,
    in_price_per_m: float,
    out_price_per_m: float,
) -> LoadMetrics:
    latencies_ms: list[float] = []
    prompt_tokens_total = 0
    completion_tokens_total = 0
    errors = 0
    request_idx = 0
    idx_lock = asyncio.Lock()
    token_lock = asyncio.Lock()

    async def worker(worker_id: int) -> None:
        nonlocal request_idx, errors, prompt_tokens_total, completion_tokens_total
        while True:
            async with idx_lock:
                if request_idx >= total_requests:
                    return
                request_idx += 1
                req_id = request_idx

            payload = {
                "message": f"load test message #{req_id}",
                "session_id": f"load-chat-{worker_id}",
            }
            t0 = time.perf_counter()
            try:
                async with session.post(f"{base_url}/chat", json=payload) as r:
                    body = await r.json()
                    if r.status != 200:
                        errors += 1
                        continue
                    use = body.get("use", {})
                    async with token_lock:
                        prompt_tokens_total += int(use.get("prompt_tokens", 0))
                        completion_tokens_total += int(use.get("completion_tokens", 0))
                    latencies_ms.append((time.perf_counter() - t0) * 1000)
            except Exception:
                errors += 1

    wall_start = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(concurrency)))
    wall = time.perf_counter() - wall_start

    success = len(latencies_ms)
    error_rate = errors / total_requests if total_requests else 0.0
    p50 = percentile(latencies_ms, 0.50)
    p95 = percentile(latencies_ms, 0.95)
    p99 = percentile(latencies_ms, 0.99)
    qps = total_requests / wall if wall > 0 else 0.0

    cost = (prompt_tokens_total / 1_000_000) * in_price_per_m + (completion_tokens_total / 1_000_000) * out_price_per_m
    accepted = error_rate < 0.02 and p95 <= p95_threshold_ms

    return LoadMetrics(
        endpoint="/chat",
        concurrency=concurrency,
        total_requests=total_requests,
        success=success,
        errors=errors,
        error_rate=round(error_rate, 4),
        p50_ms=round(p50, 2),
        p95_ms=round(p95, 2),
        p99_ms=round(p99, 2),
        qps=round(qps, 2),
        wall_time_s=round(wall, 3),
        accepted=accepted,
        total_prompt_tokens=prompt_tokens_total,
        total_completion_tokens=completion_tokens_total,
        estimated_cost_usd=round(cost, 6),
    )


def write_report(result: dict[str, Any], out_path: Path) -> None:
    checks: list[dict[str, Any]] = result["endpoint_checks"]
    load = result["load_test"]

    lines = [
        "# Performance Report (Real API)",
        "",
        "## 1) Endpoint Validation",
        "",
        "| Endpoint | Method | Status | OK | Note |",
        "|---|---|---:|---:|---|",
    ]
    for c in checks:
        lines.append(f"| {c['endpoint']} | {c['method']} | {c['status_code']} | {c['ok']} | {c['note']} |")

    lines.extend(
        [
            "",
            "## 2) 100-Concurrency Load Test (/chat)",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Concurrency | {load['concurrency']} |",
            f"| Total Requests | {load['total_requests']} |",
            f"| Success | {load['success']} |",
            f"| Errors | {load['errors']} |",
            f"| Error Rate | {load['error_rate']:.2%} |",
            f"| P50 (ms) | {load['p50_ms']} |",
            f"| P95 (ms) | {load['p95_ms']} |",
            f"| P99 (ms) | {load['p99_ms']} |",
            f"| QPS | {load['qps']} |",
            f"| Estimated Cost (USD) | {load['estimated_cost_usd']} |",
            "",
            "## 3) Acceptance",
            f"- Error rate < 2%: **{load['error_rate'] < 0.02}**",
            f"- P95 <= threshold: **{load['accepted']}**",
            "",
            "## 4) Notes",
            "- This report uses real HTTP calls to running FastAPI endpoints.",
            "- LLM may run in mock mode if OPENAI_API_KEY is not configured.",
            "- CPU/GPU metrics are not exposed by current API; add node/GPU telemetry for full observability.",
            "- Scope: Day21/Day29 acceptance now uses real API artifacts only (simulation artifacts removed).",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main_async(args: argparse.Namespace) -> None:
    timeout = aiohttp.ClientTimeout(total=args.timeout_s)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        checks: list[EndpointCheck] = []
        checks.append(await guarded_check(check_ping(session, args.base_url), "/ping", "GET"))
        agent_checks = await guarded_check(check_agent_trace_chain(session, args.base_url), "/agent/trace", "POST")
        if isinstance(agent_checks, EndpointCheck):
            checks.append(agent_checks)
        else:
            checks.extend(agent_checks)
        checks.append(await guarded_check(check_chat(session, args.base_url), "/chat", "POST"))
        checks.append(await guarded_check(check_chat_stream(session, args.base_url), "/chat/stream", "POST"))
        checks.append(await guarded_check(check_rag_query(session, args.base_url), "/rag/query", "POST"))

        load = await run_chat_load(
            session,
            args.base_url,
            args.concurrency,
            args.total_requests,
            args.p95_threshold_ms,
            args.in_price_per_m,
            args.out_price_per_m,
        )

    result = {
        "base_url": args.base_url,
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "endpoint_checks": [asdict(c) for c in checks],
        "load_test": asdict(load),
        "summary": {
            "all_endpoint_ok": all(c.ok for c in checks),
            "day21_day29_pass": load.accepted,
            "p95_threshold_ms": args.p95_threshold_ms,
            "artifact_scope": "real_api_only",
            "deprecated_simulation_artifacts_removed": True,
        },
    }

    out_json = ROOT / args.out_json
    out_md = ROOT / args.out_md
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(result, out_md)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved json: {args.out_json}")
    print(f"Saved report: {args.out_md}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--concurrency", type=int, default=100)
    parser.add_argument("--total-requests", type=int, default=1000)
    parser.add_argument("--p95-threshold-ms", type=float, default=1500.0)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--in-price-per-m", type=float, default=0.15)
    parser.add_argument("--out-price-per-m", type=float, default=0.60)
    parser.add_argument("--out-json", default="reports/performance_final_results.json")
    parser.add_argument("--out-md", default="reports/performance_report.md")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
