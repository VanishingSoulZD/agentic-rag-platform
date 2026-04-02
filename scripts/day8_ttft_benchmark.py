#!/usr/bin/env python3
"""Day 8 benchmark: TTFT + decode speed (short vs long prompt).

默认直接调用 `app.llm_client.AsyncLLMClient`，不依赖 Redis。
如需测 FastAPI SSE，可加 `--mode sse`。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass

import httpx

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm_client import AsyncLLMClient

SHORT_PROMPT = "请用两句话解释 Prefill 和 Decode 的区别。"
LONG_PROMPT = (
    "你是推理系统课程助教。请阅读下面背景并最终只输出三行总结。\n"
    + "背景："
    + "在大模型推理里，Prefill 阶段会把整段输入一次性编码并写入 KV Cache；" * 20
    + "要求：用中文，尽量简洁。"
)


@dataclass
class RunMetric:
    prompt_type: str
    run_id: int
    ttft_ms: float
    gen_seconds: float
    chars_per_sec: float
    output_chars: int


async def measure_llm_client(prompt: str, run_id: int, prompt_type: str) -> RunMetric:
    client = AsyncLLMClient()
    messages = [{"role": "user", "content": prompt}]

    start = time.perf_counter()
    first_token_at = None
    out_chars = 0

    async for event in client.stream_chat(messages):
        if event.get("type") == "token":
            if first_token_at is None:
                first_token_at = time.perf_counter()
            out_chars += len(event.get("content", ""))
        elif event.get("type") == "usage":
            break

    end = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("No token received in llm_client mode")

    ttft = (first_token_at - start) * 1000
    gen_seconds = max(end - first_token_at, 1e-9)
    return RunMetric(prompt_type, run_id, ttft, gen_seconds, out_chars / gen_seconds, out_chars)


def measure_sse(base_url: str, prompt: str, run_id: int, prompt_type: str, session_prefix: str) -> RunMetric:
    url = f"{base_url.rstrip('/')}/chat/stream"
    payload = {"message": prompt, "session_id": f"{session_prefix}-{prompt_type}-{run_id}"}

    start = time.perf_counter()
    first_token_at = None
    out_chars = 0

    with httpx.Client(timeout=120.0) as hclient:
        with hclient.stream("POST", url, json=payload) as resp:
            resp.raise_for_status()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data: "):
                    continue
                data = json.loads(line[6:])
                if data.get("type") == "token":
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    out_chars += len(data.get("content", ""))
                elif data.get("type") == "usage":
                    break

    end = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("No token received in sse mode")

    ttft = (first_token_at - start) * 1000
    gen_seconds = max(end - first_token_at, 1e-9)
    return RunMetric(prompt_type, run_id, ttft, gen_seconds, out_chars / gen_seconds, out_chars)


def summarize(metrics: list[RunMetric]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RunMetric]] = {}
    for m in metrics:
        grouped.setdefault(m.prompt_type, []).append(m)

    out: dict[str, dict[str, float]] = {}
    for key, vals in grouped.items():
        ttfts = [v.ttft_ms for v in vals]
        speeds = [v.chars_per_sec for v in vals]
        out[key] = {
            "runs": len(vals),
            "avg_ttft_ms": round(statistics.mean(ttfts), 2),
            "p50_ttft_ms": round(statistics.median(ttfts), 2),
            "avg_chars_per_sec": round(statistics.mean(speeds), 2),
            "p50_chars_per_sec": round(statistics.median(speeds), 2),
        }
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["llm_client", "sse"], default="llm_client")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--session-prefix", default="day8")
    parser.add_argument("--out", default="reports/day8_ttft_results.json")
    args = parser.parse_args()

    metrics: list[RunMetric] = []
    for i in range(1, args.runs + 1):
        if args.mode == "llm_client":
            metrics.append(await measure_llm_client(SHORT_PROMPT, i, "short"))
            metrics.append(await measure_llm_client(LONG_PROMPT, i, "long"))
        else:
            metrics.append(measure_sse(args.base_url, SHORT_PROMPT, i, "short", args.session_prefix))
            metrics.append(measure_sse(args.base_url, LONG_PROMPT, i, "long", args.session_prefix))

    result = {
        "mode": args.mode,
        "openai_model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "mock_mode": os.getenv("MOCK_LLM", "") or ("true" if not os.getenv("OPENAI_API_KEY") else "false"),
        "runs_per_prompt": args.runs,
        "metrics": [asdict(m) for m in metrics],
        "summary": summarize(metrics),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
