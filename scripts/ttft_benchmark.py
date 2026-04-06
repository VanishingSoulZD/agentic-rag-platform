#!/usr/bin/env python3
"""benchmark on real cloud API (OpenAI-compatible).

- Measures TTFT and decode speed in token/s for short vs long prompts
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from dataclasses import asdict, dataclass

import tiktoken
from openai import AsyncOpenAI, BadRequestError

SHORT_PROMPT = "请用两句话解释 Prefill 和 Decode 的区别。"
LONG_PROMPT = (
        "你是推理系统课程助教。请阅读下面背景并最终只输出三行总结。\n"
        + "背景："
        + "在大模型推理里，Prefill 阶段会把整段输入一次性编码并写入 KV Cache；" * 60
        + "要求：用中文，尽量简洁。"
)


@dataclass
class RunMetric:
    prompt_type: str
    run_id: int
    ttft_ms: float
    gen_seconds: float
    token_per_sec: float
    output_tokens: int


def _encoding_for_model(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


async def _create_stream(client: AsyncOpenAI, model: str, prompt: str, with_usage: bool):
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0,
    )
    if with_usage:
        kwargs["stream_options"] = {"include_usage": True}
    return await client.chat.completions.create(**kwargs)


async def measure_once(client: AsyncOpenAI, model: str, prompt: str, run_id: int, prompt_type: str) -> RunMetric:
    start = time.perf_counter()
    first_token_at = None
    output_text_parts: list[str] = []
    usage_completion_tokens: int | None = None

    try:
        stream = await _create_stream(client, model, prompt, with_usage=True)
    except BadRequestError:
        stream = await _create_stream(client, model, prompt, with_usage=False)

    async for chunk in stream:
        if getattr(chunk, "usage", None) and chunk.usage.completion_tokens is not None:
            usage_completion_tokens = chunk.usage.completion_tokens

        if chunk.choices:
            delta = chunk.choices[0].delta.content
            if delta:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                output_text_parts.append(delta)

    end = time.perf_counter()

    if first_token_at is None:
        raise RuntimeError("No token received from provider stream")

    output_text = "".join(output_text_parts)
    output_tokens = usage_completion_tokens
    if output_tokens is None:
        enc = _encoding_for_model(model)
        output_tokens = len(enc.encode(output_text))

    ttft_ms = (first_token_at - start) * 1000
    gen_seconds = max(end - first_token_at, 1e-9)
    token_per_sec = output_tokens / gen_seconds
    return RunMetric(prompt_type, run_id, ttft_ms, gen_seconds, token_per_sec, output_tokens)


def summarize(metrics: list[RunMetric]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[RunMetric]] = {}
    for m in metrics:
        grouped.setdefault(m.prompt_type, []).append(m)

    out: dict[str, dict[str, float]] = {}
    for key, vals in grouped.items():
        ttfts = [v.ttft_ms for v in vals]
        tps = [v.token_per_sec for v in vals]
        out[key] = {
            "runs": len(vals),
            "avg_ttft_ms": round(statistics.mean(ttfts), 2),
            "p50_ttft_ms": round(statistics.median(ttfts), 2),
            "avg_token_per_sec": round(statistics.mean(tps), 2),
            "p50_token_per_sec": round(statistics.median(tps), 2),
        }
    return out


def _build_client() -> tuple[AsyncOpenAI, str, str | None]:
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_API_BASE")
    model = os.getenv("OPENAI_MODEL")
    mock_llm = (os.getenv("MOCK_LLM") or "").lower() in {"1", "true", "yes"}

    if mock_llm:
        raise RuntimeError("MOCK_LLM=true，当前脚本要求真实云 API，请关闭 mock 后重试。")
    if not api_key:
        raise RuntimeError("未找到 OPENAI_API_KEY（请在 .env 配置）")
    if not model:
        raise RuntimeError("未找到 OPENAI_MODEL（请在 .env 配置）")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return client, model, base_url


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--out", default="reports/ttft_results.json")
    args = parser.parse_args()

    client, model, base_url = _build_client()
    metrics: list[RunMetric] = []

    for i in range(1, args.runs + 1):
        metrics.append(await measure_once(client, model, SHORT_PROMPT, i, "short"))
        await asyncio.sleep(10)
        metrics.append(await measure_once(client, model, LONG_PROMPT, i, "long"))
        await asyncio.sleep(10)

    result = {
        "provider": "openai_compatible_cloud_api",
        "base_url": base_url,
        "openai_model": model,
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
