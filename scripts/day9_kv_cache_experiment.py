#!/usr/bin/env python3
"""Day 9 - KV Cache experiment with explicit prefill/decode timing segments.

This script builds a tiny decoder-only attention simulation and measures:
1) Prefill (process long history and build KV cache)
2) Decode (append one token at a time reusing cached K/V)

It outputs segmented timings and ratios to verify prefill is the major share.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Timing:
    prefill_s: float
    decode_s: float
    total_s: float
    prefill_ratio: float
    decode_ratio: float


@dataclass
class Result:
    device: str
    dtype: str
    d_model: int
    history_tokens: int
    gen_tokens: int
    timing: Timing


def causal_softmax(scores: torch.Tensor) -> torch.Tensor:
    # scores: [T, T]
    mask = torch.triu(torch.ones_like(scores), diagonal=1).bool()
    scores = scores.masked_fill(mask, float("-inf"))
    return torch.softmax(scores, dim=-1)


def run_experiment(history_tokens: int, gen_tokens: int, d_model: int, device: str) -> Result:
    dtype = torch.float32 if device == "cpu" else torch.float16
    torch.manual_seed(42)

    # Input history states and projection weights
    x_hist = torch.randn(history_tokens, d_model, device=device, dtype=dtype)
    wq = torch.randn(d_model, d_model, device=device, dtype=dtype) / math.sqrt(d_model)
    wk = torch.randn(d_model, d_model, device=device, dtype=dtype) / math.sqrt(d_model)
    wv = torch.randn(d_model, d_model, device=device, dtype=dtype) / math.sqrt(d_model)

    # --- Prefill: process entire history and build KV cache ---
    t0 = time.perf_counter()
    q_hist = x_hist @ wq
    k_cache = x_hist @ wk
    v_cache = x_hist @ wv

    # full-history causal attention (simulating prefill heavy cost)
    scores = (q_hist @ k_cache.transpose(0, 1)) / math.sqrt(d_model)
    probs = causal_softmax(scores)
    _ = probs @ v_cache
    t1 = time.perf_counter()

    # --- Decode: generate token-by-token while reusing cache ---
    t2 = time.perf_counter()
    for _ in range(gen_tokens):
        x_t = torch.randn(1, d_model, device=device, dtype=dtype)
        q_t = x_t @ wq
        k_t = x_t @ wk
        v_t = x_t @ wv

        k_cache = torch.cat([k_cache, k_t], dim=0)
        v_cache = torch.cat([v_cache, v_t], dim=0)

        step_scores = (q_t @ k_cache.transpose(0, 1)) / math.sqrt(d_model)
        step_probs = torch.softmax(step_scores, dim=-1)
        _ = step_probs @ v_cache
    t3 = time.perf_counter()

    prefill_s = t1 - t0
    decode_s = t3 - t2
    total_s = prefill_s + decode_s

    timing = Timing(
        prefill_s=round(prefill_s, 4),
        decode_s=round(decode_s, 4),
        total_s=round(total_s, 4),
        prefill_ratio=round(prefill_s / total_s, 4),
        decode_ratio=round(decode_s / total_s, 4),
    )

    return Result(
        device=device,
        dtype=str(dtype).replace("torch.", ""),
        d_model=d_model,
        history_tokens=history_tokens,
        gen_tokens=gen_tokens,
        timing=timing,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-tokens", type=int, default=1024)
    parser.add_argument("--gen-tokens", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--out", default="reports/day9_kv_cache_results.json")
    args = parser.parse_args()

    torch.set_num_threads(1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    result = run_experiment(
        history_tokens=args.history_tokens,
        gen_tokens=args.gen_tokens,
        d_model=args.d_model,
        device=device,
    )

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
