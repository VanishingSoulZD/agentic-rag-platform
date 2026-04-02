# Prefill / Decode 原理 + 云 API 快速体验

## 1) 原理：Prefill vs Decode

- **Prefill**：一次性处理整段输入并构建 KV Cache，主要影响 **TTFT**（首 token 延迟）。
- **Decode**：基于已有 KV Cache 逐 token 生成，主要看稳态 **token/s**。
- 一般规律：长 prompt 会显著拉高 TTFT；而 decode 的 token/s 更多由模型规模、并发和服务端调度决定。

## 2) 实验方式

由于 Mac 本机通常不跑 vLLM，本次改为 **真实云 API（OpenAI 兼容）**：

- 禁止 mock：若 `MOCK_LLM=true`，脚本会直接报错退出
- 指标：
    - `TTFT`（毫秒）
    - `token/s`（`completion_tokens / decode耗时`）

运行：

```bash
python scripts/ttft_benchmark.py --runs 3
```

输出：`reports/ttft_results.json`

## 3) 结果记录

- `summary.short.avg_ttft_ms` vs `summary.long.avg_ttft_ms`
- `summary.short.avg_token_per_sec` vs `summary.long.avg_token_per_sec`

## 4) 观察小结

- 本任务的关键是把“**Prefill 看 TTFT**、**Decode 看 token/s**”分开测量。
- 在真实云 API 下，通常会看到：
    1. 长 prompt 的 TTFT 高于短 prompt（Prefill 成本上升）；
    2. token/s 在短长 prompt 间差异较小，但会受模型和负载影响。
- 若你后续换不同云模型（或不同 provider），同一脚本可直接横向比较，帮助选型与容量评估。
