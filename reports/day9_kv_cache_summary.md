# Day 9 — KV Cache 实验（Prefill/Decode 时间分段）

## 实验目标

- 演示“先把历史写入 KV Cache，再进行生成”的流程。
- 记录并对比 **Prefill** 与 **Decode** 的时间分段。

## 实验方法

使用 `scripts/day9_kv_cache_experiment.py` 构建一个简化版 decoder attention：

1. **Prefill 阶段**
   - 输入长历史序列（`history_tokens=1024`）
   - 计算 `Q/K/V`，并完成一次全历史的因果注意力
   - 产出并保留 `K/V cache`
2. **Decode 阶段**
   - 逐 token 生成（`gen_tokens=64`）
   - 每步只计算当前 token 的 `q_t/k_t/v_t`
   - 复用历史 cache，执行 `q_t @ K_cache^T`

运行命令：

```bash
python scripts/day9_kv_cache_experiment.py --history-tokens 1024 --gen-tokens 64 --d-model 256
```

## 时间分段结果

本次实测结果（`reports/day9_kv_cache_results.json`）：

- Prefill: **0.0898s**
- Decode: **0.0236s**
- Total: **0.1134s**
- Prefill 占比: **79.17%**
- Decode 占比: **20.83%**

## 结论

- 已完成“历史先入 cache，再生成”的演示。
- 时间分段清晰显示：**Prefill 占比较大（约 79%）**，满足验收标准。
- 这也符合常见推理直觉：长上下文时 Prefill 成本高；Decode 单步较轻但会随输出长度累积。
