# Day 19 — 并发 & vLLM batching 实验（吞吐 / cost）

## 实验设置

- 并发用户数：`50`
- 方法：`asyncio` 同时发起 50 个请求
- 三种后端（模拟）：
  1. `cloud_api`：每请求独立处理，含网络开销，按 token 计费
  2. `hf_single_process`：单进程串行处理（无 batching）
  3. `vllm_batching`：队列 + 动态批处理窗口（40ms）
- 请求分布：
  - prompt tokens: 305 ~ 1582（均值 1004.06）
  - output tokens: 121 ~ 320（均值 203.42）

## 实验结果（P50 / P95 / QPS / 成本）

| Backend | P50(ms) | P95(ms) | QPS | 成本(USD, total) | 成本(USD/1k req) |
|---|---:|---:|---:|---:|---:|
| cloud_api | 1413.75 | 1652.30 | 26.98 | 0.0136 | 0.2727 |
| hf_single_process | 37727.99 | 71066.93 | 0.68 | 0.0246 | 0.4913 |
| vllm_batching | 6256.11 | 9386.36 | 4.93 | 0.0038 | 0.0761 |

## 1 页结论

### 1) 延迟（Latency）

- **最好：cloud_api**（P50/P95 最低）。
- `vllm_batching` 明显优于串行 HF，但因为要等 batch window + 组批执行，尾延迟高于 cloud_api。
- `hf_single_process` 在 50 并发下出现严重排队，P95 很差。

### 2) 吞吐（QPS）

- **最好：cloud_api（26.98 QPS）**。
- `vllm_batching`（4.93 QPS）> `hf_single_process`（0.68 QPS），说明 batching 对吞吐提升明显。

### 3) 成本（Cost）

- **最好：vllm_batching（0.0761 USD/1k req）**。
- cloud_api 成本中等（0.2727 USD/1k req）。
- hf_single_process 最差（0.4913 USD/1k req），主要因为吞吐低导致单位请求分摊成本高。

### 4) 选型建议

- 如果目标是 **最低延迟**：优先 cloud_api。
- 如果目标是 **最低单位成本**：优先 vLLM batching。
- 单进程 HF 仅适合低并发开发验证，不适合 50 并发生产场景。
