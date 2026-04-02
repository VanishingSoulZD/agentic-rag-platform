# Day 8 — Prefill / Decode 原理 + vLLM 快速体验

## 1) 原理速记：为什么要分 Prefill 与 Decode

- **Prefill（首轮填充）**：模型一次性处理全部输入 prompt，完成注意力计算并把中间状态写入 **KV Cache**。这一步计算量随 prompt 长度明显增长，通常决定了 **TTFT（Time To First Token）**。
- **Decode（逐 token 生成）**：在已有 KV Cache 上，每次只新增 1 个 token，计算量主要和输出长度有关，通常看 **token/s**（或 char/s）。
- 直觉上：
  - prompt 越长，**Prefill 越重**，TTFT 越容易变大；
  - 输出越长，**Decode 总时长越长**，但稳态 token/s 往往更平稳。

---

## 2) 快速实验设计

本次在仓库内新增脚本 `scripts/day8_ttft_benchmark.py`，可对比短/长 prompt 的 TTFT 与后续速度。

- **短 prompt**：`请用两句话解释 Prefill 和 Decode 的区别。`
- **长 prompt**：拼接多段背景文本（约 900 字符级输入）。
- 默认模式 `llm_client`：直接调用 `AsyncLLMClient.stream_chat`（避免 Redis 依赖）；若服务可用，也支持 `--mode sse`。

运行命令：

```bash
python scripts/day8_ttft_benchmark.py --runs 3
```

结果保存在：`reports/day8_ttft_results.json`。

---

## 3) 实验结果（本次环境）

> 本次环境未配置 OpenAI Key，`MOCK_LLM=true`（mock 流式实现每字符 sleep 10ms）。因此数据主要用于**验证测量流程**，不代表真实 vLLM/GPU 性能上限。

| Prompt 类型 | Avg TTFT (ms) | P50 TTFT (ms) | Avg 速度 (char/s) | P50 速度 (char/s) |
|---|---:|---:|---:|---:|
| short | 0.04 | 0.03 | 96.87 | 96.79 |
| long | 0.03 | 0.03 | 97.58 | 97.56 |

---

## 4) 观察与小结（1 页）

1. **TTFT 在短/长 prompt 下几乎无差异**。这和真实大模型现象不同，根因是当前使用 mock：没有真实 Transformer Prefill 计算，首 token 几乎立即返回。
2. **后续速度约 97 char/s，且短长 prompt 接近**。这是因为 mock 流式按固定节奏输出字符（10ms/字符），所以速度基本稳定。
3. 这次结果的价值在于：
   - 验证了 TTFT / 生成速度的测量口径与脚本可复用；
   - 跑通了“短 prompt vs 长 prompt”的自动化对比流程；
   - 后续只需切换到真实 vLLM 或云 API，即可得到有意义的 Prefill/Decode 差异。

**下一步建议（真实 vLLM）**

- 在有 GPU 的机器上启动 vLLM（OpenAI 兼容接口）；
- 使用同一脚本改 `--mode sse`（或扩展为 `/v1/chat/completions` 流式）重复测量；
- 重点关注：
  - 长 prompt 的 TTFT 是否显著上升（Prefill 成本）；
  - 不同并发下 token/s 是否下降（调度与显存带宽限制）。

一句话总结：**Prefill 主要影响“首 token 等多久”，Decode 主要影响“后面吐字多快”；这次 mock 环境验证了方法，真实差异需在 vLLM/GPU 上复测。**
