# Agentic RAG Platform

一个可用于**面试演示 / 技术答辩 / 实战练习**的 Agentic RAG 项目：
- FastAPI 服务化（同步 + 流式 SSE）
- Redis 会话记忆
- 向量检索（Embedding + FAISS + 重排）
- Agent 工具调用（Planner / Executor + 可追踪执行图）
- 缓存、限流、观测（Prometheus 指标 + CSV 事件）

---

## 1. 项目价值（面试怎么讲）

你可以把这个项目定位成：

> “一个可生产演进的 RAG 后端骨架，不只做问答，还覆盖了工程化关键能力：可观测性、缓存策略、限流保护、Agent 工具链与可追踪执行过程。”

面试时建议强调 3 件事：
1. **完整链路**：从 query → rewrite（可选）→ retrieval → prompt 组装 → LLM 输出。  
2. **工程能力**：统一错误处理、会话记忆、缓存、限流、metrics、压测脚本。  
3. **可解释性**：Agent trace（图结构 + 可视化页面），能解释“为什么得出这个答案”。

---

## 2. 核心能力速览

- **基础 API**
  - `GET /ping`：健康检查
  - `POST /chat`：多轮会话问答（含 Redis history）
  - `POST /chat/stream`：SSE 流式输出（含 token / usage 事件）
- **RAG API**
  - `POST /rag/query`：检索增强问答（支持 query rewrite、top-k）
- **Agent API**
  - `POST /agent/trace`：执行 Planner-Executor 并返回 trace
  - `GET /agent/trace/{trace_id}`：读取 trace 数据
  - `GET /agent/trace/{trace_id}/view`：Mermaid HTML 可视化
- **观测与运维**
  - `GET /metrics`：Prometheus 文本指标
  - `reports/metrics_events.csv`：请求级事件落盘
  - Grafana 预置 Dashboard（TTFT、P95、Cache Hit 等）

---

## 3. Quickstart（Docker）

> 适合面试现场快速起服务，默认可用 Mock LLM。

### 3.1 一步启动

```bash
docker compose up --build
```

启动后：
- API: `http://127.0.0.1:8000`
- Prometheus: `http://127.0.0.1:9090`
- Grafana: `http://127.0.0.1:3000`（`admin/admin`）

### 3.2 停止并清理

```bash
docker compose down
```

---

## 4. Quickstart（本机）

> 适合本地开发调试、IDE 断点、快速改代码。

### 4.1 安装依赖

```bash
pip install -r requirements.txt
```

### 4.2 启动 Redis（Docker 方式）

```bash
docker run -d --name agentic-rag-platform-redis -p 6379:6379 redis:7
# 如果已存在：docker start agentic-rag-platform-redis
```

### 4.3 配置 LLM（真实 OpenAI 或 Mock）

```bash
# 真实 LLM（可选）
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="gpt-4.1-mini"
# export OPENAI_API_BASE="https://api.openai.com/v1"

# Mock 模式（无 key 时默认自动启用；也可显式开启）
export MOCK_LLM=true
```

### 4.4 启动 API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4.5 最小验证

```bash
curl -i http://127.0.0.1:8000/ping

curl -i -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"hello","session_id":"s-1"}'
```

---

## 5. 面试演示脚本（3–5 个，可直接照读）

下面给你准备了 **5 个脚本**，你可以按时间裁剪为 3 个。

---

### Script 1：3 分钟系统全览（架构 + 能力）

**目标**：先建立“你有全局视角”的印象。  
**话术模板**：

1. “这是一个 Agentic RAG 平台，不只是 RAG 检索，还覆盖 Agent trace、缓存、限流、指标。”
2. “入口是 FastAPI；会话记忆放 Redis；检索层是 Embedding + FAISS + rerank；生成层接 LLM/Mock。”
3. “我可以现场演示四类接口：`/chat`、`/chat/stream`、`/rag/query`、`/agent/trace`。”

**建议配图**：打开 Grafana Dashboard + `/agent/trace/{trace_id}/view` 页面。

---

### Script 2：5 分钟 API 基线演示（鲁棒性 + 多轮会话）

**目标**：证明接口设计和异常策略靠谱。  
**命令**：

```bash
# 1) health
curl -i http://127.0.0.1:8000/ping

# 2) 参数正确
curl -i -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"hello","session_id":"interview-s1"}'

# 3) 参数错误（缺 session_id）
curl -i -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"only-message"}'

# 4) 同 session 多轮
curl -i -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"what did I say before?","session_id":"interview-s1"}'
```

**你要强调**：
- 有统一错误格式：`invalid_request(422)` / `internal_server_error(500)`。
- history 结构化保存，支持多轮上下文。

---

### Script 3：6 分钟 RAG 链路演示（检索增强 + 可解释结果）

**目标**：证明不是“裸 LLM 问答”。

```bash
# 构建索引（首次）
python -m app.retrieval.build_index

# 跑检索验收
python -m app.retrieval.evaluate_retrieval

# 服务化 RAG 查询
curl -X POST http://127.0.0.1:8000/rag/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is FAISS?","session_id":"rag-demo-1","k":5,"rewrite_query":true}'
```

**你要强调**：
- 返回里有 `answer + sources/doc_ids`，可追溯证据来源。
- `rewrite_query` 体现对对话上下文的利用（不是单轮死检索）。

---

### Script 4：5 分钟性能与稳定性（缓存 + 指标 + 压测）

**目标**：体现工程化深度。

```bash
# 同 session 两次请求，观察 cache hit/miss
curl -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"message":"hello","session_id":"perf-s1"}'
curl -X POST http://127.0.0.1:8000/chat -H 'Content-Type: application/json' -d '{"message":"hello again","session_id":"perf-s1"}'

# 看 Prometheus 指标
curl http://127.0.0.1:8000/metrics

# 生成周报（P50/P95/P99、成功率、token）
python scripts/weekly_metrics_report.py \
  --input reports/metrics_events.csv \
  --output reports/weekly_metrics_report.csv
```

**你要强调**：
- 能从“功能可用”走到“可观测、可优化、可运维”。
- 可以量化优化效果（TTFT / P95 / cache hit rate）。

---

### Script 5：7 分钟 Agent 可解释性演示（Trace + 可视化）

**目标**：展示 Agent 工具调用链路，不再是黑盒。

```bash
# 生成一次 agent trace
curl -X POST http://127.0.0.1:8000/agent/trace \
  -H 'Content-Type: application/json' \
  -d '{"question":"帮我比较下缓存命中和未命中的延迟差异"}'
```

拿到 `trace_id` 后：

```bash
# 读取结构化图
curl http://127.0.0.1:8000/agent/trace/<trace_id>

# 浏览器打开可视化页面
# http://127.0.0.1:8000/agent/trace/<trace_id>/view
```

**你要强调**：
- Trace 可用于排障、回放、分析工具质量。
- 这比单纯“给答案”更接近真实生产要求。

---

## 6. 常用开发命令

```bash
# 单测
pytest

# 只跑主 API 用例
pytest tests/test_app_main.py

# RAG 质量评估
python -m app.retrieval.evaluate_rag_quality
```

输出文件（常见）：
- `reports/rag_eval_report.json`
- `reports/rag_eval_report.md`
- `reports/weekly_metrics_report.csv`

---

## 7. 项目结构（面试可用）

```text
app/
  main.py                     # API 路由、异常处理、middleware、指标埋点
  llm_client.py               # LLM 调用封装（真实/Mock）
  memory/chat_store.py        # Redis 会话记忆
  retrieval/                  # 索引构建、检索、评估
  optimization/               # 缓存、限流
  langchain_tools/            # 工具注册、Agent、执行图
scripts/                      # 压测与报表脚本
monitoring/                   # Prometheus + Grafana 配置
tests/                        # pytest 测试集
```

---

## 8. 面试问答速记（建议背）

**Q1：为什么要做 Agentic RAG，而不是纯 RAG？**  
A：纯 RAG 主要解决“检索+生成”，Agentic RAG 进一步解决“多工具规划与执行”，并且能追踪执行路径，利于可解释与排障。

**Q2：你怎么保证线上稳定性？**  
A：统一错误处理、会话限流、缓存分层、请求级 metrics、Prometheus + Grafana、周期性周报。

**Q3：如何证明优化有效？**  
A：用 TTFT/P95/命中率做量化指标，对比优化前后；报告文件可留痕。

---

## 9. License

[MIT](LICENSE)
