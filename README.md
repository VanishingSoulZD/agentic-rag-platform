# Agentic RAG Platform（智能代理检索增强平台）

## Quickstart-Docker

### 一步启动

```bash
docker compose up --build
```

> 启动后 API 地址：`http://127.0.0.1:8000`

### 说明

- `docker-compose.yml` 同时启动两个服务：
    - `api`：FastAPI 应用
    - `redis`：会话记忆存储
- Compose 中默认设置 `MOCK_LLM=true`，无需 OpenAI Key 即可本地联调。
- 如需停止并删除容器：
  docker compose down

## Quickstart-本机

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 启动 Redis（Docker）

```bash
docker run -d --name agentic-rag-platform-redis -p 6379:6379 redis:7
```

> 若容器已存在，可先执行：`docker start agentic-rag-platform-redis`

### 3) 配置 LLM（真实 OpenAI SDK 或 Mock）

```bash
# 真实 LLM（可选）
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="gpt-4.1-mini"
# export OPENAI_API_BASE="https://api.openai.com/v1"  # 如需兼容网关可设置

# Mock 模式（无 key 时默认自动启用；也可显式开启）
export MOCK_LLM=true
```

### 4) 启动 FastAPI 应用

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5) 验证接口

```bash
# health check
curl -i http://127.0.0.1:8000/ping

# chat（需 message + session_id）
curl -i -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"hello","session_id":"s-1"}'

# 参数校验错误（缺少 session_id）
curl -i -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"only-message"}'

# 触发服务端错误（统一错误格式）
curl -i -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"raise_error","session_id":"s-1"}'
```

期望结果：

- 参数错误返回统一格式：`{"error":"invalid_request","code":422}`
- 服务错误返回统一格式：`{"error":"internal_server_error","code":500}`

### 6) 并发验证（5 个请求）

```bash
for i in 1 2 3; do
  curl -s -X POST http://127.0.0.1:8000/chat \
    -H 'Content-Type: application/json' \
    -d "{\"message\":\"hello-$i\",\"session_id\":\"s-$i\"}" &
done
wait
```

观察点：响应会并发返回，不会严格一个接一个串行等待。

### 7) 验证聊天与会话历史（自动写入 Redis 会话历史）

```bash
# 第一轮
curl -i -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"hello","session_id":"s-1"}'

# 第二轮（同一个 session）
curl -i -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"how are you","session_id":"s-1"}'
```

期望结果：

- `/chat` 返回 `history`，其中消息是 `{'role','content'}` 结构。
- 同一 `session_id` 多轮请求可在 Redis 中累积历史（role/content 结构）。
- 重启 FastAPI 服务后（不重启 Redis），再次请求同一 `session_id` 仍能读取之前历史。
- 返回真实 LLM 回复（已配置 key）或 Mock 回复（未配置 key / `MOCK_LLM=true`）。
- 返回体中包含 `use` 字段（model、mock、prompt/completion/total token）。

### 8) 流式接口（SSE）

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H 'Content-Type: application/json' \
  -d '{"message":"hello stream","session_id":"s-1"}'
```

期望结果：

- 首个 token 会尽快以 `data: {"type":"token",...}` 事件输出。
- 最终输出 usage 事件，包含 `prompt_tokens`、`completion_tokens`（以及 total/model/mock）。
- 同一 `session_id` 的消息历史会写入 Redis。

## 测试

```bash
pytest tests/test_app_main.py
```

> 开发约定：后续新增接口/函数/方法时，必须同步补充对应单元测试。

## Embeddings + FAISS 入门

```bash
# 1) 构建索引（50 篇文档 -> token chunk -> embedding -> FAISS）
python -m app.retrieval.build_index

# 2) 运行检索验收（10 条 query，检查 top-3 是否命中）
python -m app.retrieval.evaluate_retrieval
```

实现说明：

- Embedding 模型：`SentenceTransformer("all-MiniLM-L6-v2")`
- 分块方式：`tiktoken.get_encoding("cl100k_base")` + `chunk_by_token`
- 检索流程：`FAISS top-k` + `cosine rerank` + 文档级 top-3 聚合

## RAG pipeline（检索 + prompt 拼接）

```bash
curl -X POST http://127.0.0.1:8000/rag \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is FAISS?", "session_id":"rag-s1", "k":5, "rewrite_query": true}'
```

说明：

- `/rag` 流程：可选 Query Rewrite（基于 history）→ 检索/精排 → Prompt 组装（context+history）→ LLM 生成。
- 返回包含 `answer + sources(doc chunks) + doc_ids`，并在服务端打印检索到的 `doc_ids`。



## Day 12 监控埋点（TTFT/P95/usage）

新增能力：

- API 内置埋点：`response_time_ms`、`ttft_ms`、`prompt_tokens`、`completion_tokens`、`success_rate`。
- 暴露 `GET /metrics`（Prometheus 文本格式）。
- 同步写入请求级 CSV：`reports/metrics_events.csv`。
- 每周报告脚本（含 P50/P95/P99）：

```bash
python scripts/weekly_metrics_report.py   --input reports/metrics_events.csv   --output reports/weekly_metrics_report.csv
```

周报字段：`week_start, request_count, success_rate, latency_p50_ms, latency_p95_ms, latency_p99_ms, avg_ttft_ms, prompt_tokens_total, completion_tokens_total`。
