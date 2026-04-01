# Agentic RAG Platform（智能代理检索增强平台）

## Quickstart

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
