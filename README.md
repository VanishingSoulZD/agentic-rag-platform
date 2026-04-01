# agentic-rag-platform

Agentic RAG Platform（智能代理检索增强平台）

## Quickstart（FastAPI Day 6）

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 启动 Redis（Docker）

```bash
docker run -d --name day6-redis -p 6379:6379 redis:7-alpine
```

### 3) 配置 LLM（真实 OpenAI SDK 或 Mock）

```bash
# 真实 LLM（可选）
export OPENAI_API_KEY="your_api_key"
export OPENAI_MODEL="gpt-4.1-mini"
# export OPENAI_API_BASE="https://api.openai.com/v1"

# Mock 模式（无 key 时默认自动启用；也可显式开启）
export MOCK_LLM=true
```

### 4) 启动 FastAPI 应用

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5) 普通聊天接口

```bash
curl -i -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"hello","session_id":"s-1"}'
```

### 6) 流式接口（SSE）

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
