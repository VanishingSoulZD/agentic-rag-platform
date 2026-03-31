# agentic-rag-platform

Agentic RAG Platform（智能代理检索增强平台）

## Quickstart（FastAPI Day 5）

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 启动 Redis（Docker）

```bash
docker run -d --name day5-redis -p 6379:6379 redis:7-alpine
```

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

### 5) 验证 `/chat`（自动写入 Redis 会话历史）

```bash
curl -i -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"hello","session_id":"s-1"}'
```

期望结果：
- 返回真实 LLM 回复（已配置 key）或 Mock 回复（未配置 key / `MOCK_LLM=true`）。
- 返回体中包含 `use` 字段（model、mock、prompt/completion/total token）。
- 同一 `session_id` 多轮请求可在 Redis 中累积历史（role/content 结构）。

## 测试

```bash
pytest tests/test_app_main.py
```

> 开发约定：后续新增接口/函数/方法时，必须同步补充对应单元测试。
