# Agentic RAG Platform（智能代理检索增强平台）

## Quickstart

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 启动 Redis（Docker）

```bash
docker run -d --name day4-redis -p 6379:6379 redis:7
```

> 若容器已存在，可先执行：`docker start day4-redis`

### 3) 启动 FastAPI 应用

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4) 验证接口

```bash
# health check
curl -i http://127.0.0.1:8000/ping

# chat（静态回复，需 message + session_id）
curl -i -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"你好","session_id":"s-1"}'

# 参数校验错误（缺少 session_id）
curl -i -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"hello"}'

# 触发服务端错误（统一错误格式）
curl -i -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"raise_error","session_id":"s-1"}'
```

期望结果：
- `GET /ping` 返回 `200` 和 JSON（如 `{"status":"ok"}`）
- `POST /chat` 返回 `200` 和 JSON（如 `{"answer":"这是一个静态回复","session_id":"s-1"}`）
- 参数错误返回统一格式：`{"error":"invalid_request","code":422}`
- 服务错误返回统一格式：`{"error":"internal_server_error","code":500}`

### 5) 并发验证（5 个请求）

```bash
for i in 1 2 3 4 5; do
  curl -s -X POST http://127.0.0.1:8000/chat \
    -H 'Content-Type: application/json' \
    -d "{\"message\":\"hello-$i\",\"session_id\":\"s-$i\"}" &
done
wait
```

观察点：响应会并发返回，不会严格一个接一个串行等待。

### 6) 验证聊天与会话历史（通过 `/chat` 自动存储）

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
- 同一 `session_id` 的多轮消息会持续累积（user/assistant 成对追加）。
- 重启 FastAPI 服务后（不重启 Redis），再次请求同一 `session_id` 仍能读取之前历史。

## 测试

```bash
pytest tests/test_app_main.py
```

> 开发约定：后续新增接口/函数/方法时，必须同步补充对应单元测试。
