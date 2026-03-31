# agentic-rag-platform

Agentic RAG Platform（智能代理检索增强平台）

## Quickstart（FastAPI Day 4）

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 启动 Redis（Docker）

```bash
docker run -d --name day4-redis -p 6379:6379 redis:7-alpine
```

> 若容器已存在，可先执行：`docker start day4-redis`

### 3) 启动 FastAPI 应用

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4) 验证聊天与会话历史（通过 `/chat` 自动存储）

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
- 参数错误返回统一格式：`{"error":"invalid_request","code":422}`。

## 测试

```bash
pytest tests/test_app_main.py
```

> 开发约定：后续新增接口/函数/方法时，必须同步补充对应单元测试。
