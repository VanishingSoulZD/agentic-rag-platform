# agentic-rag-platform

Agentic RAG Platform（智能代理检索增强平台）

## Quickstart（FastAPI Day 1）

### 1) 安装依赖

```bash
pip install -r requirements.txt
```

### 2) 启动 FastAPI 应用

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3) 验证接口

```bash
# health check
curl -i http://127.0.0.1:8000/ping

# chat（静态回复）
curl -i -X POST http://127.0.0.1:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"你好"}'
```

期望结果：
- `GET /ping` 返回 `200` 和 JSON（如 `{"status":"ok"}`）
- `POST /chat` 返回 `200` 和 JSON（如 `{"answer":"这是一个静态回复"}`）

## 测试

```bash
pytest tests/test_app_main.py
```

> 开发约定：后续新增接口/函数/方法时，必须同步补充对应单元测试。
