# Performance Report (Real API)

## 1) Endpoint Validation

| Endpoint | Method | Status | OK | Note |
|---|---|---:|---:|---|
| /ping | GET | 200 | True | body={'status': 'ok'} |
| /agent/trace | POST | 200 | True | trace_id=2fe7e21992c0 |
| /agent/trace/2fe7e21992c0 | GET | 200 | True | keys=['meta', 'nodes', 'edges', 'observations'] |
| /agent/trace/2fe7e21992c0/view | GET | 200 | True | html_len=2495 |
| /chat | POST | 200 | True | keys=['answer', 'session_id', 'history', 'use', 'cache', 'cache_layers'] |
| /chat/stream | POST | 200 | True | token=True,usage=True,done=True |
| /rag/query | POST | 200 | True | doc_ids=3 |

## 2) 100-Concurrency Load Test (/chat)

| Metric | Value |
|---|---:|
| Concurrency | 100 |
| Total Requests | 1000 |
| Success | 1000 |
| Errors | 0 |
| Error Rate | 0.00% |
| P50 (ms) | 207.67 |
| P95 (ms) | 378.12 |
| P99 (ms) | 1138.88 |
| QPS | 416.81 |
| Estimated Cost (USD) | 0.006877 |

## 3) Acceptance
- Error rate < 2%: **True**
- P95 <= threshold (1500.0ms): **True**
- Overall pass (error + p95): **True**

## 4) Notes
- This report uses real HTTP calls to running FastAPI endpoints.
- LLM may run in mock mode if OPENAI_API_KEY is not configured.
- CPU/GPU metrics are not exposed by current API; add node/GPU telemetry for full observability.
