# Performance Report (Real API)

## 1) Endpoint Validation

| Endpoint | Method | Status | OK | Note |
|---|---|---:|---:|---|
| /ping | GET | 200 | True | body={'status': 'ok'} |
| /agent/trace | POST | 200 | True | trace_id=28241502436f |
| /agent/trace/28241502436f | GET | 200 | True | keys=['meta', 'nodes', 'edges', 'observations'] |
| /agent/trace/28241502436f/view | GET | 200 | True | html_len=2495 |
| /chat | POST | 200 | True | keys=['answer', 'session_id', 'history', 'use', 'cache', 'cache_layers'] |
| /chat/stream | POST | 200 | True | token=True,usage=True,done=True |
| /rag/query | POST | 500 | False | error=internal_server_error code=500 |

## 2) 100-Concurrency Load Test (/chat)

| Metric | Value |
|---|---:|
| Concurrency | 100 |
| Total Requests | 1000 |
| Success | 1000 |
| Errors | 0 |
| Error Rate | 0.00% |
| P50 (ms) | 242.94 |
| P95 (ms) | 494.81 |
| P99 (ms) | 497.59 |
| QPS | 341.5 |
| Estimated Cost (USD) | 0.005665 |

## 3) Acceptance
- Error rate < 2%: **True**
- P95 <= threshold (1500.0ms): **True**
- Overall pass (error + p95): **True**

## 4) Notes
- This report uses real HTTP calls to running FastAPI endpoints.
- LLM may run in mock mode if OPENAI_API_KEY is not configured.
- CPU/GPU metrics are not exposed by current API; add node/GPU telemetry for full observability.


## 5) Scope Clarification (Day21/Day29)
- Day21/Day29 acceptance is now based on **real API test artifacts only**.
- Legacy simulation artifacts have been removed from the repository:
  - `scripts/stability_stress_test.py`
  - `reports/stability_stress_results.json`
  - `reports/stability_stress_summary.md`
- This file and `reports/performance_final_results.json` are the authoritative outputs.
