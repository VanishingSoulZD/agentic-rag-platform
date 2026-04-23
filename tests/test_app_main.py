import json
import logging

from fastapi.testclient import TestClient

import app.main as main

client = TestClient(main.app)
error_client = TestClient(main.app, raise_server_exceptions=False)


logger = logging.getLogger(__name__)


def _cleanup_session(session_id: str) -> None:
    main.chat_store.delete_session(session_id)


def test_ping_returns_200_and_status_ok() -> None:
    response = client.get("/ping")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_metrics_endpoint_returns_prometheus_text() -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert "response_time_ms" in response.text
    assert "success_rate" in response.text
    assert "cache_hit_rate" in response.text


def test_chat_returns_200_and_answer_with_session_id() -> None:
    session_id = "test_chat_returns_200_and_answer_with_session_id"
    _cleanup_session(session_id)

    response = client.post("/chat", json={"message": "hello", "session_id": session_id})

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == session_id
    assert "answer" in body
    assert "use" in body
    assert set(body["use"].keys()) == {
        "model",
        "mock",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }

    history = main.chat_store.get_memory(session_id)
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hello"}
    assert history[-1]["role"] == "assistant"

    _cleanup_session(session_id)


def test_chat_stores_multi_turn_messages_and_ttl_when_redis_available() -> None:
    session_id = "test_chat_stores_multi_turn_messages_and_ttl_when_redis_available"
    _cleanup_session(session_id)

    response1 = client.post(
        "/chat", json={"message": "hello", "session_id": session_id}
    )
    response2 = client.post(
        "/chat", json={"message": "how are you", "session_id": session_id}
    )

    assert response1.status_code == 200
    assert response2.status_code == 200

    history = main.chat_store.get_memory(session_id)
    assert len(history) == 4

    if (
        not main.chat_store.using_memory_fallback
        and main.chat_store.is_redis_available()
    ):
        logger.debug("Redis is available")
        ttl = main.chat_store.get_redis_client().ttl(
            main.chat_store.memory_key(session_id)
        )
        assert 0 < ttl <= main.REDIS_TTL_SECONDS
    else:
        # fallback mode: no Redis TTL, but history remains functional.
        logger.debug("Redis is NOT available")
        assert main.chat_store.using_memory_fallback is True

    _cleanup_session(session_id)


def test_chat_stream_outputs_token_then_usage_events() -> None:
    session_id = "test_chat_stream_outputs_token_then_usage_events"
    _cleanup_session(session_id)

    with client.stream(
        "POST",
        "/chat/stream",
        json={"message": "hello stream", "session_id": session_id},
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line]

    data_lines = [line for line in lines if line.startswith("data: ")]
    assert len(data_lines) >= 3

    first_event = json.loads(data_lines[0].replace("data: ", "", 1))
    assert first_event["type"] == "token"
    assert first_event["content"]

    usage_events = []
    for line in data_lines:
        payload = line.replace("data: ", "", 1)
        if payload == "[DONE]":
            continue
        event = json.loads(payload)
        if event.get("type") == "usage":
            usage_events.append(event)

    assert usage_events
    usage = usage_events[-1]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0

    history = main.chat_store.get_memory(session_id)
    assert history[-1]["role"] == "assistant"
    assert history[-1]["content"]

    _cleanup_session(session_id)


def test_chat_stream_cache_key_includes_history_context(monkeypatch) -> None:
    session_id = "test_chat_stream_cache_key_includes_history_context"
    _cleanup_session(session_id)
    main.cache_manager.response_cache.exact = {}
    main.cache_manager.response_cache.semantic_entries = []
    main.cache_manager.embedding_cache.provider._model_failed = True

    call_count = {"n": 0}

    async def _fake_stream_chat(_history):
        call_count["n"] += 1
        yield {"type": "token", "content": "hello"}
        yield {
            "type": "usage",
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
            "model": "mock",
            "mock": True,
        }

    monkeypatch.setattr(main.llm_client, "stream_chat", _fake_stream_chat)

    payload = {"message": "hello stream cache", "session_id": session_id}
    with client.stream("POST", "/chat/stream", json=payload) as first:
        assert first.status_code == 200
        first_lines = [line for line in first.iter_lines() if line]
    with client.stream("POST", "/chat/stream", json=payload) as second:
        assert second.status_code == 200
        second_lines = [line for line in second.iter_lines() if line]

    assert call_count["n"] == 2
    first_data = [line for line in first_lines if line.startswith("data: ")]
    second_data = [line for line in second_lines if line.startswith("data: ")]
    assert any('"total_tokens": 7' in item for item in first_data)
    assert any('"total_tokens": 7' in item for item in second_data)

    _cleanup_session(session_id)


def test_rag_returns_retrieval_based_answer_and_doc_ids(monkeypatch) -> None:
    class _FakeLLMResult:
        answer = "RAG final answer"
        model = "mock"
        mock = True
        prompt_tokens = 1
        completion_tokens = 1
        total_tokens = 2

    def _fake_rag_search(query: str, k: int = 5):
        return {
            "query": query,
            "doc_ids": ["doc1.txt", "doc2.txt"],
            "docs": [
                {"doc_id": "doc1.txt", "text": "a"},
                {"doc_id": "doc2.txt", "text": "b"},
            ],
        }

    async def _fake_chat(_messages):
        return _FakeLLMResult()

    monkeypatch.setattr(main, "rag_search", _fake_rag_search)
    monkeypatch.setattr(main.llm_client, "chat", _fake_chat)

    response = client.post(
        "/rag/query",
        json={
            "query": "what is rag?",
            "session_id": "rag-s1",
            "k": 3,
            "rewrite_query": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "RAG final answer"
    assert body["doc_ids"] == ["doc1.txt", "doc2.txt"]


def test_query_rag_returns_retrieval_items_and_answer(monkeypatch) -> None:
    class _FakeLLMResult:
        answer = "query_rag final answer"
        model = "mock"
        mock = True
        prompt_tokens = 2
        completion_tokens = 3
        total_tokens = 5

    def _fake_rag_search(query: str, k: int = 5):
        return {
            "query": query,
            "doc_ids": ["doc11.txt", "doc5.txt"],
            "docs": [
                {"doc_id": "doc11.txt", "text": "faiss text", "rerank_score": 0.91},
                {"doc_id": "doc5.txt", "text": "vector db text", "rerank_score": 0.73},
            ],
        }

    async def _fake_chat(_messages):
        return _FakeLLMResult()

    monkeypatch.setattr(main, "rag_search", _fake_rag_search)
    monkeypatch.setattr(main.llm_client, "chat", _fake_chat)

    response = client.post(
        "/rag/query",
        json={
            "query": "what is faiss",
            "session_id": "qrag-s1",
            "k": 2,
            "rewrite_query": False,
        },
    )
    assert response.status_code == 200
    body = response.json()

    assert body["answer"] == "query_rag final answer"
    assert body["doc_ids"] == ["doc11.txt", "doc5.txt"]
    assert body["retrieval"]["doc_ids"] == ["doc11.txt", "doc5.txt"]
    assert body["retrieval"]["rerank_scores"] == [0.91, 0.73]
    assert len(body["retrieval"]["items"]) == 2


def test_query_rag_semantic_cache_hit_reduces_llm_calls(monkeypatch) -> None:
    class _FakeLLMResult:
        answer = "cached answer"
        model = "mock"
        mock = True
        prompt_tokens = 2
        completion_tokens = 3
        total_tokens = 5

    call_count = {"n": 0}

    def _fake_rag_search(query: str, k: int = 5):
        return {
            "query": query,
            "doc_ids": ["doc11.txt"],
            "docs": [
                {"doc_id": "doc11.txt", "text": "faiss text", "rerank_score": 0.91},
            ],
        }

    async def _fake_chat(_messages):
        call_count["n"] += 1
        return _FakeLLMResult()

    monkeypatch.setattr(main, "rag_search", _fake_rag_search)
    monkeypatch.setattr(main.llm_client, "chat", _fake_chat)

    # reset layered caches + avoid external model loading
    main.cache_manager.response_cache.exact = {}
    main.cache_manager.response_cache.semantic_entries = []
    main.cache_manager.retrieval_cache.entries = []
    main.cache_manager.embedding_cache.provider._model_failed = True

    payload = {
        "query": "what is faiss",
        "session_id": "cache-s1",
        "k": 2,
        "rewrite_query": False,
    }
    r1 = client.post("/rag/query", json=payload)
    r2 = client.post("/rag/query", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert call_count["n"] == 1
    assert r1.json()["cache"]["hit"] is False
    assert r2.json()["cache"]["hit"] is True


def test_query_rag_retrieval_cache_respects_k(monkeypatch) -> None:
    class _FakeLLMResult:
        answer = "answer"
        model = "mock"
        mock = True
        prompt_tokens = 1
        completion_tokens = 1
        total_tokens = 2

    def _fake_rag_search(query: str, k: int = 5):
        docs = [
            {"doc_id": f"doc{i}.txt", "text": f"t{i}", "rerank_score": 1.0 - (i * 0.01)}
            for i in range(1, k + 1)
        ]
        return {"query": query, "doc_ids": [d["doc_id"] for d in docs], "docs": docs}

    async def _fake_chat(_messages):
        return _FakeLLMResult()

    monkeypatch.setattr(main, "rag_search", _fake_rag_search)
    monkeypatch.setattr(main.llm_client, "chat", _fake_chat)

    main.cache_manager.response_cache.exact = {}
    main.cache_manager.response_cache.semantic_entries = []
    main.cache_manager.retrieval_cache.entries = []
    main.cache_manager.embedding_cache.provider._model_failed = True

    payload_k1 = {
        "query": "what is faiss",
        "session_id": "k-s1",
        "k": 1,
        "rewrite_query": False,
    }
    payload_k3 = {
        "query": "what is faiss",
        "session_id": "k-s2",
        "k": 3,
        "rewrite_query": False,
    }
    r1 = client.post("/rag/query", json=payload_k1)
    r2 = client.post("/rag/query", json=payload_k3)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert len(r1.json()["doc_ids"]) == 1
    assert len(r2.json()["doc_ids"]) == 3


def test_query_rag_rate_limiter_per_session(monkeypatch) -> None:
    class _FakeLLMResult:
        answer = "ok"
        model = "mock"
        mock = True
        prompt_tokens = 1
        completion_tokens = 1
        total_tokens = 2

    def _fake_rag_search(query: str, k: int = 5):
        return {
            "query": query,
            "doc_ids": ["doc1.txt"],
            "docs": [{"doc_id": "doc1.txt", "text": "x", "rerank_score": 0.9}],
        }

    async def _fake_chat(_messages):
        return _FakeLLMResult()

    monkeypatch.setattr(main, "rag_search", _fake_rag_search)
    monkeypatch.setattr(main.llm_client, "chat", _fake_chat)

    old_limit = main.session_rate_limiter.limit
    old_window_seconds = main.session_rate_limiter.window_seconds

    try:
        # force rate limit on second request
        main.session_rate_limiter.limit = 1
        main.session_rate_limiter.window_seconds = 60
        main.session_rate_limiter._events.clear()
        main.cache_manager.response_cache.exact = {}
        main.cache_manager.response_cache.semantic_entries = []
        main.cache_manager.retrieval_cache.entries = []
        main.cache_manager.embedding_cache.provider._model_failed = True

        payload = {
            "query": "what is faiss",
            "session_id": "limit-s1",
            "k": 1,
            "rewrite_query": False,
        }
        first = client.post("/rag/query", json=payload)
        second = client.post("/rag/query", json=payload)

        assert first.status_code == 200
        assert second.status_code == 429
        body = second.json()
        assert body["error"] == "rate_limited"
        assert body["code"] == 429
    finally:
        main.session_rate_limiter.limit = old_limit
        main.session_rate_limiter.window_seconds = old_window_seconds
        main.session_rate_limiter._events.clear()


def test_chat_rate_limiter_per_session(monkeypatch) -> None:
    class _FakeLLMResult:
        answer = "ok"
        model = "mock"
        mock = True
        prompt_tokens = 1
        completion_tokens = 1
        total_tokens = 2

    async def _fake_chat(_messages):
        return _FakeLLMResult()

    monkeypatch.setattr(main.llm_client, "chat", _fake_chat)

    session_id = "test_chat_rate_limiter_per_session"
    _cleanup_session(session_id)

    old_limit = main.session_rate_limiter.limit
    old_window_seconds = main.session_rate_limiter.window_seconds

    try:
        main.session_rate_limiter.limit = 1
        main.session_rate_limiter.window_seconds = 60
        main.session_rate_limiter._events.clear()

        first = client.post(
            "/chat", json={"message": "hello", "session_id": session_id}
        )
        second = client.post(
            "/chat", json={"message": "again", "session_id": session_id}
        )

        assert first.status_code == 200
        assert second.status_code == 429
        body = second.json()
        assert body["error"] == "rate_limited"
        assert body["code"] == 429
    finally:
        main.session_rate_limiter.limit = old_limit
        main.session_rate_limiter.window_seconds = old_window_seconds
        main.session_rate_limiter._events.clear()
        _cleanup_session(session_id)


def test_chat_stream_rate_limiter_per_session(monkeypatch) -> None:
    async def _fake_stream_chat(_history):
        yield {"type": "token", "content": "hello"}
        yield {
            "type": "usage",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "model": "mock",
            "mock": True,
        }

    monkeypatch.setattr(main.llm_client, "stream_chat", _fake_stream_chat)

    session_id = "test_chat_stream_rate_limiter_per_session"
    _cleanup_session(session_id)

    old_limit = main.session_rate_limiter.limit
    old_window_seconds = main.session_rate_limiter.window_seconds

    try:
        main.session_rate_limiter.limit = 1
        main.session_rate_limiter.window_seconds = 60
        main.session_rate_limiter._events.clear()

        with client.stream(
            "POST", "/chat/stream", json={"message": "hello", "session_id": session_id}
        ) as first:
            assert first.status_code == 200
            list(first.iter_lines())

        with client.stream(
            "POST", "/chat/stream", json={"message": "again", "session_id": session_id}
        ) as second:
            assert second.status_code == 429
            body = b"".join(second.iter_bytes()).decode()

        assert json.loads(body)["error"] == "rate_limited"
    finally:
        main.session_rate_limiter.limit = old_limit
        main.session_rate_limiter.window_seconds = old_window_seconds
        main.session_rate_limiter._events.clear()
        _cleanup_session(session_id)


def test_chat_stream_rate_limited_requests_are_recorded_in_metrics(monkeypatch) -> None:
    async def _fake_stream_chat(_history):
        yield {"type": "token", "content": "hello"}
        yield {
            "type": "usage",
            "prompt_tokens": 1,
            "completion_tokens": 1,
            "total_tokens": 2,
            "model": "mock",
            "mock": True,
        }

    monkeypatch.setattr(main.llm_client, "stream_chat", _fake_stream_chat)

    session_id = "test_chat_stream_rate_limited_requests_are_recorded_in_metrics"
    _cleanup_session(session_id)

    old_limit = main.session_rate_limiter.limit
    old_window_seconds = main.session_rate_limiter.window_seconds
    request_count_before = main.metrics_store._request_count
    success_count_before = main.metrics_store._success_count

    try:
        main.session_rate_limiter.limit = 1
        main.session_rate_limiter.window_seconds = 60
        main.session_rate_limiter._events.clear()

        with client.stream(
            "POST", "/chat/stream", json={"message": "hello", "session_id": session_id}
        ) as first:
            assert first.status_code == 200
            list(first.iter_lines())

        with client.stream(
            "POST", "/chat/stream", json={"message": "again", "session_id": session_id}
        ) as second:
            assert second.status_code == 429
            body = b"".join(second.iter_bytes()).decode()

        assert json.loads(body)["error"] == "rate_limited"
        assert main.metrics_store._request_count == request_count_before + 2
        assert main.metrics_store._success_count == success_count_before + 2
    finally:
        main.session_rate_limiter.limit = old_limit
        main.session_rate_limiter.window_seconds = old_window_seconds
        main.session_rate_limiter._events.clear()
        _cleanup_session(session_id)
