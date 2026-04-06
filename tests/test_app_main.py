import asyncio
import json
import time

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

import app.main as main
from app.llm_client import AsyncLLMClient

client = TestClient(main.app)
error_client = TestClient(main.app, raise_server_exceptions=False)


def _cleanup_session(session_id: str) -> None:
    main.chat_store.delete_session(session_id)


def test_ping_returns_200_and_status_ok() -> None:
    response = client.get('/ping')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_ping_response_contains_request_id_header() -> None:
    response = client.get('/ping')

    assert response.status_code == 200
    assert response.headers.get('X-Request-ID')




def test_metrics_endpoint_returns_prometheus_text() -> None:
    response = client.get('/metrics')

    assert response.status_code == 200
    assert 'response_time_ms' in response.text
    assert 'success_rate' in response.text
    assert 'cache_hit_rate' in response.text

def test_chat_returns_200_and_answer_with_session_id() -> None:
    session_id = 'test_chat_returns_200_and_answer_with_session_id'
    _cleanup_session(session_id)

    response = client.post('/chat', json={'message': 'hello', 'session_id': session_id})

    assert response.status_code == 200
    body = response.json()
    assert body['session_id'] == session_id
    assert 'answer' in body
    assert 'use' in body
    assert set(body['use'].keys()) == {
        'model',
        'mock',
        'prompt_tokens',
        'completion_tokens',
        'total_tokens',
    }

    history = main.chat_store.get_memory(session_id)
    assert len(history) == 2
    assert history[0] == {'role': 'user', 'content': 'hello'}
    assert history[-1]['role'] == 'assistant'

    _cleanup_session(session_id)


def test_chat_validation_error_returns_unified_format() -> None:
    response = client.post('/chat', json={'message': 'only-message'})

    assert response.status_code == 422
    assert response.json() == {'error': 'invalid_request', 'code': 422}


def test_chat_internal_error_returns_unified_format() -> None:
    session_id = 'test_chat_internal_error_returns_unified_format'
    response = error_client.post(
        '/chat',
        json={'message': 'raise_error', 'session_id': session_id},
    )

    assert response.status_code == 500
    assert response.json() == {'error': 'internal_server_error', 'code': 500}


def test_chat_concurrent_requests_do_not_serialize() -> None:
    concurrent_count = 3
    for i in range(concurrent_count):
        _cleanup_session(f'test_chat_concurrent_requests_do_not_serialize-{i}')

    async def _run() -> float:
        transport = ASGITransport(app=main.app)
        async with AsyncClient(transport=transport, base_url='http://testserver') as ac:
            payloads = [
                {'message': f'hello-{i}', 'session_id': f'test_chat_concurrent_requests_do_not_serialize-{i}'}
                for i in range(concurrent_count)
            ]
            start = time.perf_counter()
            responses = await asyncio.gather(
                *(ac.post('/chat', json=payload) for payload in payloads)
            )
            elapsed = time.perf_counter() - start

        assert all(resp.status_code == 200 for resp in responses)
        return elapsed

    elapsed = asyncio.run(_run())

    for i in range(concurrent_count):
        _cleanup_session(f'test_chat_concurrent_requests_do_not_serialize-{i}')

    assert elapsed < concurrent_count


def test_llm_client_mock_chat_returns_usage_and_answer() -> None:
    llm = AsyncLLMClient(api_key=None)

    result = asyncio.run(
        llm.chat([{'role': 'user', 'content': 'hello'}])
    )

    assert result.mock is True
    assert result.answer.startswith('[MOCK]')
    assert result.total_tokens >= result.prompt_tokens + result.completion_tokens - 1


def test_chat_stores_multi_turn_messages_and_ttl_when_redis_available() -> None:
    session_id = 'test_chat_stores_multi_turn_messages_and_ttl_when_redis_available'
    _cleanup_session(session_id)

    response1 = client.post('/chat', json={'message': 'hello', 'session_id': session_id})
    response2 = client.post('/chat', json={'message': 'how are you', 'session_id': session_id})

    assert response1.status_code == 200
    assert response2.status_code == 200

    history = main.chat_store.get_memory(session_id)
    assert len(history) == 4

    if not main.chat_store.using_memory_fallback and main.chat_store.is_redis_available():
        ttl = main.chat_store.get_redis_client().ttl(main.chat_store.memory_key(session_id))
        assert 0 < ttl <= main.REDIS_TTL_SECONDS
    else:
        # fallback mode: no Redis TTL, but history remains functional.
        assert main.chat_store.using_memory_fallback is True

    _cleanup_session(session_id)


def test_memory_survives_app_restart_with_same_store_backend() -> None:
    session_id = 'test_memory_survives_app_restart_with_same_store_backend'
    _cleanup_session(session_id)
    response1 = client.post('/chat', json={'message': 'hello', 'session_id': session_id})
    assert response1.status_code == 200
    history1 = main.chat_store.get_memory(session_id)
    assert len(history1) == 2

    restarted_client = TestClient(main.app)
    response2 = restarted_client.post('/chat', json={'message': 'how are you', 'session_id': session_id})
    assert response2.status_code == 200
    history2 = main.chat_store.get_memory(session_id)
    assert len(history2) == 4

    _cleanup_session(session_id)


def test_chat_stream_outputs_token_then_usage_events() -> None:
    session_id = 'test_chat_stream_outputs_token_then_usage_events'
    _cleanup_session(session_id)

    with client.stream(
            'POST',
            '/chat/stream',
            json={'message': 'hello stream', 'session_id': session_id},
    ) as response:
        assert response.status_code == 200
        lines = [line for line in response.iter_lines() if line]

    data_lines = [line for line in lines if line.startswith('data: ')]
    assert len(data_lines) >= 3

    first_event = json.loads(data_lines[0].replace('data: ', '', 1))
    assert first_event['type'] == 'token'
    assert first_event['content']

    usage_events = []
    for line in data_lines:
        payload = line.replace('data: ', '', 1)
        if payload == '[DONE]':
            continue
        event = json.loads(payload)
        if event.get('type') == 'usage':
            usage_events.append(event)

    assert usage_events
    usage = usage_events[-1]
    assert usage['prompt_tokens'] > 0
    assert usage['completion_tokens'] > 0

    history = main.chat_store.get_memory(session_id)
    assert history[-1]['role'] == 'assistant'
    assert history[-1]['content']

    _cleanup_session(session_id)


def test_rag_returns_retrieval_based_answer_and_doc_ids(monkeypatch) -> None:
    class _FakeLLMResult:
        answer = 'RAG final answer'
        model = 'mock'
        mock = True
        prompt_tokens = 1
        completion_tokens = 1
        total_tokens = 2

    def _fake_rag_search(query: str, k: int = 5):
        return {
            'query': query,
            'doc_ids': ['doc1.txt', 'doc2.txt'],
            'docs': [
                {'doc_id': 'doc1.txt', 'text': 'a'},
                {'doc_id': 'doc2.txt', 'text': 'b'},
            ],
        }

    async def _fake_chat(_messages):
        return _FakeLLMResult()

    monkeypatch.setattr(main, 'rag_search', _fake_rag_search)
    monkeypatch.setattr(main.llm_client, 'chat', _fake_chat)

    response = client.post('/rag',
                           json={'query': 'what is rag?', 'session_id': 'rag-s1', 'k': 3, 'rewrite_query': False})
    assert response.status_code == 200
    body = response.json()
    assert body['answer'] == 'RAG final answer'
    assert body['doc_ids'] == ['doc1.txt', 'doc2.txt']
