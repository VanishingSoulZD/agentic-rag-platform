import asyncio
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app.main as main
from app.llm_client import AsyncLLMClient

client = TestClient(main.app)


def _require_redis() -> None:
    try:
        main.get_redis_client().ping()
    except ConnectionError:
        pytest.skip('Redis is not available on 127.0.0.1:6379; start docker redis first.')


def _cleanup_session(session_id: str) -> None:
    main.get_redis_client().delete(main.memory_key(session_id))


def test_ping_returns_200_and_status_ok() -> None:
    response = client.get('/ping')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_chat_validation_error_returns_unified_format() -> None:
    response = client.post('/chat', json={'message': 'hello'})

    assert response.status_code == 422
    assert response.json() == {'error': 'invalid_request', 'code': 422}


def test_llm_client_mock_chat_returns_usage_and_answer() -> None:
    llm = AsyncLLMClient(api_key=None)

    result = asyncio.run(
        llm.chat([{'role': 'user', 'content': '你好，介绍一下你自己'}])
    )

    assert result.mock is True
    assert result.answer.startswith('[MOCK]')
    assert result.total_tokens >= result.prompt_tokens + result.completion_tokens - 1


def test_chat_stores_multi_turn_messages_and_use_field() -> None:
    _require_redis()
    session_id = 'session-day5-chat'
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

    history = main.get_memory(session_id)
    assert len(history) >= 2
    assert history[0] == {'role': 'user', 'content': 'hello'}
    assert history[-1]['role'] == 'assistant'

    _cleanup_session(session_id)
