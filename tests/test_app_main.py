import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app.main as main

client = TestClient(main.app)
error_client = TestClient(main.app, raise_server_exceptions=False)


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


def test_chat_internal_error_returns_unified_format() -> None:
    response = error_client.post(
        '/chat',
        json={'message': 'raise_error', 'session_id': 's-1'},
    )

    assert response.status_code == 500
    assert response.json() == {'error': 'internal_server_error', 'code': 500}


def test_chat_stores_multi_turn_messages_in_redis() -> None:
    _require_redis()
    session_id = 'session-day4-chat'
    _cleanup_session(session_id)

    response1 = client.post('/chat', json={'message': 'hello', 'session_id': session_id})
    response2 = client.post('/chat', json={'message': 'how are you', 'session_id': session_id})

    assert response1.status_code == 200
    assert response2.status_code == 200

    history = main.get_memory(session_id)
    assert history == [
        {'role': 'user', 'content': 'hello'},
        {'role': 'assistant', 'content': '这是一个静态回复'},
        {'role': 'user', 'content': 'how are you'},
        {'role': 'assistant', 'content': '这是一个静态回复'},
    ]

    ttl = main.get_redis_client().ttl(main.memory_key(session_id))
    assert 0 < ttl <= main.REDIS_TTL_SECONDS

    _cleanup_session(session_id)
