import sys
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app

client = TestClient(app)
error_client = TestClient(app, raise_server_exceptions=False)


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
