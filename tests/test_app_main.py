import asyncio
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app


client = TestClient(app)


def test_ping_returns_200_and_status_ok() -> None:
    response = client.get('/ping')

    assert response.status_code == 200
    assert response.json() == {'status': 'ok'}


def test_chat_returns_200_and_static_answer_with_session_id() -> None:
    response = client.post('/chat', json={'message': '你好', 'session_id': 's-1'})

    assert response.status_code == 200
    assert response.json() == {'answer': '这是一个静态回复', 'session_id': 's-1'}


def test_chat_request_validation_requires_message_and_session_id() -> None:
    response = client.post('/chat', json={'message': 'only-message'})

    assert response.status_code == 422


def test_chat_concurrent_requests_do_not_serialize() -> None:
    async def _run() -> float:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url='http://testserver') as ac:
            payloads = [
                {'message': f'hello-{i}', 'session_id': f's-{i}'}
                for i in range(5)
            ]
            start = time.perf_counter()
            responses = await asyncio.gather(
                *(ac.post('/chat', json=payload) for payload in payloads)
            )
            elapsed = time.perf_counter() - start

        assert all(resp.status_code == 200 for resp in responses)
        return elapsed

    elapsed = asyncio.run(_run())

    # 单请求 sleep 为 1s；若串行约 5s，并发应显著小于 5s。
    assert elapsed < 3.0
