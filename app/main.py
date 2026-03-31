import asyncio
import json
import logging
import os
import time

import redis
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI()

REDIS_TTL_SECONDS = 24 * 60 * 60
REDIS_KEY_PREFIX = 'chat:memory:'

redis_client: redis.Redis | None = None


class ChatRequest(BaseModel):
    message: str
    session_id: str


def get_redis_client() -> redis.Redis:
    global redis_client
    if redis_client is None:
        redis_url = os.getenv('REDIS_URL', 'redis://127.0.0.1:6379/0')
        redis_client = redis.from_url(redis_url, decode_responses=True)
    return redis_client


def memory_key(session_id: str) -> str:
    return f'{REDIS_KEY_PREFIX}{session_id}'


def get_memory(session_id: str) -> list[dict[str, str]]:
    raw_messages = get_redis_client().lrange(memory_key(session_id), 0, -1)
    return [json.loads(item) for item in raw_messages]


def append_message(session_id: str, message: dict[str, str]) -> None:
    client = get_redis_client()
    key = memory_key(session_id)
    client.rpush(key, json.dumps(message, ensure_ascii=False))
    client.expire(key, REDIS_TTL_SECONDS)


@app.middleware('http')
async def log_request_response(request: Request, call_next):
    start = time.perf_counter()
    logger.info('request method=%s path=%s', request.method, request.url.path)

    response = await call_next(request)

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        'response method=%s path=%s status=%s elapsed_ms=%.2f',
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
    )
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning('validation_error path=%s detail=%s', request.url.path, exc.errors())
    return JSONResponse(
        status_code=422,
        content={
            'error': 'invalid_request',
            'code': 422,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception('internal_error path=%s error=%s', request.url.path, str(exc))
    return JSONResponse(
        status_code=500,
        content={
            'error': 'internal_server_error',
            'code': 500,
        },
    )


@app.get('/ping')
def ping() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/chat')
async def chat(req: ChatRequest) -> dict[str, object]:
    if req.message == 'raise_error':
        raise RuntimeError('mocked error for testing')

    await asyncio.sleep(0.2)

    append_message(req.session_id, {'role': 'user', 'content': req.message})
    answer = '这是一个静态回复'
    append_message(req.session_id, {'role': 'assistant', 'content': answer})

    return {
        'answer': answer,
        'session_id': req.session_id,
        'history': get_memory(req.session_id),
    }
