import asyncio
import json
import logging
import os
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.llm_client import AsyncLLMClient
from app.logging_setup import configure_logging, reset_request_id, set_request_id
from app.memory import ChatStoreConfig, HybridChatStore
from app.retrieval.retriever import rag_search

configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI()

REDIS_TTL_SECONDS = 24 * 60 * 60
REDIS_KEY_PREFIX = 'chat:memory:'
REQUEST_ID_HEADER = 'X-Request-ID'

chat_store = HybridChatStore(
    ChatStoreConfig(
        redis_url=os.getenv('REDIS_URL', 'redis://127.0.0.1:6379/0'),
        key_prefix=REDIS_KEY_PREFIX,
        ttl_seconds=REDIS_TTL_SECONDS,
    )
)
llm_client = AsyncLLMClient(
    timeout_seconds=float(os.getenv('OPENAI_TIMEOUT_SECONDS', '20')),
    max_retries=int(os.getenv('OPENAI_MAX_RETRIES', '2')),
)


class ChatRequest(BaseModel):
    message: str
    session_id: str


class RagRequest(BaseModel):
    query: str
    session_id: str
    k: int = 5
    rewrite_query: bool = True


@app.middleware('http')
async def log_request_response(request: Request, call_next):
    request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid4()))
    token = set_request_id(request_id)
    start = time.perf_counter()

    logger.info(
        'request_start method=%s path=%s client=%s',
        request.method,
        request.url.path,
        request.client.host if request.client else '-',
    )

    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            'request_failed method=%s path=%s elapsed_ms=%.2f',
            request.method,
            request.url.path,
            elapsed_ms,
        )
        raise
    else:
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            'request_end method=%s path=%s status=%s elapsed_ms=%.2f',
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response
    finally:
        reset_request_id(token)


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

    # 模拟 I/O 等待，验证接口在并发请求下不会阻塞整个服务线程。
    await asyncio.sleep(1)

    chat_store.append_message(req.session_id, {'role': 'user', 'content': req.message})

    history = chat_store.get_memory(req.session_id)
    llm_result = await llm_client.chat(history)

    chat_store.append_message(req.session_id, {'role': 'assistant', 'content': llm_result.answer})

    return {
        'answer': llm_result.answer,
        'session_id': req.session_id,
        'history': chat_store.get_memory(req.session_id),
        'use': {
            'model': llm_result.model,
            'mock': llm_result.mock,
            'prompt_tokens': llm_result.prompt_tokens,
            'completion_tokens': llm_result.completion_tokens,
            'total_tokens': llm_result.total_tokens,
        },
    }


@app.post('/chat/stream')
async def chat_stream(req: ChatRequest):
    chat_store.append_message(req.session_id, {'role': 'user', 'content': req.message})
    history = chat_store.get_memory(req.session_id)

    async def event_gen():
        full_answer = ''

        async for event in llm_client.stream_chat(history):
            if event['type'] == 'token':
                token = str(event['content'])
                full_answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
            elif event['type'] == 'usage':
                usage_payload = {
                    'type': 'usage',
                    'prompt_tokens': event['prompt_tokens'],
                    'completion_tokens': event['completion_tokens'],
                    'total_tokens': event['total_tokens'],
                    'model': event['model'],
                    'mock': event['mock'],
                }
                logger.info(
                    'chat_stream_usage session_id=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s',
                    req.session_id,
                    event['prompt_tokens'],
                    event['completion_tokens'],
                    event['total_tokens'],
                )
                yield f"data: {json.dumps(usage_payload, ensure_ascii=False)}\n\n"

        chat_store.append_message(req.session_id, {'role': 'assistant', 'content': full_answer})
        yield 'data: [DONE]\n\n'

    return StreamingResponse(event_gen(), media_type='text/event-stream')


async def _rewrite_query_with_history(query: str, history: list[dict[str, str]]) -> str:
    if not history:
        return query

    recent_history = history[-6:]
    messages = [
        {
            'role': 'system',
            'content': (
                'Rewrite the latest user query into a standalone search query. '
                'Return only rewritten query text.'
            ),
        },
        *recent_history,
        {'role': 'user', 'content': query},
    ]
    result = await llm_client.chat(messages)
    rewritten = result.answer.strip()
    return rewritten or query


def _build_rag_messages(
        query: str,
        history: list[dict[str, str]],
        docs: list[dict[str, object]],
) -> list[dict[str, str]]:
    context = "\n\n".join([f"[{d['doc_id']}] {d['text']}" for d in docs])
    recent_history = history[-8:]
    return [
        {
            'role': 'system',
            'content': (
                'You are a RAG assistant. Answer with retrieved context + conversation history only. '
                'If context does not contain the answer, say: I don\'t know.'
            ),
        },
        *recent_history,
        {
            'role': 'user',
            'content': f"Retrieved context:\n{context}\n\nQuestion: {query}",
        },
    ]


@app.post('/rag')
async def rag(req: RagRequest) -> dict[str, object]:
    history = chat_store.get_memory(req.session_id)
    rewritten_query = req.query
    if req.rewrite_query:
        rewritten_query = await _rewrite_query_with_history(req.query, history)

    retrieval_result = rag_search(rewritten_query, k=req.k)
    docs = retrieval_result['docs']
    doc_ids = retrieval_result['doc_ids']

    prompt_messages = _build_rag_messages(req.query, history, docs)
    llm_result = await llm_client.chat(prompt_messages)

    chat_store.append_message(req.session_id, {'role': 'user', 'content': req.query})
    chat_store.append_message(req.session_id, {'role': 'assistant', 'content': llm_result.answer})

    logger.info(f'Retrieved {doc_ids=}')

    return {
        'session_id': req.session_id,
        'query': req.query,
        'rewritten_query': rewritten_query,
        'answer': llm_result.answer,
        'sources': docs,
        'doc_ids': doc_ids,
        'use': {
            'model': llm_result.model,
            'mock': llm_result.mock,
            'prompt_tokens': llm_result.prompt_tokens,
            'completion_tokens': llm_result.completion_tokens,
            'total_tokens': llm_result.total_tokens,
        },
    }
