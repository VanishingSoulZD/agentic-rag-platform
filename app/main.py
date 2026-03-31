import asyncio
import logging
import time

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


class ChatRequest(BaseModel):
    message: str
    session_id: str


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
async def chat(req: ChatRequest) -> dict[str, str]:
    if req.message == 'raise_error':
        raise RuntimeError('mocked error for testing')

    # 模拟 I/O 等待，验证接口在并发请求下不会阻塞整个服务线程。
    await asyncio.sleep(1)
    return {
        'answer': '这是一个静态回复',
        'session_id': req.session_id,
    }
