import asyncio

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ChatRequest(BaseModel):
    message: str
    session_id: str


@app.get('/ping')
def ping() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/chat')
async def chat(req: ChatRequest) -> dict[str, str]:
    # 模拟 I/O 等待，验证接口在并发请求下不会阻塞整个服务线程。
    await asyncio.sleep(1)
    return {
        'answer': '这是一个静态回复',
        'session_id': req.session_id,
    }
