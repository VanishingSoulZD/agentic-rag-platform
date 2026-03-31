from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


@app.get('/ping')
def ping() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/chat')
def chat(req: ChatRequest) -> dict[str, str]:
    return {'answer': '这是一个静态回复'}
