import asyncio
from time import sleep

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.llm_client import AsyncLLMClient
from app.retrieval.retriever import ask_sync

llm_client = AsyncLLMClient()

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: bool | None = None


@app.get("/")
async def read_root():
    await asyncio.sleep(2)
    return {"Hello": "World11"}


@app.get("/test")
def test():
    sleep(2)
    return {"Hello": "World11"}


@app.get("/items/{item_id}")
async def read_item(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q": q}


@app.post("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    return {"item_id": item_id, "item_name": item.name}


@app.post("/chat")
async def chat(message: str):
    result = await llm_client.chat([{"role": "user", "content": message}])
    return {"message": result.answer}


@app.post("/chat/stream")
async def chat_stream(message: str):
    async def event_generator():
        async for event in llm_client.stream_chat([{"role": "user", "content": message}]):
            if event["type"] == "token":
                yield f"data: {event['content']}\n\n"
            if event["type"] == "usage":
                yield f"data: {event}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class QueryRequest(BaseModel):
    session_id: str
    question: str


@app.post("/ask")
def ask_endpoint(req: QueryRequest):
    result = ask_sync(req.session_id, req.question)
    doc_ids = [c["doc_id"] for c in result["docs"]]
    print("Retrieved doc_ids:", doc_ids)
    return result
