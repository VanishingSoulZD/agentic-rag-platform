import asyncio
from time import sleep

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from myfaiss.rag import ask
from myllm import MyLLM

llm = MyLLM()

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
    return {"message": llm.llm(message)}

@app.post("/chat/stream")
async def chat_stream():
    message = "说爱我"
    stream = llm.stream_chat(message)

    prompt_tokens = 0
    completion_tokens = 0

    async def event_generator():

        nonlocal prompt_tokens, completion_tokens

        for chunk in stream:

            # token 内容
            delta = chunk.choices[0].delta.content

            if delta:
                completion_tokens += 1

                yield f"data: {delta}\n\n"

            # usage 在最后一个 chunk
            if chunk.usage:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

        # logger.info(
        #     "usage prompt=%s completion=%s",
        #     prompt_tokens,
        #     completion_tokens
        # )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )

class QueryRequest(BaseModel):
    session_id: str
    question: str

@app.post("/ask")
def ask_endpoint(req: QueryRequest):
    result = ask(req.session_id, req.question)
    # 打印 doc_ids（验收要求）
    doc_ids = [c["doc_id"] for c in result["docs"]]
    print("Retrieved doc_ids:", doc_ids)

    return result
# if __name__ == "__main__":
#     uvicorn.run(app)




