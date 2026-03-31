import json

import redis

TTL = 86400

class RedisMemory:
    def __init__(self, host='localhost', port=6379, ttl=86400):
        self.r = redis.Redis(
            host=host,
            port=port,
            decode_responses=True)
        self.ttl = ttl

    def append_message(self, session_id: str, message: dict):
        key = f"app:memory:chat:{session_id}"
        # self.r.rpush(key, json.dumps(message))
        # self.r.ltrim(key, -20, -1)
        # self.r.expire(key, self.ttl)

        pipe = self.r.pipeline()
        pipe.rpush(key, json.dumps(message))
        pipe.ltrim(key, -20, -1)
        pipe.expire(key, self.ttl)
        pipe.execute()

    def get_memory(self, session_id: str) -> list[dict]:
        key = f"app:memory:chat:{session_id}"
        data = self.r.lrange(key, 0, -1)
        return [json.loads(x) for x in data]


r = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True,
)

def append_message(session_id: str, message: dict):
    key = f"chat:{session_id}"
    data = r.get(key)
    if data:
        messages = json.loads(data)
    else:
        messages = []
    messages.append(message)
    r.set(key, json.dumps(messages), ex=TTL)

def get_memory(session_id: str):

    key = f"chat:{session_id}"

    data = r.get(key)

    if not data:
        return []

    return json.loads(data)

def append_message2(session_id: str, message: dict):
    key = f"chat:{session_id}"
    r.rpush(key, json.dumps(message))
    r.ltrim(key, -20, -1)
    r.expire(key, TTL)

def get_memory2(session_id: str):
    ket = f"chat:{session_id}"
    data = r.lrange(ket, 0, -1)
    return [json.loads(x) for x in data]