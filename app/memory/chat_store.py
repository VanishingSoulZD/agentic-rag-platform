from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


@dataclass
class ChatStoreConfig:
    redis_url: str = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
    key_prefix: str = "chat:memory:"
    ttl_seconds: int = 24 * 60 * 60
    max_messages: int = 100


class InMemoryChatStore:
    def __init__(self, max_messages: int = 100):
        self.max_messages = max_messages
        self._data: dict[str, list[dict[str, str]]] = {}

    def append_message(self, session_id: str, message: dict[str, str]) -> None:
        history = self._data.setdefault(session_id, [])
        history.append(message)
        if len(history) > self.max_messages:
            self._data[session_id] = history[-self.max_messages:]

    def get_memory(self, session_id: str) -> list[dict[str, str]]:
        return list(self._data.get(session_id, []))

    def delete_session(self, session_id: str) -> None:
        self._data.pop(session_id, None)


class RedisChatStore:
    def __init__(self, config: ChatStoreConfig):
        self.config = config
        self.client = redis.from_url(config.redis_url, decode_responses=True)

    def memory_key(self, session_id: str) -> str:
        return f"{self.config.key_prefix}{session_id}"

    def append_message(self, session_id: str, message: dict[str, str]) -> None:
        key = self.memory_key(session_id)
        pipeline = self.client.pipeline()
        pipeline.rpush(key, json.dumps(message, ensure_ascii=False))
        pipeline.ltrim(key, -self.config.max_messages, -1)
        pipeline.expire(key, self.config.ttl_seconds)
        pipeline.execute()

    def get_memory(self, session_id: str) -> list[dict[str, str]]:
        key = self.memory_key(session_id)
        raw_messages = self.client.lrange(key, 0, -1)
        return [json.loads(item) for item in raw_messages]

    def delete_session(self, session_id: str) -> None:
        self.client.delete(self.memory_key(session_id))


class HybridChatStore:
    """Use Redis by default; fallback to in-memory when Redis is unavailable."""

    def __init__(self, config: ChatStoreConfig):
        self.config = config
        self.redis_store = RedisChatStore(config)
        self.memory_store = InMemoryChatStore(max_messages=config.max_messages)
        self._prefer_memory = False

    def memory_key(self, session_id: str) -> str:
        return self.redis_store.memory_key(session_id)

    def get_redis_client(self) -> redis.Redis:
        return self.redis_store.client

    @property
    def using_memory_fallback(self) -> bool:
        return self._prefer_memory

    def is_redis_available(self) -> bool:
        try:
            self.redis_store.client.ping()
            return True
        except RedisError:
            return False

    def _use_memory_fallback(self, op_name: str, err: Exception) -> None:
        if not self._prefer_memory:
            logger.warning("redis_unavailable op=%s err=%s; fallback=in-memory", op_name, repr(err))
        self._prefer_memory = True

    def append_message(self, session_id: str, message: dict[str, str]) -> None:
        if self._prefer_memory:
            self.memory_store.append_message(session_id, message)
            return
        try:
            self.redis_store.append_message(session_id, message)
        except RedisError as err:
            self._use_memory_fallback("append_message", err)
            self.memory_store.append_message(session_id, message)

    def get_memory(self, session_id: str) -> list[dict[str, str]]:
        if self._prefer_memory:
            return self.memory_store.get_memory(session_id)
        try:
            return self.redis_store.get_memory(session_id)
        except RedisError as err:
            self._use_memory_fallback("get_memory", err)
            return self.memory_store.get_memory(session_id)

    def delete_session(self, session_id: str) -> None:
        if self._prefer_memory:
            self.memory_store.delete_session(session_id)
            return
        try:
            self.redis_store.delete_session(session_id)
        except RedisError as err:
            self._use_memory_fallback("delete_session", err)
            self.memory_store.delete_session(session_id)
