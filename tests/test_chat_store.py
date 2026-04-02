from redis.exceptions import RedisError

from app.memory.chat_store import ChatStoreConfig, HybridChatStore


class _BrokenRedisStore:
    def memory_key(self, session_id: str) -> str:
        return f"x:{session_id}"

    @property
    def client(self):
        class _Dummy:
            def ping(self):
                raise RedisError("down")

        return _Dummy()

    def append_message(self, session_id: str, message: dict[str, str]) -> None:
        raise RedisError("down")

    def get_memory(self, session_id: str):
        raise RedisError("down")

    def delete_session(self, session_id: str) -> None:
        raise RedisError("down")


def test_hybrid_chat_store_falls_back_to_memory_when_redis_unavailable():
    store = HybridChatStore(ChatStoreConfig())
    store.redis_store = _BrokenRedisStore()

    store.append_message("s1", {"role": "user", "content": "hello"})
    history = store.get_memory("s1")

    assert history == [{"role": "user", "content": "hello"}]

    store.delete_session("s1")
    assert store.get_memory("s1") == []
