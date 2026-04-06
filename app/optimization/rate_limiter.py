from __future__ import annotations

import time
from collections import defaultdict, deque


class SessionRateLimiter:
    def __init__(self, limit: int = 20, window_seconds: int = 60):
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, session_id: str) -> tuple[bool, int]:
        now = time.time()
        q = self._events[session_id]

        while q and q[0] <= now - self.window_seconds:
            q.popleft()

        if len(q) >= self.limit:
            retry_after = max(1, int(self.window_seconds - (now - q[0])))
            return False, retry_after

        q.append(now)
        return True, 0
