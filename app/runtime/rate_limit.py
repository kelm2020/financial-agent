from __future__ import annotations

import asyncio
import time
from collections import deque


class SlidingWindowRateLimiter:
    """Small process-local boundary limiter; Redis replacement is explicitly F5."""

    def __init__(self, *, limit: int, window_seconds: float) -> None:
        if limit < 1 or window_seconds <= 0:
            raise ValueError("Rate limit and window must be positive")
        self._limit = limit
        self._window = window_seconds
        self._entries: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def allow(self, key: str) -> bool:
        now = time.monotonic()
        async with self._lock:
            timestamps = self._entries.setdefault(key, deque())
            while timestamps and now - timestamps[0] >= self._window:
                timestamps.popleft()
            if len(timestamps) >= self._limit:
                return False
            timestamps.append(now)
            return True
