"""Async-safe TTL cache used by tool handlers to avoid hammering Garmin."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any


class TTLCache:
    """Tiny in-memory cache with per-entry TTLs.

    Keys are strings; values can be anything. Lookups are O(1). The cache is
    intentionally simple: no LRU eviction, no max size. For a single-user
    MCP server the working set is small, and the process is restarted on
    every Cloud Run cold start anyway.
    """

    def __init__(self) -> None:
        self._data: dict[str, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def make_key(name: str, args: dict[str, Any] | None = None) -> str:
        """Build a stable cache key from a tool name and its arguments."""
        args_json = json.dumps(args, sort_keys=True, default=str) if args else ""
        digest = hashlib.sha256(f"{name}:{args_json}".encode()).hexdigest()[:16]
        return f"{name}:{digest}"

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            expires, value = entry
            if expires <= time.monotonic():
                del self._data[key]
                return None
            return value

    async def set(self, key: str, value: Any, ttl_seconds: float) -> None:
        async with self._lock:
            self._data[key] = (time.monotonic() + ttl_seconds, value)

    async def get_or_compute(
        self,
        key: str,
        ttl_seconds: float,
        compute: Callable[[], Awaitable[Any]],
        incomplete: Callable[[Any], bool] | None = None,
        incomplete_ttl_seconds: float = 60,
    ) -> Any:
        """Return cached value or compute and store it.

        ``incomplete`` flags a result that Garmin has not finished producing yet
        (e.g. last night's HRV before the watch synced). Such a result is kept
        only ``incomplete_ttl_seconds`` so the next call asks Garmin again
        instead of serving the empty answer for the full TTL.

        Note: two concurrent callers with the same key may both miss and both
        call ``compute``. That is fine for our workload because Garmin calls
        are idempotent reads and the rate limiter serializes them anyway.
        """
        cached = await self.get(key)
        if cached is not None:
            return cached
        value = await compute()
        ttl = ttl_seconds
        if incomplete is not None:
            try:
                if incomplete(value):
                    ttl = min(ttl_seconds, incomplete_ttl_seconds)
            except Exception:  # a bad predicate must never break the tool
                ttl = min(ttl_seconds, incomplete_ttl_seconds)
        await self.set(key, value, ttl)
        return value

    async def clear(self) -> None:
        async with self._lock:
            self._data.clear()

    async def size(self) -> int:
        async with self._lock:
            now = time.monotonic()
            return sum(1 for expires, _ in self._data.values() if expires > now)
