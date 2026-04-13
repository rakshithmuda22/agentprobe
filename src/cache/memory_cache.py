"""In-memory cache with TTL support."""

from __future__ import annotations

import time
from typing import Any


class MemoryCache:
    """Simple dict-based cache with per-key TTL expiration."""

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}  # key -> (value, expires_at)

    def get(self, key: str) -> Any | None:
        """Return cached value or None if missing/expired."""
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at and time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int = 0) -> None:
        """Store a value with optional TTL in seconds (0 = no expiry)."""
        expires_at = (time.time() + ttl) if ttl > 0 else 0.0
        self._store[key] = (value, expires_at)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
