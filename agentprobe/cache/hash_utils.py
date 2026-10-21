"""Hashing utilities for function content caching."""

from __future__ import annotations

import hashlib


def function_hash(source: str) -> str:
    """Return SHA-256 hex digest of function source code."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def cache_key(prefix: str, content_hash: str) -> str:
    """Build a namespaced cache key."""
    return f"{prefix}:{content_hash}"
