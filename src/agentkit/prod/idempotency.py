"""Idempotency / dedup store.

In prod, the same message request can be delivered more than once (Kafka at-least-once,
retried API calls). We dedupe on a deterministic key so a prospect is never messaged
twice for the same logical request.

Backed by Redis when `REDIS_URL` is set (shared across workers); otherwise an in-process
store (fine for a single worker / tests). Same interface either way.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any


def idempotency_key(record: dict[str, Any]) -> str:
    """Stable key from the parts of the record that define a unique send intent."""
    from .. import tenants  # lazy: avoid import cycle
    inp = record.get("input", {}) or {}
    basis = {
        "tenant_id": tenants._tenant_id_for_record(record),
        "task_id": record.get("task_id"),
        "property": inp.get("property_name"),
        "last_interaction": inp.get("last_interaction"),
        "first_name": (inp.get("profile", {}) or {}).get("first_name"),
    }
    raw = json.dumps(basis, sort_keys=True, ensure_ascii=False)
    # "msg2:" — prefix bump from "msg:" so tenant-less keys from a prior deploy don't collide.
    return "msg2:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class _MemoryStore:
    def __init__(self, ttl_s: int):
        self._ttl = ttl_s
        self._data: dict[str, float] = {}
        self._lock = threading.Lock()

    def seen(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            exp = self._data.get(key)
            if exp and exp > now:
                return True
            self._data[key] = now + self._ttl
            # opportunistic cleanup
            for k, v in list(self._data.items()):
                if v <= now:
                    self._data.pop(k, None)
            return False


class _RedisStore:
    def __init__(self, url: str, ttl_s: int):
        import redis  # lazy

        self._r = redis.Redis.from_url(url)
        self._ttl = ttl_s

    def seen(self, key: str) -> bool:
        # SET NX returns True only on first insert.
        created = self._r.set(key, "1", nx=True, ex=self._ttl)
        return not bool(created)


class IdempotencyStore:
    def __init__(self, ttl_s: int = 86_400):
        url = os.getenv("REDIS_URL")
        self.backend = "memory"
        self._impl: Any = _MemoryStore(ttl_s)
        if url:
            try:
                self._impl = _RedisStore(url, ttl_s)
                self._impl._r.ping()
                self.backend = "redis"
            except Exception:
                self._impl = _MemoryStore(ttl_s)

    def is_duplicate(self, record: dict[str, Any]) -> bool:
        """True if this exact send intent has been processed already."""
        return self._impl.seen(idempotency_key(record))
