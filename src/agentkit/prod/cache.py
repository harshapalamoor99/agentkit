"""Response cache.

Caches the agent's parsed output keyed by a profile/constraint fingerprint, so
repeated or near-identical requests skip the LLM call. Redis-backed when `REDIS_URL`
is set (shared, TTL'd); otherwise an in-process LRU-ish dict for single-worker/tests.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any


def fingerprint(record: dict[str, Any], constraints: dict[str, Any]) -> str:
    from .. import tenants  # lazy: avoid import cycle
    inp = record.get("input", {}) or {}
    profile = inp.get("profile", {}) or {}
    basis = {
        "tenant_id": tenants._tenant_id_for_record(record),
        "channel": constraints.get("channel"),
        "primary_cta": constraints.get("primary_cta"),
        "first_name": profile.get("first_name"),
        "amenity_interest": profile.get("amenity_interest"),
        "city_interest": profile.get("city_interest"),
        "property": inp.get("property_name"),
        "move_date_target": inp.get("move_date_target"),
    }
    raw = json.dumps(basis, sort_keys=True, ensure_ascii=False)
    # "llmout2:" — prefix bump from "llmout:" so tenant-less keys from a prior deploy
    # can't serve one tenant's cached output to another.
    return "llmout2:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class _MemoryCache:
    def __init__(self, ttl_s: int, maxsize: int = 1024):
        self.ttl, self.maxsize = ttl_s, maxsize
        self._d: dict[str, tuple[float, str]] = {}

    def get(self, key: str) -> str | None:
        v = self._d.get(key)
        if v and v[0] > time.time():
            return v[1]
        self._d.pop(key, None)
        return None

    def set(self, key: str, value: str) -> None:
        if len(self._d) >= self.maxsize:
            self._d.pop(next(iter(self._d)), None)
        self._d[key] = (time.time() + self.ttl, value)


class _RedisCache:
    def __init__(self, url: str, ttl_s: int):
        import redis

        self._r = redis.Redis.from_url(url)
        self.ttl = ttl_s

    def get(self, key: str) -> str | None:
        v = self._r.get(key)
        return v.decode() if v else None

    def set(self, key: str, value: str) -> None:
        self._r.setex(key, self.ttl, value)


class ResponseCache:
    def __init__(self, ttl_s: int = 900):
        url = os.getenv("REDIS_URL")
        self.backend = "memory"
        self._impl: Any = _MemoryCache(ttl_s)
        self.hits = 0
        self.misses = 0
        if url:
            try:
                self._impl = _RedisCache(url, ttl_s)
                self._impl._r.ping()
                self.backend = "redis"
            except Exception:
                self._impl = _MemoryCache(ttl_s)

    def get(self, key: str) -> dict | None:
        raw = self._impl.get(key)
        if raw is None:
            self.misses += 1
            return None
        self.hits += 1
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None

    def set(self, key: str, output: dict) -> None:
        self._impl.set(key, json.dumps(output, ensure_ascii=False))

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 3) if total else 0.0
