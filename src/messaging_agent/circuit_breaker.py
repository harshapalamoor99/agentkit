"""LLM circuit breaker (AC-11).

Protects the orchestration pipeline from a degraded LLM backend (sustained 5xx, timeouts,
rate-limits). After CIRCUIT_BREAKER_THRESHOLD consecutive failures the breaker OPENS and
subsequent calls fast-abort (return immediately) instead of waiting on a dead backend.
After CIRCUIT_BREAKER_COOLDOWN_S it goes HALF_OPEN to probe a single request; a success
closes it, a failure re-opens it.

Because the agent is LLM-only there is no template fallback — an open breaker simply means
the record aborts to a safe no-send fast, keeping the pipeline responsive and well under
the latency SLA rather than hammering a failing dependency.
"""
from __future__ import annotations

import threading
import time

from . import config

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, threshold: int | None = None, cooldown_s: float | None = None):
        self.threshold = threshold if threshold is not None else config.CIRCUIT_BREAKER_THRESHOLD
        self.cooldown_s = cooldown_s if cooldown_s is not None else config.CIRCUIT_BREAKER_COOLDOWN_S
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        if self._opened_at is None:
            return CLOSED
        if (time.monotonic() - self._opened_at) >= self.cooldown_s:
            return HALF_OPEN
        return OPEN

    def allow(self) -> bool:
        """True if a call may proceed. Open breaker (within cooldown) blocks; half-open
        lets a single probe through."""
        return self.state != OPEN

    def record_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.threshold:
                self._opened_at = time.monotonic()

    def snapshot(self) -> dict:
        return {"state": self.state, "consecutive_failures": self._consecutive_failures,
                "threshold": self.threshold}


# --- Per-key registry (G8) --------------------------------------------------
# A single process can serve multiple tenants/backends. A degraded backend for one
# (provider, tenant) must NOT trip the breaker for everyone, so breakers are isolated
# by key. The bare module-level `breaker` remains as the default/shared instance.
_registry: dict[tuple, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_breaker(provider: str | None = None, tenant_id: str | None = None) -> "CircuitBreaker":
    """Return the breaker for a (provider, tenant) pair, creating it on first use.

    With no key this returns the shared default breaker (backward compatible)."""
    if provider is None and tenant_id is None:
        return breaker
    key = (provider, tenant_id)
    with _registry_lock:
        b = _registry.get(key)
        if b is None:
            b = CircuitBreaker()
            _registry[key] = b
        return b


def reset_all() -> None:
    """Close the default breaker and drop all per-key breakers (used by tests)."""
    breaker.record_success()
    with _registry_lock:
        _registry.clear()


# Module-level default breaker (shared / used when no key is supplied).
breaker = CircuitBreaker()
