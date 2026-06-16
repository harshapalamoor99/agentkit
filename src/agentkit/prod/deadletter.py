"""Dead-letter queue + replay for safe-aborted records (G5).

The agent guarantees p99 < 2s by aborting to a safe no-send when the LLM is slow,
erroring, rate-limited, or unconfigured. That protects latency, but a record that
*should* have been sent is then silently dropped. This module captures those records so
they are observable and re-processable once the backend recovers.

Only *retriable* aborts are dead-lettered — transient backend failures (timeout, circuit
open, budget exhausted, API error) and a missing provider. Business no-sends
(NO_CONSENT) and permanently bad input (BAD_RECORD) are NOT retriable and never enqueued.

Transport: JSONL file by default (`DLQ_LOG_PATH`, default ./deadletter.jsonl); Kafka when
`KAFKA_BOOTSTRAP_SERVERS` + `KAFKA_DLQ_TOPIC` are set — same graceful-degradation pattern
as audit/telemetry.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

# Abort reasons that represent a transient/infra failure worth retrying. A business
# no-send (NO_CONSENT) or unparseable input (BAD_RECORD) is terminal, not dead-lettered.
RETRIABLE_ABORT_REASONS = {
    "LLM_UNAVAILABLE", "LLM_TIMEOUT", "LLM_BUDGET_EXHAUSTED",
    "LLM_CIRCUIT_OPEN", "LLM_RETRIES_EXHAUSTED",
}


def is_retriable_abort(reason: str | None) -> bool:
    if not reason:
        return False
    return reason in RETRIABLE_ABORT_REASONS or reason.startswith("LLM_ERROR")


class _FileDLQ:
    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, entry: dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def rewrite(self, entries: list[dict[str, Any]]) -> None:
        with self._lock:
            self.path.write_text(
                "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
                encoding="utf-8")


class _KafkaDLQ:
    def __init__(self, servers: str, topic: str):
        from kafka import KafkaProducer  # lazy

        self._topic = topic
        self._producer = KafkaProducer(
            bootstrap_servers=servers.split(","),
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        )

    def append(self, entry: dict[str, Any]) -> None:
        self._producer.send(self._topic, entry)
        self._producer.flush()

    def read_all(self) -> list[dict[str, Any]]:  # pragma: no cover - infra path
        raise NotImplementedError("Kafka DLQ replay is driven by a consumer, not read_all")

    def rewrite(self, entries: list[dict[str, Any]]) -> None:  # pragma: no cover
        raise NotImplementedError("Kafka DLQ is append-only")


class DeadLetterQueue:
    def __init__(self, path: str | None = None):
        servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        topic = os.getenv("KAFKA_DLQ_TOPIC")
        self.backend = "file"
        self._impl: Any = _FileDLQ(path or os.getenv("DLQ_LOG_PATH", "deadletter.jsonl"))
        if servers and topic:
            try:
                self._impl = _KafkaDLQ(servers, topic)
                self.backend = "kafka"
            except Exception:
                pass

    def enqueue(self, record: dict[str, Any], reason: str,
                metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        entry = {
            "task_id": record.get("task_id"),
            "record": record,
            "abort_reason": reason,
            "attempts": int((metadata or {}).get("attempts", 0)) + 1,
            "enqueued_at": datetime.now(timezone.utc).isoformat(),
            "metadata": {k: v for k, v in (metadata or {}).items() if k != "attempts"},
        }
        self._impl.append(entry)
        return entry

    def pending(self) -> list[dict[str, Any]]:
        return self._impl.read_all()

    async def replay(self, process: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]
                     ) -> dict[str, Any]:
        """Re-run each pending entry through `process` (e.g. Pipeline.process_one).

        Entries that now succeed (a real send, or a terminal business no-send) are
        cleared; entries that abort again with a retriable reason are kept (with an
        incremented attempt count) for a future replay. Returns a summary.
        """
        entries = self._impl.read_all()
        if not entries:
            return {"replayed": 0, "recovered": 0, "still_failing": 0}

        remaining: list[dict[str, Any]] = []
        recovered = 0
        for entry in entries:
            out = await process(entry["record"])
            reason = out.get("abort_reason")
            if is_retriable_abort(reason):
                entry["attempts"] = int(entry.get("attempts", 1)) + 1
                entry["last_reason"] = reason
                remaining.append(entry)
            else:
                recovered += 1  # sent, or now a terminal/legitimate outcome
        self._impl.rewrite(remaining)
        return {"replayed": len(entries), "recovered": recovered,
                "still_failing": len(remaining)}


# Process-level DLQ.
store = DeadLetterQueue()
