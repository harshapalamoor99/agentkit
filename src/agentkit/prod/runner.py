"""Concurrent async batch runner — the production processing core.

Processes records with bounded concurrency and wires in the production concerns:
structured logging, idempotency/dedup, response caching, durable audit, and
LangSmith run config. The same `process_one` coroutine is what a Kafka worker calls
per message (see worker.py); the runner just fans many of them out concurrently.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

from .. import channels
from ..graph import app
from . import tracing
from .audit import AuditStore
from .cache import ResponseCache, fingerprint
from .deadletter import DeadLetterQueue, is_retriable_abort
from .idempotency import IdempotencyStore
from .logging_config import get_logger, log_event

log = get_logger()


def _constraints_for_fingerprint(record: dict[str, Any]) -> dict[str, Any]:
    cfg = (record.get("assertions", {}) or {}).get("constraints", {}) or {}
    return {"channel": channels.select_channel(record),
            "primary_cta": cfg.get("primary_cta", "book_tour")}


class Pipeline:
    def __init__(self, *, concurrency: int = 16, use_cache: bool = True,
                 use_idempotency: bool = True, use_audit: bool = True,
                 use_deadletter: bool = True):
        self.sem = asyncio.Semaphore(concurrency)
        self.cache = ResponseCache() if use_cache else None
        self.idem = IdempotencyStore() if use_idempotency else None
        self.audit = AuditStore() if use_audit else None
        self.dlq = DeadLetterQueue() if use_deadletter else None
        self.processed = 0
        self.deduped = 0
        self.errors = 0
        self.sent = 0
        self.dead_lettered = 0
        log_event(log, "pipeline_init", concurrency=concurrency,
                  cache=self.cache.backend if self.cache else "off",
                  idempotency=self.idem.backend if self.idem else "off",
                  audit=self.audit.backend if self.audit else "off",
                  deadletter=self.dlq.backend if self.dlq else "off",
                  tracing=tracing.tracing_enabled())

    async def process_one(self, record: dict[str, Any], dataset: list[dict] | None = None) -> dict[str, Any]:
        task_id = record.get("task_id", "unknown")
        t0 = time.perf_counter()

        if self.idem and self.idem.is_duplicate(record):
            self.deduped += 1
            log_event(log, "deduped", task_id=task_id)
            return {"task_id": task_id, "status": "deduplicated", "should_send": False}

        key = fingerprint(record, _constraints_for_fingerprint(record)) if self.cache else None
        if self.cache and key:
            cached = self.cache.get(key)
            if cached is not None:
                log_event(log, "cache_hit", task_id=task_id,
                          latency_ms=round((time.perf_counter() - t0) * 1000, 1))
                return {**cached, "cached": True}

        async with self.sem:
            try:
                init = {"record": record, "dataset": dataset or [],
                        "task_id": task_id,
                        "raw_line": json.dumps(record, ensure_ascii=False)}
                state = await app.ainvoke(init, config=tracing.run_config(task_id))
                out = state.get("final_output", {"task_id": task_id, "error": "no output"})
            except Exception as exc:  # never let one bad record kill the worker
                self.errors += 1
                log_event(log, "process_error", task_id=task_id, error=repr(exc))
                return {"task_id": task_id, "status": "error", "should_send": False,
                        "error": repr(exc)}

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.processed += 1

        if self.cache and key:
            self.cache.set(key, out)
        if self.audit:
            try:
                self.audit.record(out)
            except Exception as exc:
                log_event(log, "audit_error", task_id=task_id, error=repr(exc))

        ev = out.get("evaluation", {}) or {}
        if out.get("should_send"):
            self.sent += 1
        # G5: route retriable (transient/infra) aborts to the dead-letter queue so they
        # can be replayed once the backend recovers, instead of being silently dropped.
        abort_reason = out.get("abort_reason")
        if self.dlq and is_retriable_abort(abort_reason):
            try:
                self.dlq.enqueue(record, abort_reason)
                self.dead_lettered += 1
                log_event(log, "dead_lettered", task_id=task_id, reason=abort_reason)
            except Exception as exc:
                log_event(log, "deadletter_error", task_id=task_id, error=repr(exc))

        log_event(log, "node_complete", node="pipeline", task_id=task_id,
                  latency_ms=latency_ms, should_send=out.get("should_send"),
                  ac_score=ev.get("score"), critical_fail=bool(ev.get("critical_fails")),
                  used_fallback=out.get("used_fallback"))
        out["cached"] = False
        return out

    async def process_batch(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(self.process_one(r, records) for r in records), return_exceptions=False
        )
        wall_s = time.perf_counter() - t0
        eligible = self.processed  # records that actually ran the pipeline
        metrics = {
            "records": len(records),
            "processed": self.processed,
            "deduped": self.deduped,
            "errors": self.errors,
            "sent": self.sent,
            "dead_lettered": self.dead_lettered,
            # G5: send-success SLO — fraction of processed records that actually sent.
            # A low value means the backend is degrading send availability even while
            # latency stays green (records aborting to safe no-send).
            "send_success_rate": round(self.sent / eligible, 3) if eligible else None,
            "wall_s": round(wall_s, 3),
            "throughput_rps": round(len(records) / wall_s, 1) if wall_s else 0.0,
            "cache_hit_rate": self.cache.hit_rate if self.cache else None,
        }
        log_event(log, "batch_complete", **metrics)
        return {"metrics": metrics, "results": results}

    async def replay_dead_letters(self) -> dict[str, Any]:
        """Re-process everything currently in the DLQ (G5)."""
        if not self.dlq:
            return {"replayed": 0, "recovered": 0, "still_failing": 0}
        return await self.dlq.replay(lambda r: self.process_one(r))


def run_file(path: str, concurrency: int = 16) -> dict[str, Any]:
    from pathlib import Path

    records = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    pipe = Pipeline(concurrency=concurrency)
    return asyncio.run(pipe.process_batch(records))


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Concurrent production runner")
    ap.add_argument("input")
    ap.add_argument("--concurrency", type=int, default=int(os.getenv("CONCURRENCY", "16")))
    ap.add_argument("--out")
    args = ap.parse_args()

    report = run_file(args.input, args.concurrency)
    if args.out:
        from pathlib import Path
        Path(args.out).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in report["results"]) + "\n",
            encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
