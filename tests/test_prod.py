"""Tests for the production layer: runner, idempotency, cache, audit, worker."""
import asyncio
import json
import os

for _k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY",
           "GOOGLE_API_KEY", "GEMINI_API_KEY", "REDIS_URL", "DATABASE_URL"):
    os.environ.pop(_k, None)

import pytest  # noqa: E402

from agentkit.prod.cache import ResponseCache, fingerprint  # noqa: E402
from agentkit.prod.idempotency import IdempotencyStore, idempotency_key  # noqa: E402
from agentkit.prod.runner import Pipeline  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "evals")


def load(name):
    return [json.loads(l) for l in open(os.path.join(DATA, name), encoding="utf-8") if l.strip()]


def test_idempotency_dedups_identical():
    store = IdempotencyStore()
    rec = load("sample_8613.jsonl")[0]
    assert store.is_duplicate(rec) is False
    assert store.is_duplicate(rec) is True


def test_idempotency_key_stable_and_distinct():
    recs = load("sample_8613.jsonl")
    assert idempotency_key(recs[0]) == idempotency_key(recs[0])
    assert idempotency_key(recs[0]) != idempotency_key(recs[1])


def test_cache_roundtrip():
    cache = ResponseCache()
    key = "k1"
    assert cache.get(key) is None
    cache.set(key, {"hello": "world"})
    assert cache.get(key) == {"hello": "world"}
    assert cache.hit_rate == 0.5  # one miss, one hit


def test_fingerprint_distinguishes_profiles():
    recs = load("sample_8613.jsonl")
    c = {"channel": "sms", "primary_cta": "book_tour"}
    assert fingerprint(recs[0], c) != fingerprint(recs[1], c)


def test_pipeline_batch_processes_all():
    recs = load("sample_8613.jsonl")
    pipe = Pipeline(concurrency=4, use_audit=False)
    report = asyncio.run(pipe.process_batch(recs))
    assert report["metrics"]["processed"] == 2
    assert report["metrics"]["errors"] == 0


def test_pipeline_dedups_within_batch():
    recs = load("sample_8613.jsonl")
    pipe = Pipeline(concurrency=4, use_audit=False)
    asyncio.run(pipe.process_batch(recs))          # first pass
    report = asyncio.run(pipe.process_batch(recs))  # second pass -> all dupes
    assert report["metrics"]["deduped"] >= 2


def test_pipeline_cache_hit_without_idempotency():
    recs = load("sample_8613.jsonl")
    pipe = Pipeline(concurrency=4, use_idempotency=False, use_audit=False)

    async def go():
        await pipe.process_batch(recs)
        return await asyncio.gather(*(pipe.process_one(r, recs) for r in recs))

    second = asyncio.run(go())
    assert all(r.get("cached") for r in second)


def test_pipeline_handles_bad_record_without_crash():
    pipe = Pipeline(concurrency=2, use_audit=False, use_idempotency=False)
    out = asyncio.run(pipe.process_one({"_raw_line": "{garbage"}))
    assert out["should_send"] is False


def test_audit_writes_file(tmp_path):
    os.environ["AUDIT_LOG_PATH"] = str(tmp_path / "audit.jsonl")
    from agentkit.prod.audit import AuditStore

    store = AuditStore()
    assert store.backend == "file"
    store.record({"task_id": "t1", "should_send": True,
                  "next_message": {"channel": "sms"}, "evaluation": {"score": "22/22"}})
    rows = (tmp_path / "audit.jsonl").read_text().strip().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["task_id"] == "t1"
    os.environ.pop("AUDIT_LOG_PATH", None)
