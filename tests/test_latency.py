"""Latency-budget regression tests (LLM-only).

Guarantees the end-to-end request stays under the 2s SLA on every path. Because the
agent is LLM-only, a slow / unparseable model no longer falls back to a fabricated
message — it aborts safely (should_send=False) within the budget. p99 < 2s is enforced
by the shared deadline.
"""
import asyncio
import json
import os
import time

for _k in ("LITELLM_API_KEY", "LITELLM_API_BASE", "LITELLM_MODEL",
           "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY",
           "GOOGLE_API_KEY", "GEMINI_API_KEY"):
    os.environ.pop(_k, None)

import agentkit.nodes.llm as llmnode  # noqa: E402
from agentkit.graph import app  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "evals")
SLA_MS = 2000


def _rec():
    line = next(l for l in open(os.path.join(DATA, "sample_8613.jsonl")) if l.strip())
    return json.loads(line)


def _run(record):
    init = {"record": record, "dataset": [record],
            "task_id": record["task_id"], "raw_line": json.dumps(record)}
    t = time.perf_counter()
    state = asyncio.run(app.ainvoke(init))
    return (time.perf_counter() - t) * 1000, state["final_output"]


def test_fast_valid_llm_sends_under_sla():
    class Fast:
        provider = "mock"
        available = True
        async def generate(self, system, user, **k):
            await asyncio.sleep(0.05)
            return json.dumps({"should_send": True, "next_message": {
                "channel": "sms", "subject": None,
                "body": "Hi Taylor—welcome to Oak Ridge! Tour our Richardson community this week. Reply STOP to opt out.",
                "cta": {"type": "schedule_tour"}},
                "next_action": {"type": "start_cadence", "name": "x"}})
    llmnode._client = Fast()
    ms, out = _run(_rec())
    assert ms < SLA_MS
    assert out["should_send"] is True
    assert out["used_fallback"] is False
    assert not out["evaluation"]["critical_fails"]


def test_slow_llm_aborts_under_sla():
    class Slow:
        provider = "mock"
        available = True
        async def generate(self, *a, **k):
            await asyncio.sleep(3.0)
            return "{}"
    llmnode._client = Slow()
    ms, out = _run(_rec())
    assert ms < SLA_MS, f"{ms}ms exceeded SLA"
    assert out["should_send"] is False
    assert out.get("abort_reason")
    assert not out["evaluation"]["critical_fails"]


def test_retry_loop_stays_under_sla():
    """Unparseable + slow responses force the retry loop; total must stay < 2s, then abort."""
    class BadSlow:
        provider = "mock"
        available = True
        async def generate(self, *a, **k):
            await asyncio.sleep(1.7)
            return "definitely not json"
    llmnode._client = BadSlow()
    ms, out = _run(_rec())
    assert ms < SLA_MS, f"retry path {ms}ms exceeded SLA"
    assert out["should_send"] is False
    assert not out["evaluation"]["critical_fails"]


def test_no_provider_aborts_immediately():
    class NoProvider:
        provider = None
        available = False
        async def generate(self, *a, **k):
            raise AssertionError("should not be called")
    llmnode._client = NoProvider()
    ms, out = _run(_rec())
    assert ms < SLA_MS
    assert out["should_send"] is False
    assert out.get("abort_reason") == "LLM_UNAVAILABLE"
