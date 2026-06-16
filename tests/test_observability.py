"""Tests for token/cost accounting and OpenTelemetry node instrumentation.

Tracing degrades to a no-op when opentelemetry isn't installed/enabled, so these tests
assert the *wiring* (spans never break a node, cost flows into lineage/output) rather
than requiring a real OTLP backend.
"""
import asyncio
import json

import pytest

from messaging_agent import cost, observability


# --- cost accounting ---

def test_estimate_cost_known_model():
    out = cost.estimate_cost({
        "model": "gpt-4o-mini", "prompt_tokens": 1_000_000,
        "completion_tokens": 1_000_000, "total_tokens": 2_000_000,
    })
    assert out["priced"] is True
    assert out["input_cost_usd"] == pytest.approx(0.15)
    assert out["output_cost_usd"] == pytest.approx(0.60)
    assert out["cost_usd"] == pytest.approx(0.75)


def test_estimate_cost_unknown_model_is_zero_not_error():
    out = cost.estimate_cost({"model": "totally-made-up", "prompt_tokens": 500})
    assert out["priced"] is False
    assert out["cost_usd"] == 0.0


def test_estimate_cost_matches_dated_model_suffix():
    out = cost.estimate_cost({
        "model": "claude-3-5-sonnet-20241022",
        "prompt_tokens": 1_000_000, "completion_tokens": 0,
    })
    assert out["priced"] is True
    assert out["input_cost_usd"] == pytest.approx(3.00)


def test_accumulate_sums_across_calls():
    running = None
    running = cost.accumulate(running, {"model": "gpt-4o-mini",
                                        "prompt_tokens": 100, "completion_tokens": 50})
    running = cost.accumulate(running, {"model": "gpt-4o-mini",
                                        "prompt_tokens": 200, "completion_tokens": 80})
    assert running["calls"] == 2
    assert running["prompt_tokens"] == 300
    assert running["completion_tokens"] == 130
    assert running["cost_usd"] > 0


def test_accumulate_none_usage_starts_empty():
    running = cost.accumulate(None, None)
    assert running["calls"] == 0
    assert running["cost_usd"] == 0.0


def test_price_table_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PRICE_TABLE", json.dumps({"my-model": {"input": 1.0, "output": 2.0}}))
    cost.reset_price_cache()
    try:
        out = cost.estimate_cost({"model": "my-model",
                                  "prompt_tokens": 1_000_000, "completion_tokens": 1_000_000})
        assert out["cost_usd"] == pytest.approx(3.0)
    finally:
        monkeypatch.delenv("LLM_PRICE_TABLE", raising=False)
        cost.reset_price_cache()


# --- observability wiring ---

def test_span_noop_by_default_is_safe():
    with observability.span("test", {"k": "v"}) as sp:
        sp.set_attribute("a", 1)  # must not raise even as no-op


def test_instrument_sync_node_is_transparent():
    def node(state):
        return {**state, "marker": "ran"}

    wrapped = observability.instrument_node("demo", node)
    out = wrapped({"task_id": "t1"})
    assert out["marker"] == "ran"
    assert out["task_id"] == "t1"


def test_instrument_async_node_is_transparent():
    async def node(state):
        return {**state, "marker": "async-ran"}

    wrapped = observability.instrument_node("demo_async", node)
    out = asyncio.run(wrapped({"task_id": "t2"}))
    assert out["marker"] == "async-ran"


def test_instrument_node_reraises_and_records():
    def boom(state):
        raise ValueError("kaboom")

    wrapped = observability.instrument_node("boom", boom)
    with pytest.raises(ValueError, match="kaboom"):
        wrapped({"task_id": "t3"})


def test_status_reports_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MESSAGING_AGENT_TRACING", raising=False)
    st = observability.status()
    assert st["tracing_enabled"] is False


# --- end-to-end: cost flows into lineage + emitted output ---

class _UsageLLM:
    provider = "mock"
    available = True
    last_usage = {"model": "gpt-4o-mini", "prompt_tokens": 1000,
                  "completion_tokens": 500, "total_tokens": 1500}

    async def warmup(self):
        return None

    async def generate(self, system, user, **kwargs):
        return json.dumps({
            "should_send": True,
            "next_message": {"channel": "email", "subject": "Hi", "body": "Hello there"},
            "next_action": {"type": "no_action"},
        })


@pytest.fixture
def _usage_llm():
    import messaging_agent.nodes.llm as llmnode
    original = llmnode._client
    llmnode._client = _UsageLLM()
    yield
    llmnode._client = original


def test_cost_appears_in_final_output(_usage_llm):
    from messaging_agent.graph import build_graph

    record = {
        "task_id": "cost-1", "domain": "support",
        "consent": {"email_opt_in": True},
        "channel_preferences": ["email"],
        "input": {"ticket_subject": "Help", "profile": {"first_name": "Jo"}},
        "assertions": {"constraints": {"primary_cta": "resolve_ticket"}},
    }
    app = build_graph()
    state = asyncio.run(app.ainvoke({"task_id": "cost-1", "record": record, "dataset": [record]}))
    final = state["final_output"]
    assert final["cost"] is not None
    assert final["cost"]["total_tokens"] == 1500
    assert final["cost"]["cost_usd"] > 0
    assert final["lineage"]["cost"]["cost_usd"] == final["cost"]["cost_usd"]
