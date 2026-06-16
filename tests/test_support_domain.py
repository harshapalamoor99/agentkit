"""Proves the engine is domain-agnostic: run the full graph under a second domain.

The same compiled LangGraph ``app`` (intake -> safety -> sanitize -> context -> llm ->
parse -> evaluate -> emit) drives the unrelated ``support`` domain with no core changes —
only a :class:`Domain` subclass and a per-record ``"domain": "support"`` selector.
"""
import asyncio

import pytest

from messaging_agent import domain as domain_mod
from messaging_agent.graph import app


class _SupportMockLLM:
    """Returns a valid support follow-up message, mimicking a good model."""
    provider = "mock"
    available = True

    async def warmup(self):
        return None

    async def generate(self, system, user, **kwargs):
        import json
        return json.dumps({
            "should_send": True,
            "next_message": {
                "channel": "email",
                "subject": "Re: your ticket",
                "body": "Hi Sam, just following up on your login issue — is it resolved now?",
                "cta": {"type": "resolve_ticket"},
            },
            "next_action": {"type": "follow_up_in_days", "value": 2},
        })


@pytest.fixture
def _support_llm():
    import messaging_agent.nodes.llm as llmnode
    original = llmnode._client
    llmnode._client = _SupportMockLLM()
    yield
    llmnode._client = original


def _run(record):
    init = {"task_id": record.get("task_id"), "record": record, "dataset": [record]}
    return asyncio.run(app.ainvoke(init))["final_output"]


SUPPORT_RECORD = {
    "task_id": "sup-1",
    "domain": "support",
    "consent": {"email_opt_in": True, "push_opt_in": False},
    "channel_preferences": ["email", "push"],
    "input": {
        "ticket_subject": "Login issue",
        "profile": {"first_name": "Sam"},
    },
    "assertions": {"constraints": {"primary_cta": "resolve_ticket"}},
}


def test_support_domain_is_registered():
    assert "support" in domain_mod.available_domains()
    assert domain_mod.get_domain({"domain": "support"}).name == "support"


def test_support_domain_runs_through_the_same_graph(_support_llm):
    out = _run(SUPPORT_RECORD)
    assert out["should_send"] is True
    msg = out["next_message"]
    assert msg["channel"] == "email"  # the only consented channel
    assert "unsubscribe" in msg["body"].lower()  # domain opt-out repair applied
    assert msg["cta"]["type"] == "resolve_ticket"
    # All support criteria pass and none are leasing AC ids.
    ids = {r["id"] for r in out["ac_results"]}
    assert ids == {"SC-01", "SC-02", "SC-03"}
    assert out["evaluation"]["all_critical_pass"] is True


def test_support_domain_blocks_unconsented_channel(_support_llm):
    rec = {**SUPPORT_RECORD, "consent": {"email_opt_in": False, "push_opt_in": False}}
    out = _run(rec)
    assert out["should_send"] is False


def test_default_domain_still_leasing():
    # No selector -> the engine defaults to the leasing reference domain.
    assert domain_mod.get_domain({}).name == "leasing"
