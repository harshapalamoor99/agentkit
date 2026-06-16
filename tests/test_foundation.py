"""Tests for the reusable foundation: RAG knowledge layer + multi-agent surface."""
import asyncio
import json

import pytest

from messaging_agent import (
    AgentRouter,
    AgentService,
    InMemoryKnowledgeBase,
    KnowledgeDoc,
    available_domains,
)
from messaging_agent import knowledge as kb_mod
from messaging_agent.domain import DecisionContext, Domain, register_domain


# --------------------------------------------------------------------------------------
# Knowledge base
# --------------------------------------------------------------------------------------

def test_in_memory_kb_retrieves_relevant_snippet():
    kb = InMemoryKnowledgeBase([
        KnowledgeDoc(id="d1", text="Pet policy: cats and dogs welcome, $300 deposit."),
        KnowledgeDoc(id="d2", text="The gym is open 24/7 with peloton bikes."),
        KnowledgeDoc(id="d3", text="Parking is available in the covered garage."),
    ])
    hits = kb.retrieve("do you allow pets and what is the deposit", k=2)
    assert hits
    assert hits[0].id == "d1"
    assert hits[0].score > 0


def test_in_memory_kb_metadata_filter_isolates():
    kb = InMemoryKnowledgeBase([
        KnowledgeDoc(id="a", text="gym hours and amenities", metadata={"tenant_id": "t1"}),
        KnowledgeDoc(id="b", text="gym hours and amenities", metadata={"tenant_id": "t2"}),
    ])
    hits = kb.retrieve("gym", k=5, where={"tenant_id": "t2"})
    assert [h.id for h in hits] == ["b"]


def test_from_jsonl(tmp_path):
    p = tmp_path / "kb.jsonl"
    p.write_text(
        json.dumps({"id": "x", "text": "rooftop pool open in summer", "source": "faq"}) + "\n"
        + json.dumps({"content": "laundry on every floor"}) + "\n",
        encoding="utf-8")
    kb = InMemoryKnowledgeBase.from_jsonl(p, name="t")
    assert kb.retrieve("pool")[0].id == "x"
    assert kb.retrieve("laundry")  # the second doc loaded via the "content" key


# --------------------------------------------------------------------------------------
# A RAG-enabled domain wired through the graph
# --------------------------------------------------------------------------------------

_KB = InMemoryKnowledgeBase([
    KnowledgeDoc(id="faq-pets", text="Pets: cats and dogs allowed with a deposit."),
    KnowledgeDoc(id="faq-gym", text="Amenities: 24/7 gym and a rooftop pool."),
], name="rag_test_kb")


class _RagDomain(Domain):
    name = "rag_test"

    def channel_consent_map(self):
        return {"email": "email_opt_in"}

    def knowledge_base(self):
        return _KB

    def knowledge_query(self, record, facts):
        return (record.get("input", {}) or {}).get("question")

    def build_decision_context(self, *, record, sanitized, tenant, dataset):
        facts = {"allowed_channels": self.consented_channels(record), "primary_cta": "answer"}
        return DecisionContext(facts=facts, guardrails={"allowed_channels": facts["allowed_channels"]},
                               system_prompt="answer using knowledge", user_prompt="Q",
                               extras={"lineage": {"task_id": record.get("task_id")}})

    def normalize(self, output, state):
        # Capture the prompt that the context node assembled (with knowledge appended).
        return {"should_send": True,
                "next_message": {"channel": "email", "subject": "re", "body": "ok"},
                "next_action": {"type": "no_action"}, "_warnings": []}

    def evaluate_all(self, output, record, sanitized):
        return [{"id": "R-01", "severity": "critical", "title": "ok", "pass": True, "detail": ""}]


register_domain(_RagDomain())


class _EchoLLM:
    provider = "mock"
    available = True

    async def warmup(self):
        return None

    async def generate(self, system, user, **kwargs):
        # The user prompt should contain the retrieved knowledge block.
        _EchoLLM.last_user = user
        return json.dumps({"should_send": True,
                           "next_message": {"channel": "email", "subject": "re", "body": "ok"},
                           "next_action": {"type": "no_action"}})


@pytest.fixture
def _echo_llm():
    import messaging_agent.nodes.llm as llmnode
    original = llmnode._client
    llmnode._client = _EchoLLM()
    yield
    llmnode._client = original


def _run(record):
    from messaging_agent.graph import app
    init = {"task_id": record.get("task_id"), "record": record, "dataset": [record]}
    return asyncio.run(app.ainvoke(init))["final_output"]


def test_rag_injects_retrieved_knowledge_into_prompt_and_output(_echo_llm):
    rec = {
        "task_id": "rag-1", "domain": "rag_test",
        "consent": {"email_opt_in": True},
        "input": {"question": "what are the gym and pool amenities"},
    }
    out = _run(rec)
    # Retrieved knowledge is surfaced on the output and lineage.
    ids = {k["id"] for k in out["knowledge"]}
    assert "faq-gym" in ids
    assert out["lineage"]["knowledge"]
    # And it was actually placed into the LLM prompt as grounding.
    assert "rooftop pool" in _EchoLLM.last_user


# --------------------------------------------------------------------------------------
# Multi-agent surface
# --------------------------------------------------------------------------------------

def test_agent_service_as_tool_descriptor():
    tool = AgentService("leasing").as_tool()
    assert tool["name"] == "leasing_agent"
    assert "record" in tool["input_schema"]["properties"]
    assert callable(tool["callable"])


def test_router_dispatches_by_record_domain(_echo_llm):
    router = AgentRouter(default_domain="rag_test")
    rec = {"task_id": "r2", "domain": "rag_test",
           "consent": {"email_opt_in": True},
           "input": {"question": "pets?"}}
    out = asyncio.run(router.dispatch(rec))
    assert out["routed_to"] == "rag_test"
    assert out["should_send"] is True


def test_router_uses_classifier_then_default(_echo_llm):
    seen = {}

    def classifier(record):
        seen["called"] = True
        return "rag_test" if record.get("input", {}).get("question") else None

    router = AgentRouter(classifier=classifier, default_domain="leasing")
    rec = {"task_id": "r3", "consent": {"email_opt_in": True},
           "input": {"question": "gym?"}}
    out = asyncio.run(router.dispatch(rec))
    assert seen.get("called") is True
    assert out["routed_to"] == "rag_test"


def test_available_domains_includes_builtins():
    names = available_domains()
    assert "leasing" in names and "support" in names
