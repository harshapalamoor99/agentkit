"""Tests for tool-calling adapters (OpenAI/Anthropic) and the RAG eval dataset."""
import asyncio
import json
from pathlib import Path

import pytest

from messaging_agent import (
    AgentRouter,
    AgentService,
    ToolRegistry,
    router_as_tool,
    run_tool_call,
    to_anthropic_tool,
    to_openai_tool,
)


class _EchoLLM:
    provider = "mock"
    available = True

    async def warmup(self):
        return None

    async def generate(self, system, user, **kwargs):
        return json.dumps({
            "should_send": True,
            "next_message": {"channel": "email", "subject": "re", "body": "ok"},
            "next_action": {"type": "no_action"},
        })


@pytest.fixture
def _echo_llm():
    import messaging_agent.nodes.llm as llmnode
    original = llmnode._client
    llmnode._client = _EchoLLM()
    yield
    llmnode._client = original


SUPPORT_RECORD = {
    "task_id": "tool-1", "domain": "support",
    "consent": {"email_opt_in": True, "push_opt_in": False},
    "channel_preferences": ["email"],
    "input": {"ticket_subject": "Login issue", "profile": {"first_name": "Sam"}},
    "assertions": {"constraints": {"primary_cta": "resolve_ticket"}},
}


# --- adapter shape ---

def test_to_openai_tool_shape():
    spec = to_openai_tool(AgentService("support").as_tool())
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "support_agent"
    assert spec["function"]["parameters"]["required"] == ["record"]


def test_to_anthropic_tool_shape():
    spec = to_anthropic_tool(AgentService("support").as_tool())
    assert spec["name"] == "support_agent"
    assert "record" in spec["input_schema"]["properties"]
    assert "type" not in spec  # anthropic tools have no top-level "type"


# --- execution from a model-issued call ---

def test_run_tool_call_with_json_string_args_openai_style(_echo_llm):
    svc = AgentService("support")
    args = json.dumps({"record": SUPPORT_RECORD})
    out = asyncio.run(run_tool_call(svc, "support_agent", args))
    assert isinstance(out, str)
    parsed = json.loads(out)
    assert parsed["should_send"] is True
    assert parsed["next_message"]["channel"] == "email"


def test_run_tool_call_with_dict_args_anthropic_style(_echo_llm):
    svc = AgentService("support")
    out = asyncio.run(run_tool_call(svc, "support_agent", {"record": SUPPORT_RECORD},
                                    as_json=False))
    assert isinstance(out, dict)
    assert out["should_send"] is True


def test_run_tool_call_bad_json_raises():
    svc = AgentService("support")
    with pytest.raises(ValueError):
        asyncio.run(run_tool_call(svc, "support_agent", "{not json"))


# --- multi-agent tool registry ---

def test_tool_registry_advertises_and_dispatches(_echo_llm):
    reg = ToolRegistry(["leasing", "support"])
    names = {t["function"]["name"] for t in reg.tools_openai()}
    assert names == {"leasing_agent", "support_agent"}
    out = asyncio.run(reg.dispatch("support_agent", {"record": SUPPORT_RECORD}))
    assert json.loads(out)["should_send"] is True


def test_tool_registry_unknown_tool_returns_error():
    reg = ToolRegistry(["support"])
    out = asyncio.run(reg.dispatch("nope", {"record": {}}))
    assert "error" in json.loads(out)


def test_router_as_tool_auto_routes(_echo_llm):
    router = AgentRouter(default_domain="support")
    tool = router_as_tool(router)
    assert tool["name"] == "messaging_agent"
    result = asyncio.run(tool["callable"]({"record": SUPPORT_RECORD}))
    assert result["routed_to"] == "support"


# --- RAG eval dataset ---

def test_rag_eval_dataset_scores_full_recall():
    from messaging_agent.evals.harness import run_file
    path = Path(__file__).resolve().parents[1] / "data" / "evals" / "rag_knowledge.jsonl"
    report = run_file(str(path))
    summary = report["summary"]
    assert summary["knowledge"]["records_with_knowledge"] == 4
    assert summary["knowledge"]["mean_recall"] == 1.0
    for item in report["items"]:
        assert item["knowledge_recall"] == 1.0


def test_bundled_leasing_kb_loads():
    from messaging_agent.domains.leasing import LeasingDomain
    kb = LeasingDomain().knowledge_base()
    assert kb is not None
    hits = kb.retrieve("pet deposit", k=2, where={"tenant_id": "oakridge_pm"})
    assert any(h.id == "oak-pets" for h in hits)
    # tenant isolation: no summit docs leak into an oakridge query
    assert all(h.metadata.get("tenant_id") == "oakridge_pm" for h in hits)
