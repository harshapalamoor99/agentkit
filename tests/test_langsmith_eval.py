"""Tests for the LangSmith eval adapter.

These never touch the LangSmith SaaS: they exercise the gating logic, the pure scorers
(which are the single source of truth, shared with the local harness), the run/example
evaluator wrappers (using lightweight stand-in objects), and the target callable against
the hermetic mock LLM.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest

from agentkit.evals import langsmith_eval as lse


# --- gating ---

def test_langsmith_disabled_without_key(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert lse.langsmith_enabled() is False
    st = lse.status()
    assert st["enabled"] is False
    assert st["has_api_key"] is False


def test_network_entrypoints_raise_when_disabled(monkeypatch):
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="not installed|not configured"):
        lse.push_dataset("x", [{"task_id": "1"}])
    with pytest.raises(RuntimeError, match="not installed|not configured"):
        asyncio.run(lse.run_experiment("x"))


# --- pure scorers (shared with the harness) ---

_OUTPUT = {
    "next_message": {"channel": "email", "subject": "Tour Oak Ridge",
                     "body": "Hi Taylor, welcome to Oak Ridge! Book a tour. Reply STOP."},
    "next_action": {"type": "start_cadence"},
    "should_send": True,
    "knowledge": [{"id": "oak-pets"}, {"id": "oak-parking"}],
    "evaluation": {"critical_fails": []},
}
_RECORD = {"task_id": "r1", "input": {"property_name": "Oak Ridge",
           "profile": {"first_name": "Taylor", "amenity_interest": ["pool"]}}}


def test_score_personalization_detects_name():
    res = lse.score_personalization(_OUTPUT, _RECORD)
    assert res["key"] == "personalization"
    assert res["score"] > 0  # name + property present


def test_score_knowledge_recall_full_and_partial():
    full = lse.score_knowledge_recall(_OUTPUT, ["oak-pets", "oak-parking"])
    assert full["score"] == 1.0
    partial = lse.score_knowledge_recall(_OUTPUT, ["oak-pets", "missing"])
    assert partial["score"] == 0.5
    assert lse.score_knowledge_recall(_OUTPUT, None) is None


def test_score_ac_no_critical():
    assert lse.score_ac_no_critical(_OUTPUT)["score"] == 1.0
    bad = {**_OUTPUT, "evaluation": {"critical_fails": ["AC-08"]}}
    res = lse.score_ac_no_critical(bad)
    assert res["score"] == 0.0
    assert "AC-08" in res["comment"]


def test_score_semantic_requires_expected():
    assert lse.score_semantic(_OUTPUT, None) is None
    res = lse.score_semantic(_OUTPUT, {"next_message": {"body": "welcome to Oak Ridge book a tour"}})
    assert res["key"] == "semantic_match"
    assert 0.0 <= res["score"] <= 1.0


def test_score_judge_runs_heuristic_offline():
    # No provider key in tests → judge falls back to its heuristic, still returns a score.
    res = asyncio.run(lse.score_judge(_OUTPUT, _RECORD))
    assert res["key"] == "judge_quality"
    assert 0.0 <= res["score"] <= 1.0


# --- evaluator wrappers (run/example stand-ins) ---

def _example(record, expected=None, expected_ids=None):
    return SimpleNamespace(
        inputs={"record": record},
        outputs={"expected": expected, "expected_knowledge_ids": expected_ids},
    )


def test_build_evaluators_default_set():
    evals = lse.build_evaluators(use_judge=False)
    assert len(evals) == 4
    run = SimpleNamespace(outputs=_OUTPUT)
    example = _example(_RECORD, expected_ids=["oak-pets", "oak-parking"])
    results = {e(run, example)["key"] for e in evals}
    assert results == {"personalization", "semantic_match", "knowledge_recall",
                       "ac_no_critical_fail"}


def test_build_evaluators_with_judge_is_async():
    evals = lse.build_evaluators(use_judge=True)
    assert len(evals) == 5
    judge_eval = evals[-1]
    run = SimpleNamespace(outputs=_OUTPUT)
    out = asyncio.run(judge_eval(run, _example(_RECORD)))
    assert out["key"] == "judge_quality"


def test_knowledge_recall_evaluator_reads_reference():
    evals = lse.build_evaluators()
    kr = [e for e in evals if e.__name__ == "knowledge_recall_evaluator"][0]
    run = SimpleNamespace(outputs=_OUTPUT)
    res = kr(run, _example(_RECORD, expected_ids=["oak-pets", "oak-parking"]))
    assert res["score"] == 1.0


# --- target callable against the mock LLM ---

def test_make_target_returns_final_output():
    target = lse.make_target()
    record = {
        "task_id": "tgt-1", "domain": "support",
        "consent": {"email_opt_in": True},
        "channel_preferences": ["email"],
        "input": {"ticket_subject": "Help", "profile": {"first_name": "Jo"}},
        "assertions": {"constraints": {"primary_cta": "resolve_ticket"}},
    }
    out = asyncio.run(target({"record": record, "dataset": [record]}))
    assert out["task_id"] == "tgt-1"
    assert "should_send" in out
