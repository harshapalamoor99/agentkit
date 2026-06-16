"""Golden regression tests against the external answer-key dataset.

Two layers (see ai-engineering-excellence testing pyramid):

* Hermetic layer (default, runs in CI): with the autouse mock LLM, the agent's
  ``should_send`` and ``channel`` are produced deterministically (consent gating +
  ``channels.select_channel``). We assert those match the golden ``expected`` for
  every record, guarding against regressions in consent/channel-selection logic.

* Live eval layer (``@pytest.mark.eval``, opt-in): with a real provider configured,
  assert the full decision — ``should_send``, ``channel`` AND ``next_action.type`` —
  matches the golden expected for all records. Deselect in CI via ``-m 'not eval'``.
"""
import asyncio
import json
import os

import pytest

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "evals")
FIXTURES = {
    "golden_full": (os.path.join(FIXTURE_DIR, "golden_full.jsonl"), 36),
    "golden_prospect_examples": (os.path.join(FIXTURE_DIR, "golden_prospect_examples.jsonl"), 2),
    "hard": (os.path.join(FIXTURE_DIR, "hard.jsonl"), 29),
}


def _load(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _expected(record):
    exp = record.get("expected") or {}
    msg = exp.get("next_message")
    na = exp.get("next_action") or {}
    return {
        "should_send": bool(msg),
        "channel": (msg or {}).get("channel") if msg else None,
        "next_action": na.get("type"),
    }


def _actual(final):
    msg = final.get("next_message") or {}
    send = final.get("should_send")
    return {
        "should_send": send,
        "channel": msg.get("channel") if send else None,
        "next_action": (final.get("next_action") or {}).get("type"),
    }


def _run_all(records):
    from agentkit.graph import app

    out = {}
    for rec in records:
        init = {
            "record": rec,
            "dataset": records,
            "task_id": rec.get("task_id", "t"),
            "raw_line": json.dumps(rec, ensure_ascii=False),
        }
        out[rec.get("task_id")] = asyncio.run(app.ainvoke(init))["final_output"]
    return out


@pytest.mark.parametrize("fixture_key", list(FIXTURES))
def test_golden_should_send_and_channel_match(fixture_key):
    """Deterministic fields must match the golden answer key for every record."""
    path, expected_count = FIXTURES[fixture_key]
    records = _load(path)
    assert len(records) == expected_count
    finals = _run_all(records)

    ss_mismatch, ch_mismatch = [], []
    for rec in records:
        tid = rec.get("task_id")
        exp, act = _expected(rec), _actual(finals[tid])
        if act["should_send"] != exp["should_send"]:
            ss_mismatch.append((tid, exp["should_send"], act["should_send"]))
        if act["channel"] != exp["channel"]:
            ch_mismatch.append((tid, exp["channel"], act["channel"]))

    assert not ss_mismatch, f"should_send mismatches: {ss_mismatch}"
    assert not ch_mismatch, f"channel mismatches: {ch_mismatch}"


@pytest.mark.parametrize("fixture_key", list(FIXTURES))
def test_golden_no_critical_fails(fixture_key):
    """No record may produce a CRITICAL acceptance-criteria failure."""
    path, _ = FIXTURES[fixture_key]
    records = _load(path)
    finals = _run_all(records)
    offenders = {
        tid: [c["id"] for c in (f.get("evaluation") or {}).get("critical_fails", [])]
        for tid, f in finals.items()
        if (f.get("evaluation") or {}).get("critical_fails")
    }
    assert not offenders, f"critical AC failures: {offenders}"


@pytest.mark.eval
@pytest.mark.parametrize("fixture_key", list(FIXTURES))
def test_golden_full_decision_match_live(fixture_key):
    """Live LLM: should_send, channel AND next_action.type must match every record.

    The suite's autouse fixture installs a hermetic mock for every test, so this test
    explicitly reinstalls a real ``LLMClient`` to exercise the configured provider.
    Opt in with ``RUN_LIVE_EVAL=1 pytest -m eval``; skipped otherwise (incl. CI).
    """
    if os.environ.get("RUN_LIVE_EVAL") != "1":
        pytest.skip("set RUN_LIVE_EVAL=1 to run the live golden eval")

    import agentkit.nodes.llm as llmnode
    from agentkit.llm_client import LLMClient

    real = LLMClient()
    if not getattr(real, "provider", None):
        pytest.skip("no live provider configured (set LITELLM_API_KEY/_API_BASE etc.)")

    path, _ = FIXTURES[fixture_key]
    saved = llmnode._client
    llmnode._client = real
    try:
        records = _load(path)
        finals = _run_all(records)
    finally:
        llmnode._client = saved

    mismatches = []
    for rec in records:
        tid = rec.get("task_id")
        exp, act = _expected(rec), _actual(finals[tid])
        if act != exp:
            mismatches.append((tid, exp, act))
    assert not mismatches, f"live decision mismatches: {mismatches}"
