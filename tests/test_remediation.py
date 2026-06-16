"""Tests for remediation work: corrective retry (G3) and raw-vs-repair metrics (G1)."""
import asyncio
import json
import os

for _k in ("LITELLM_API_KEY", "LITELLM_API_BASE", "LITELLM_MODEL",
           "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY",
           "GOOGLE_API_KEY", "GEMINI_API_KEY"):
    os.environ.pop(_k, None)

import agentkit.nodes.llm as llmnode  # noqa: E402
from agentkit.graph import app  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "evals")


def _rec():
    line = next(l for l in open(os.path.join(DATA, "sample_8613.jsonl")) if l.strip())
    return json.loads(line)


def _run(record):
    init = {"record": record, "dataset": [record],
            "task_id": record["task_id"], "raw_line": json.dumps(record)}
    return asyncio.run(app.ainvoke(init))


_GOOD_BODY = ("Hi Taylor, welcome to Oak Ridge Apartments near Richardson, TX. "
              "Book a tour this week! Reply STOP to opt out. "
              "Oak Ridge is an Equal Housing Opportunity provider.")


def _valid_json(channel="sms", send_at=None):
    msg = {"channel": channel, "subject": None, "body": _GOOD_BODY,
           "cta": {"type": "schedule_tour"}}
    if send_at:
        msg["send_at"] = send_at
    return json.dumps({
        "should_send": True,
        "next_message": msg,
        "next_action": {"type": "start_cadence", "name": "welcome"},
    })


# --- G3: corrective retry ---------------------------------------------------

def test_retry_uses_corrective_prompt_and_recovers():
    """First response is junk; the retry must (a) succeed and (b) be re-prompted with
    the prior output + the rejection reason (real self-correction, not a repeat)."""
    prompts_seen = []

    class JunkThenValid:
        provider = "mock"
        available = True

        def __init__(self):
            self.n = 0

        async def generate(self, system, user, **k):
            prompts_seen.append(user)
            self.n += 1
            return "not json at all" if self.n == 1 else _valid_json()

    llmnode._client = JunkThenValid()
    out = _run(_rec())["final_output"]

    assert out["should_send"] is True
    assert len(prompts_seen) >= 2
    # The second prompt must reference the correction + the prior bad output.
    assert "CORRECTION REQUIRED" in prompts_seen[1]
    assert "not json at all" in prompts_seen[1]


def test_retry_carries_specific_validation_error():
    """A response that parses but violates a guardrail (empty body) should retry with
    that specific reason, not a generic JSON error."""
    prompts_seen = []

    class EmptyBodyThenValid:
        provider = "mock"
        available = True

        def __init__(self):
            self.n = 0

        async def generate(self, system, user, **k):
            prompts_seen.append(user)
            self.n += 1
            if self.n == 1:
                return json.dumps({"should_send": True,
                                   "next_message": {"channel": "sms", "subject": None,
                                                    "body": "   ", "cta": {}},
                                   "next_action": {"type": "no_action"}})
            return _valid_json()

    llmnode._client = EmptyBodyThenValid()
    out = _run(_rec())["final_output"]
    assert out["should_send"] is True
    assert "body" in prompts_seen[1].lower()


# --- G1: raw decision vs post-repair compliance -----------------------------

def test_repair_metrics_distinguish_llm_from_guardrail():
    """LLM picks a NON-consented channel (voice). Final output is compliance-repaired
    to a consented channel (so ACs pass), but the metrics must record that a repair
    was needed and that the LLM's raw decision was non-compliant."""
    class WrongChannel:
        provider = "mock"
        available = True

        async def generate(self, system, user, **k):
            return _valid_json(channel="voice")  # not consented in the sample record

    llmnode._client = WrongChannel()
    state = _run(_rec())
    out = state["final_output"]
    ev = out["evaluation"]

    # Post-repair output still ships a compliant (consented) channel.
    assert out["should_send"] is True
    assert out["next_message"]["channel"] in ("sms", "email")
    assert not ev["critical_fails"]

    # But the metrics expose that the LLM itself was non-compliant and needed repair.
    assert ev["repairs_count"] >= 1
    assert ev["llm_raw_compliant"] is False
    assert any("channel_repair" in r for r in ev["repairs"])
    assert ev["llm_raw_decision"]["channel"] == "voice"


def test_clean_llm_output_marked_raw_compliant():
    """A fully-compliant LLM output needs no repair -> llm_raw_compliant True."""
    class Clean:
        provider = "mock"
        available = True

        async def generate(self, system, user, **k):
            from agentkit import timing
            send_at = timing.compute_send_at(_rec(), "sms")
            return _valid_json(channel="sms", send_at=send_at)

    llmnode._client = Clean()
    out = _run(_rec())["final_output"]
    ev = out["evaluation"]
    assert ev["llm_raw_compliant"] is True
    assert ev["repairs_count"] == 0


# --- G2: semantic match (embeddings with offline fallback) ------------------

def test_semantic_score_lexical_method_offline():
    from agentkit.evals import semantic
    expected = {"next_message": {"channel": "sms", "body": "Book a tour today"},
                "next_action": {"type": "no_action"}}
    s = asyncio.run(semantic.score_async(expected, expected, client=None))
    assert s["method"] == "lexical"
    assert s["overall"] == 1.0


def test_semantic_score_uses_embeddings_when_available():
    from agentkit.evals import semantic

    class FakeEmbedClient:
        available = True

        async def embed(self, texts):
            # Identical strings -> identical vectors -> cosine 1.0; else orthogonal.
            base = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
            return [base["a"] if t.strip() == "match" else base["b"] for t in texts]

    produced = {"next_message": {"channel": "sms", "body": "match",
                                 "cta": {"type": "schedule_tour"}},
                "next_action": {"type": "no_action"}}
    expected = {"next_message": {"channel": "sms", "body": "match",
                                 "cta": {"type": "schedule_tour"}},
                "next_action": {"type": "no_action"}}
    s = asyncio.run(semantic.score_async(produced, expected, client=FakeEmbedClient()))
    assert s["method"] == "embedding"
    assert s["fields"]["body_sim"] == 1.0


def test_cosine_basic():
    from agentkit.evals.semantic import _cosine
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0


# --- G6: broadened fair-housing + output PII --------------------------------

def test_broadened_fair_housing_patterns():
    from agentkit import safety_rules
    for phrase in ["perfect for a single Christian male",
                   "we prefer a young couple",
                   "no immigrants",
                   "adults only community"]:
        assert safety_rules.has_toxic(phrase), phrase
    # Ordinary copy must NOT trip the scanner.
    assert not safety_rules.has_toxic(
        "Welcome to Oak Ridge Apartments! Book a tour of our pet-friendly community.")


def test_output_pii_detects_address_and_card():
    from agentkit import safety_rules
    assert "street_address" in safety_rules.find_pii("Visit us at 1234 Oak Ridge Boulevard today")
    assert "credit_card" in safety_rules.find_pii("card 4111 1111 1111 1111 on file")
    assert safety_rules.find_pii("Book a tour this week!") == []


# --- G6: runtime fair-housing judge gate ------------------------------------

def test_fairness_gate_flags_high_confidence_violation():
    from agentkit.evals import judge

    class ViolationClient:
        available = True

        async def generate(self, system, user, **k):
            return '{"violation": true, "confidence": 0.95, "reason": "steering"}'

    v = asyncio.run(judge.fairness_gate("some body", client=ViolationClient()))
    assert v["violation"] is True


def test_fairness_gate_ignores_low_confidence_and_no_client():
    from agentkit.evals import judge

    class LowConf:
        available = True

        async def generate(self, system, user, **k):
            return '{"violation": true, "confidence": 0.3, "reason": "maybe"}'

    assert asyncio.run(judge.fairness_gate("b", client=LowConf()))["violation"] is False
    assert asyncio.run(judge.fairness_gate("b", client=None))["violation"] is False


def test_runtime_judge_gate_forces_no_send(monkeypatch):
    """With the gate enabled, a fair-housing veto from the judge forces a safe no-send
    even though the generated message passed the regex floor."""
    import agentkit.config as cfg
    monkeypatch.setattr(cfg, "RUNTIME_FAIRHOUSING_JUDGE", True)

    class GenThenVeto:
        provider = "mock"
        available = True

        async def generate(self, system, user, **k):
            if "Fair Housing Act compliance reviewer" in system:
                return '{"violation": true, "confidence": 0.99, "reason": "steering"}'
            return _valid_json(channel="sms")

    llmnode._client = GenThenVeto()
    out = _run(_rec())["final_output"]
    assert out["should_send"] is False
    assert "fair-housing" in (out.get("reasoning") or "").lower()


# --- G7: geo timezone correctness ------------------------------------------

def test_geo_hawaii_alaska_zip_mapping():
    from agentkit import geo
    assert geo.tz_from_zip("96801") == "Pacific/Honolulu"   # Honolulu, HI
    assert geo.tz_from_zip("99501") == "America/Anchorage"  # Anchorage, AK
    assert geo.tz_from_zip("90001") == "America/Los_Angeles"  # LA still Pacific


def test_geo_non_us_not_assumed_us_timezone():
    from agentkit import geo
    rec = {"input": {"country": "United Kingdom", "zip_code": "90001"}}
    zone, source = geo.resolve_timezone(rec)
    assert zone is None and source == "none"  # don't guess a US zone for a UK prospect
    # Explicit IANA tz is still honored regardless of country.
    rec2 = {"input": {"country": "United Kingdom", "timezone": "Europe/London"}}
    zone2, source2 = geo.resolve_timezone(rec2)
    assert source2 == "iana_field" and zone2 is not None


# --- G8: per-key circuit breaker isolation ----------------------------------

def test_breaker_isolated_per_key():
    import agentkit.circuit_breaker as cb
    cb.reset_all()
    a = cb.get_breaker("litellm", "tenant_a")
    b = cb.get_breaker("litellm", "tenant_b")
    assert a is not b
    for _ in range(cb.config.CIRCUIT_BREAKER_THRESHOLD):
        a.record_failure()
    assert a.allow() is False   # tenant A tripped
    assert b.allow() is True    # tenant B unaffected
    # Same key returns the same instance.
    assert cb.get_breaker("litellm", "tenant_a") is a


def test_breaker_default_when_no_key():
    import agentkit.circuit_breaker as cb
    assert cb.get_breaker() is cb.breaker


# --- G9: unjustified no-send is audited -------------------------------------

def test_unjustified_no_send_is_flagged(monkeypatch):
    events = []
    import agentkit.telemetry as telemetry
    monkeypatch.setattr(telemetry.store, "emit_event",
                        lambda t, p: events.append((t, p)) or {})

    class Declines:
        provider = "mock"
        available = True

        async def generate(self, system, user, **k):
            return json.dumps({"should_send": False, "next_message": None,
                               "next_action": {"type": "no_action"},
                               "reasoning": "not interested"})

    llmnode._client = Declines()
    out = _run(_rec())["final_output"]  # sample record HAS sms/email consent
    assert out["should_send"] is False
    assert out["suppression"]["unjustified"] is True
    assert any(t == "suppressed_send" for t, _ in events)


def test_justified_no_send_not_flagged(adversarial_rec=None):
    """No consented channel -> no-send is justified, not flagged."""
    rec = _rec()
    rec["consent"] = {"email_opt_in": False, "sms_opt_in": False, "voice_opt_in": False}
    # No provider call needed; intake vetoes. Use default mock.
    out = _run(rec)["final_output"]
    assert out["should_send"] is False
    supp = out.get("suppression")
    # Either no suppression audit (aborted at intake) or marked justified.
    assert supp is None or supp.get("unjustified") is False


# --- G5: dead-letter queue + replay -----------------------------------------

def test_retriable_abort_is_dead_lettered(tmp_path):
    """A transient backend failure (LLM unavailable) routes the record to the DLQ."""
    import asyncio
    from agentkit.prod.runner import Pipeline
    from agentkit.prod.deadletter import DeadLetterQueue

    class Down:
        provider = "mock"
        available = False

        async def generate(self, system, user, **k):  # pragma: no cover
            raise RuntimeError("should not be called when unavailable")

    llmnode._client = Down()
    pipe = Pipeline(concurrency=2, use_cache=False, use_idempotency=False,
                    use_audit=False)
    pipe.dlq = DeadLetterQueue(path=str(tmp_path / "dlq.jsonl"))
    out = asyncio.run(pipe.process_one(_rec()))
    assert out["should_send"] is False
    assert out.get("abort_reason") == "LLM_UNAVAILABLE"
    pending = pipe.dlq.pending()
    assert len(pending) == 1
    assert pending[0]["abort_reason"] == "LLM_UNAVAILABLE"
    assert pipe.dead_lettered == 1


def test_no_consent_not_dead_lettered(tmp_path):
    """A terminal business no-send (NO_CONSENT) is never enqueued for replay."""
    import asyncio
    from agentkit.prod.runner import Pipeline
    from agentkit.prod.deadletter import DeadLetterQueue

    rec = _rec()
    rec["consent"] = {"email_opt_in": False, "sms_opt_in": False, "voice_opt_in": False}
    pipe = Pipeline(concurrency=2, use_cache=False, use_idempotency=False,
                    use_audit=False)
    pipe.dlq = DeadLetterQueue(path=str(tmp_path / "dlq.jsonl"))
    out = asyncio.run(pipe.process_one(rec))
    assert out["should_send"] is False
    assert pipe.dlq.pending() == []
    assert pipe.dead_lettered == 0


def test_replay_recovers_when_backend_healthy(tmp_path):
    """Replay re-runs DLQ entries; a now-healthy backend clears them."""
    import asyncio
    from agentkit.prod.runner import Pipeline
    from agentkit.prod.deadletter import DeadLetterQueue

    class Down:
        provider = "mock"
        available = False

        async def generate(self, system, user, **k):  # pragma: no cover
            raise RuntimeError("down")

    llmnode._client = Down()
    pipe = Pipeline(concurrency=2, use_cache=False, use_idempotency=False,
                    use_audit=False)
    pipe.dlq = DeadLetterQueue(path=str(tmp_path / "dlq.jsonl"))
    asyncio.run(pipe.process_one(_rec()))
    assert len(pipe.dlq.pending()) == 1

    # Backend recovers: swap in a healthy client that emits a valid send.
    class Healthy:
        provider = "mock"
        available = True

        async def generate(self, system, user, **k):
            from agentkit import timing
            send_at = timing.compute_send_at(_rec(), "sms")
            return _valid_json("sms", send_at)

    llmnode._client = Healthy()
    summary = asyncio.run(pipe.replay_dead_letters())
    assert summary["replayed"] == 1
    assert summary["recovered"] == 1
    assert summary["still_failing"] == 0
    assert pipe.dlq.pending() == []
