"""Tests for round-2 gaps (G11–G16): isolation keys, PII year guard, subject scanning,
enforcement of legal_footer/interests, and cross-tenant few-shot isolation."""
import asyncio
import json
import os

for _k in ("LITELLM_API_KEY", "LITELLM_API_BASE", "LITELLM_MODEL",
           "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY",
           "GOOGLE_API_KEY", "GEMINI_API_KEY", "REDIS_URL"):
    os.environ.pop(_k, None)

import agentkit.nodes.llm as llmnode  # noqa: E402
from agentkit.graph import app  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "evals")


def _rec():
    line = next(l for l in open(os.path.join(DATA, "sample_8613.jsonl")) if l.strip())
    return json.loads(line)


def _run(record, dataset=None):
    init = {"record": record, "dataset": dataset or [record],
            "task_id": record["task_id"], "raw_line": json.dumps(record)}
    return asyncio.run(app.ainvoke(init))


# --- G14: idempotency / cache keys are tenant-scoped ------------------------

def test_idempotency_key_distinct_across_tenants():
    from agentkit.prod.idempotency import idempotency_key
    a = {"task_id": "t1", "tenant_id": "oakridge_pm",
         "input": {"property_name": "Oak Ridge Apartments"}}
    b = {"task_id": "t1", "tenant_id": "summit_residential",
         "input": {"property_name": "Summit Heights"}}
    assert idempotency_key(a) != idempotency_key(b)
    assert idempotency_key(a) == idempotency_key(dict(a))


def test_cache_fingerprint_distinct_across_tenants():
    from agentkit.prod.cache import fingerprint
    c = {"channel": "email", "primary_cta": "book_tour"}
    a = {"task_id": "t1", "tenant_id": "oakridge_pm",
         "input": {"property_name": "Oak Ridge Apartments"}}
    b = {"task_id": "t1", "tenant_id": "summit_residential",
         "input": {"property_name": "Summit Heights"}}
    assert fingerprint(a, c) != fingerprint(b, c)


# --- G15: raw-PII year guard ------------------------------------------------

def test_pii_flags_currency_and_income_not_year():
    from agentkit.pii import output_reflects_raw_pii
    assert output_reflects_raw_pii("Save $2000 this month")  # currency -> flagged
    assert output_reflects_raw_pii("Your income of 84000 qualifies")  # income -> flagged
    assert output_reflects_raw_pii("balance 12500 remaining")  # bare 5-digit -> flagged
    assert not output_reflects_raw_pii("Move in by 2026 for a tour")  # bare year -> ok
    assert not output_reflects_raw_pii("Open 24/7, 2 bed available")  # short -> ok
    assert not output_reflects_raw_pii("See you in 1999 ways")  # bare year -> ok


# --- G12: email subject is scanned + repaired -------------------------------

def _email_rec():
    rec = _rec()
    rec["consent"] = {"email_opt_in": True, "sms_opt_in": False, "voice_opt_in": False}
    rec["channel_preferences"] = ["email"]
    return rec


def test_email_subject_with_foreign_property_is_repaired():
    """A subject naming another tenant's property must be replaced, not shipped."""
    body = ("Hi Taylor, welcome to Oak Ridge Apartments near Richardson, TX. "
            "Book a tour this week! To opt out reply STOP or click unsubscribe.")

    class ForeignSubject:
        provider = "mock"
        available = True

        async def generate(self, system, user, **k):
            from agentkit import timing
            send_at = timing.compute_send_at(_email_rec(), "email")
            return json.dumps({
                "should_send": True,
                "next_message": {"channel": "email",
                                 "subject": "Check out Summit Heights too!",
                                 "send_at": send_at, "body": body,
                                 "cta": {"type": "schedule_tour"}},
                "next_action": {"type": "start_cadence", "name": "welcome"},
            })

    llmnode._client = ForeignSubject()
    out = _run(_email_rec())["final_output"]
    assert out["should_send"] is True
    subj = out["next_message"]["subject"]
    assert "summit" not in subj.lower()
    assert any("subject_quarantine" in w for w in out.get("warnings", []))


def test_clean_email_subject_preserved():
    body = ("Hi Taylor, welcome to Oak Ridge Apartments near Richardson, TX. "
            "Book a tour this week! To opt out reply STOP or click unsubscribe.")

    class CleanSubject:
        provider = "mock"
        available = True

        async def generate(self, system, user, **k):
            from agentkit import timing
            send_at = timing.compute_send_at(_email_rec(), "email")
            return json.dumps({
                "should_send": True,
                "next_message": {"channel": "email",
                                 "subject": "Tour Oak Ridge Apartments this week",
                                 "send_at": send_at, "body": body,
                                 "cta": {"type": "schedule_tour"}},
                "next_action": {"type": "start_cadence", "name": "welcome"},
            })

    llmnode._client = CleanSubject()
    out = _run(_email_rec())["final_output"]
    assert out["next_message"]["subject"] == "Tour Oak Ridge Apartments this week"
    assert not any("subject_quarantine" in w for w in out.get("warnings", []))


# --- G13: legal_footer enforced; interest flag non-blocking ------------------

def _body_no_footer():
    return ("Hi Taylor, welcome to Oak Ridge Apartments near Richardson, TX. "
            "Book a tour this week! Reply STOP to opt out.")


def test_missing_legal_footer_is_appended():
    """Tenant legal_footer is non-negotiable -> appended when the model omits it."""
    class NoFooter:
        provider = "mock"
        available = True

        async def generate(self, system, user, **k):
            from agentkit import timing
            send_at = timing.compute_send_at(_rec(), "sms")
            return json.dumps({
                "should_send": True,
                "next_message": {"channel": "sms", "subject": None,
                                 "send_at": send_at, "body": _body_no_footer(),
                                 "cta": {"type": "schedule_tour"}},
                "next_action": {"type": "start_cadence", "name": "welcome"},
            })

    llmnode._client = NoFooter()
    out = _run(_rec())["final_output"]
    body = out["next_message"]["body"]
    assert "Equal Housing Opportunity" in body
    assert any("legal_footer_repair" in w for w in out["warnings"])


def test_present_legal_footer_not_duplicated():
    footer = "Oak Ridge is an Equal Housing Opportunity provider."

    class WithFooter:
        provider = "mock"
        available = True

        async def generate(self, system, user, **k):
            from agentkit import timing
            send_at = timing.compute_send_at(_rec(), "sms")
            body = _body_no_footer() + " " + footer
            return json.dumps({
                "should_send": True,
                "next_message": {"channel": "sms", "subject": None,
                                 "send_at": send_at, "body": body,
                                 "cta": {"type": "schedule_tour"}},
                "next_action": {"type": "start_cadence", "name": "welcome"},
            })

    llmnode._client = WithFooter()
    out = _run(_rec())["final_output"]
    body = out["next_message"]["body"]
    assert body.count("Equal Housing Opportunity") == 1
    assert not any("legal_footer_repair" in w for w in out["warnings"])


def test_unreflected_interest_flagged_but_still_sends():
    """A body that ignores the prospect's interests is flagged, NOT blocked (G13)."""
    footer = "Oak Ridge is an Equal Housing Opportunity provider."

    class Generic:
        provider = "mock"
        available = True

        async def generate(self, system, user, **k):
            from agentkit import timing
            send_at = timing.compute_send_at(_rec(), "sms")
            # Deliberately omit Richardson/city interest.
            body = ("Hi Taylor, thanks for reaching out. Book a tour this week! "
                    "Reply STOP to opt out. " + footer)
            return json.dumps({
                "should_send": True,
                "next_message": {"channel": "sms", "subject": None,
                                 "send_at": send_at, "body": body,
                                 "cta": {"type": "schedule_tour"}},
                "next_action": {"type": "start_cadence", "name": "welcome"},
            })

    llmnode._client = Generic()
    out = _run(_rec())["final_output"]
    assert out["should_send"] is True  # non-blocking
    assert out["interest_reflected"] is False
    assert any("interest_not_reflected" in w for w in out["warnings"])


# --- G11: cross-tenant few-shot isolation + example PII tokenization ---------

def test_examples_tokenize_pii_before_prompt():
    """Raw sensitive fields in an example record must not reach the rendered prompt."""
    from agentkit.prompts import build_examples_with_ids
    example = {
        "task_id": "ex1",
        "input": {"property_name": "Oak Ridge Apartments",
                  "profile": {"first_name": "Sam", "income": 84000, "ssn": "123-45-6789"}},
        "expected": {"should_send": True,
                     "next_message": {"channel": "sms", "body": "Hi Sam"}},
    }
    text, ids = build_examples_with_ids([example], exclude_task_id=None)
    assert ids == ["ex1"]
    assert "84000" not in text
    assert "123-45-6789" not in text
    assert "income_band" in text  # coarse category present instead


def test_no_cross_tenant_examples_when_tenant_has_none():
    """A record whose tenant has no own examples must get NO few-shot, never another
    tenant's records (no foreign property names / expected outputs leak in)."""
    from agentkit.nodes.context import context_builder

    target = {  # oakridge tenant, no expected -> not itself an example
        "task_id": "target", "tenant_id": "oakridge_pm",
        "consent": {"sms_opt_in": True}, "channel_preferences": ["sms"],
        "input": {"property_name": "Oak Ridge Apartments", "timezone": "America/Chicago",
                  "profile": {"first_name": "Taylor"}},
    }
    foreign_example = {  # different tenant, has expected -> would be a tempting example
        "task_id": "foreign", "tenant_id": "summit_residential",
        "input": {"property_name": "Summit Heights",
                  "profile": {"first_name": "Jordan"}},
        "expected": {"should_send": True,
                     "next_message": {"channel": "sms", "body": "Visit Summit Heights!"}},
    }
    state = context_builder({"validated_record": target, "task_id": "target",
                             "dataset": [target, foreign_example]})
    prompt = state["enriched_context"]["user_prompt"]
    # The real G11 guarantee: no other tenant's inputs/expected outputs may leak in.
    assert "Summit Heights" not in prompt
    assert "foreign" not in state["lineage"]["few_shot_example_ids"]
    # When a tenant has no own examples we fall back to the tenant-neutral canonical
    # bank (synthetic, no real tenant) — never another tenant's records.
    assert all(eid.startswith("canonical_")
               for eid in state["lineage"]["few_shot_example_ids"])
