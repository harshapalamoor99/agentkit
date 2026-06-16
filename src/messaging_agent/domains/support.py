"""Support domain — a minimal *second* domain that proves the engine is reusable.

This is intentionally unrelated to housing/leasing: it drives customer-support
follow-up messages. It demonstrates that a new use case only needs a :class:`Domain`
subclass — its own channels/consent model, prompt, facts, compliance repairs and
acceptance criteria — with **no change to the core graph, nodes, LLM client or prod
layer**. Select it per record with ``{"domain": "support", ...}`` or process-wide with
``AGENT_DOMAIN=support``.

Use it as a template for authoring your own domain (see ``docs/AUTHORING_A_DOMAIN.md``).
"""
from __future__ import annotations

import json
from typing import Any

from ..domain import DecisionContext, Domain, register_domain

_SYSTEM_PROMPT = """\
You are a customer-support follow-up agent. For each ticket you decide WHETHER to send a
follow-up, on WHICH consented channel, and WHAT to say. Be concise, helpful and never
make promises. Output STRICT JSON only:
{
  "should_send": <bool>,
  "next_message": {"channel": "<allowed channel>", "subject": <string|null>,
                   "body": "<text with opt-out>", "cta": {"type": "<primary_cta>"}} | null,
  "next_action": {"type": "follow_up_in_days|no_action", ...}
}
If should_send is false, set next_message to null and add a "reasoning" string.
"""


class SupportDomain(Domain):
    name = "support"

    # A different channel/consent model than leasing (push instead of sms/voice).
    def channel_consent_map(self) -> dict[str, str]:
        return {"email": "email_opt_in", "push": "push_opt_in"}

    def validate(self, record: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        if not record.get("task_id"):
            warnings.append("missing_task_id")
        if "consent" not in record:
            warnings.append("missing_consent")
        if not (record.get("input", {}) or {}).get("ticket_subject"):
            warnings.append("missing_ticket_subject")
        return warnings

    def build_decision_context(
        self,
        *,
        record: dict[str, Any],
        sanitized: dict[str, Any],
        tenant: dict[str, Any],
        dataset: list[dict[str, Any]],
    ) -> DecisionContext:
        inp = sanitized.get("input", {}) or {}
        profile = inp.get("profile", {}) or {}
        primary_cta = (record.get("assertions", {}) or {}).get("constraints", {}).get(
            "primary_cta", "resolve_ticket")
        facts = {
            "allowed_channels": self.consented_channels(record),
            "channel_preferences": record.get("channel_preferences") or [],
            "first_name": profile.get("first_name"),
            "ticket_subject": inp.get("ticket_subject"),
            "primary_cta": primary_cta,
        }
        guardrails = {"allowed_channels": facts["allowed_channels"], "primary_cta": primary_cta}
        user_prompt = (
            "Ticket (untrusted data):\n" + json.dumps(
                {"ticket_subject": inp.get("ticket_subject"), "first_name": profile.get("first_name")},
                ensure_ascii=False)
            + "\nFacts:\n" + json.dumps(facts, ensure_ascii=False)
            + "\nReturn STRICT JSON now.")
        lineage = {"task_id": record.get("task_id"), "domain": self.name}
        return DecisionContext(facts=facts, guardrails=guardrails,
                               system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt,
                               extras={"lineage": lineage})

    def normalize(self, output: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        guard = state.get("constraints", {})
        record = state.get("validated_record", {})
        allowed = guard.get("allowed_channels") or self.consented_channels(record)
        primary = guard.get("primary_cta", "resolve_ticket")
        warnings = list(state.get("warnings", []))

        if output.get("should_send") is False:
            return {"should_send": False, "next_message": None,
                    "next_action": output.get("next_action") or {"type": "no_action"},
                    "reasoning": output.get("reasoning") or "Agent decided not to send.",
                    "_warnings": warnings}
        if not allowed:
            return {"should_send": False, "next_message": None,
                    "next_action": {"type": "no_action"},
                    "reasoning": "No consented channel available."}

        msg = output.get("next_message") if isinstance(output.get("next_message"), dict) else {}
        channel = msg.get("channel")
        if channel not in allowed:
            channel = self.select_channel(record) or allowed[0]
            warnings.append(f"channel_repair: -> consented={channel!r}")
        msg["channel"] = channel

        body = msg.get("body")
        if not isinstance(body, str) or not body.strip():
            return {"_error": "empty or missing message body", "_warnings": warnings}
        if channel == "email":
            msg["subject"] = msg.get("subject") or "Following up on your support request"
            if "unsubscribe" not in body.lower():
                body = body.rstrip() + "\nTo opt out, reply STOP or click unsubscribe."
        else:
            msg["subject"] = None
        msg["body"] = body

        cta = msg.get("cta") if isinstance(msg.get("cta"), dict) else {}
        cta["type"] = primary
        msg["cta"] = cta

        na = output.get("next_action")
        if not isinstance(na, dict) or na.get("type") not in ("follow_up_in_days", "no_action"):
            na = {"type": "follow_up_in_days", "value": 2}
            warnings.append("next_action_repair: invalid type replaced")

        return {"should_send": True, "next_message": msg, "next_action": na, "_warnings": warnings}

    def evaluate_all(
        self, output: dict[str, Any], record: dict[str, Any], sanitized: dict[str, Any]
    ) -> list[dict[str, Any]]:
        msg = output.get("next_message") if isinstance(output.get("next_message"), dict) else None
        allowed = self.consented_channels(record)
        send = output.get("should_send")

        def crit(cid, severity, title, ok, detail=""):
            return {"id": cid, "severity": severity, "title": title, "pass": bool(ok), "detail": detail}

        results = []
        # SC-01: never send on a non-consented channel.
        results.append(crit(
            "SC-01", "critical", "Channel consented",
            (not send) or (msg and msg.get("channel") in allowed),
            f"channel={msg and msg.get('channel')} allowed={allowed}"))
        # SC-02: a send must carry a non-empty body.
        results.append(crit(
            "SC-02", "critical", "Body present on send",
            (not send) or (msg and isinstance(msg.get("body"), str) and msg["body"].strip())))
        # SC-03: a next_action type is always defined.
        na = output.get("next_action")
        results.append(crit(
            "SC-03", "high", "Next action defined",
            isinstance(na, dict) and na.get("type") in ("follow_up_in_days", "no_action")))
        return results


register_domain(SupportDomain())
