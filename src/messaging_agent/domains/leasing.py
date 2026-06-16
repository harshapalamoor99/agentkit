"""Leasing domain — the reference implementation of :class:`messaging_agent.domain.Domain`.

This packages the original multifamily-housing leasing agent's behavior behind the
pluggable domain seam: the system prompt, fact extraction, channel/consent model,
output compliance repairs and the AC-01..23 acceptance criteria. It delegates to the
existing leasing modules (``channels``, ``timing``, ``criteria``, ``tenants``, ``pii``,
``prompts``, ``safety_rules``) so behavior is identical to the pre-refactor agent.

It is registered as the **default** domain, so existing callers get unchanged behavior.
"""
from __future__ import annotations

import re
from typing import Any

from .. import channels, config, pii, prompts, safety_rules, tenants, timing
from ..criteria import evaluate_all as _criteria_evaluate_all
from ..domain import DecisionContext, Domain, register_domain

# Pricing-incentive language barred on rent-regulated (HUD/LIHTC) properties (AC-9).
_PRICING_INCENTIVE_RE = re.compile(
    r"\b(special|discount|\d+%\s*off|waived?\s+(fee|deposit)|free\s+(month|rent)|"
    r"move[- ]in\s+special|concession|promo(tion)?|deal|save\s+\$?\d+|"
    r"reduced\s+rent|rent\s+special|limited[- ]time)\b", re.IGNORECASE)


def _normalize_asset_class(record: dict) -> str:
    raw = record.get("asset_class") or (record.get("input", {}) or {}).get("asset_class")
    if not isinstance(raw, str):
        return "market_rate"
    norm = raw.strip().lower().replace(" ", "_")
    return norm or "market_rate"


def _ensure_opt_out(body: str, channel: str) -> str:
    low = body.lower()
    if channel == "sms":
        if "stop" not in low:
            body = body.rstrip() + " Reply STOP to opt out."
    else:
        if not any(k in low for k in ("unsubscribe", "opt out", "opt-out", "stop")):
            body = body.rstrip() + "\nTo opt out of these emails, click unsubscribe or reply STOP."
    return body


def _scan_text(text: str, guard: dict) -> str | None:
    """Output-safety scan shared by body and subject (G12).

    Returns a short reason code if ``text`` violates a hard rule (prompt-injection/markup,
    PII, toxic/fair-housing, foreign-tenant property, raw PII, or a pricing incentive on
    a regulated property), else None.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    if safety_rules._INJECTION_RE.search(text) or safety_rules._CODE_RE.search(text):
        return "prompt-injection or markup"
    if safety_rules.find_pii(text) or safety_rules.has_toxic(text):
        return "PII or fair-housing-violating content"
    if pii.output_reflects_raw_pii(text):
        return "raw PII reflected"
    low = text.lower()
    for fp in (guard.get("foreign_properties") or []):
        if isinstance(fp, str) and fp.lower() in low:
            return "foreign tenant property referenced"
    if guard.get("is_rent_regulated") and _PRICING_INCENTIVE_RE.search(text):
        return "pricing incentive on rent-regulated property"
    return None


class LeasingDomain(Domain):
    name = "leasing"

    # --- Consent / channels (delegate to the leasing channels module) ---
    def channel_consent_map(self) -> dict[str, str]:
        return dict(config.CHANNEL_CONSENT_FLAG)

    def consented_channels(self, record: dict[str, Any]) -> list[str]:
        return channels.consented_channels(record)

    def select_channel(self, record: dict[str, Any]) -> str | None:
        return channels.select_channel(record)

    # --- Intake validation ---
    def validate(self, record: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        if not record.get("task_id"):
            warnings.append("missing_task_id")
        if "consent" not in record:
            warnings.append("missing_consent")
        inp = record.get("input", {}) or {}
        if not inp.get("timezone"):
            warnings.append("missing_timezone")
        profile = inp.get("profile", {}) or {}
        if not profile.get("first_name"):
            warnings.append("missing_first_name")
        if not inp.get("move_date_target"):
            warnings.append("missing_move_date_target")
        if not record.get("channel_preferences"):
            warnings.append("empty_channel_preferences")
        return warnings

    # --- Context / prompting ---
    def build_decision_context(
        self,
        *,
        record: dict[str, Any],
        sanitized: dict[str, Any],
        tenant: dict[str, Any],
        dataset: list[dict[str, Any]],
    ) -> DecisionContext:
        asset_class = _normalize_asset_class(record)
        is_regulated = asset_class in config.REGULATED_ASSET_CLASSES

        allowed = channels.consented_channels(record)
        inp = record.get("input", {}) or {}
        constraints_cfg = (record.get("assertions", {}) or {}).get("constraints", {}) or {}
        primary_cta = constraints_cfg.get("primary_cta", "book_tour")

        sprofile = (sanitized.get("input", {}) or {}).get("profile", {}) or {}
        interests = []
        amen = sprofile.get("amenity_interest")
        if isinstance(amen, list):
            interests.extend(str(a) for a in amen if a)
        elif amen:
            interests.append(str(amen))
        if sprofile.get("city_interest"):
            interests.append(str(sprofile["city_interest"]))

        facts = {
            "allowed_channels": allowed,
            "channel_preferences": record.get("channel_preferences") or [],
            "first_name": sprofile.get("first_name"),
            "interests": interests,
            "timezone": inp.get("timezone"),
            "business_hours_local": [config.BUSINESS_HOUR_START, config.BUSINESS_HOUR_END],
            "tcpa_window_local": [config.TCPA_EARLIEST_HOUR, config.TCPA_LATEST_HOUR],
            "last_interaction": inp.get("last_interaction"),
            "move_date_target": inp.get("move_date_target"),
            "primary_cta": primary_cta,
            "brand_voice": tenant.get("brand_voice"),
            "legal_footer": tenant.get("legal_footer"),
            "asset_class": asset_class,
            "is_rent_regulated": is_regulated,
            "screening_metadata": (sanitized.get("input", {}) or {}).get("screening_metadata"),
        }

        guardrails = {
            "allowed_channels": allowed,
            "timezone": inp.get("timezone"),
            "primary_cta": primary_cta,
            "include_opt_out": constraints_cfg.get("include_opt_out_instructions", True),
            "no_pii_leak": constraints_cfg.get("no_pii_leak", True),
            "tenant_id": tenant.get("tenant_id"),
            "foreign_properties": tenants.foreign_property_names(record),
            "asset_class": asset_class,
            "is_rent_regulated": is_regulated,
        }

        # AC-10: few-shot examples isolated to this tenant's own records.
        own_dataset = [r for r in dataset if tenants.same_tenant(record, r)]
        examples, example_ids = prompts.build_examples_with_ids(
            own_dataset, exclude_task_id=record.get("task_id"), target_record=record,
            fallback_pool=prompts.load_canonical_examples())

        user_prompt = prompts.build_user_prompt(sanitized, facts, examples)

        lineage = {
            "task_id": record.get("task_id"),
            "tenant_id": tenant.get("tenant_id"),
            "asset_class": asset_class,
            "prompt_template_version": config.PROMPT_TEMPLATE_VERSION,
            "few_shot_example_ids": example_ids,
            "input_snapshot": prompts._trim_for_prompt(sanitized),
        }

        return DecisionContext(
            facts=facts,
            guardrails=guardrails,
            system_prompt=prompts.SYSTEM_PROMPT,
            user_prompt=user_prompt,
            example_ids=example_ids,
            extras={"asset_class": asset_class, "lineage": lineage},
        )

    # --- Output validation / compliance repair ---
    def normalize(self, output: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Validate the LLM's decisions and repair ONLY for legal compliance.

        The LLM owns the decisions (channel, timing, copy, next_action). Here we verify
        they don't break a hard compliance rule and minimally repair if they do.
        """
        guard = state.get("constraints", {})
        record = state.get("validated_record", state.get("sanitized_record", {}))
        allowed = guard.get("allowed_channels") or channels.consented_channels(record)
        primary = guard.get("primary_cta", "book_tour")
        cta_type = "schedule_tour" if primary in ("book_tour", "schedule_tour") else primary
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

        msg = output.get("next_message")
        if not isinstance(msg, dict):
            msg = {}

        channel = msg.get("channel")
        if channel not in allowed:
            repaired = channels.select_channel(record)
            warnings.append(f"channel_repair: llm={channel!r} -> consented={repaired!r}")
            channel = repaired
        msg["channel"] = channel

        send_at = msg.get("send_at")
        if not (isinstance(send_at, str) and timing.is_tz_aware(send_at)
                and timing.in_business_hours(send_at)):
            repaired = timing.compute_send_at(record, channel)
            if repaired:
                warnings.append(f"send_at_repair: llm={send_at!r} -> {repaired!r}")
                send_at = repaired
        if send_at:
            msg["send_at"] = send_at

        if channel == "sms":
            msg["subject"] = None
        else:
            prop = (state.get("sanitized_record", {}).get("input", {}) or {}).get("property_name", "our community")
            safe_subject = f"Tour {prop} — book your visit"
            subj = msg.get("subject")
            if not (isinstance(subj, str) and subj.strip()):
                msg["subject"] = safe_subject
                warnings.append("subject_repair: empty email subject filled")
            else:
                subj = safety_rules.clean_text(subj, max_len=200)
                reason = _scan_text(subj, guard)
                if reason:
                    warnings.append(f"subject_quarantine: {reason} -> safe subject")
                    subj = safe_subject
                msg["subject"] = subj

        body = msg.get("body")
        if not isinstance(body, str) or not body.strip():
            return {"_error": "empty or missing message body"}
        body = safety_rules.clean_text(body, max_len=5000)
        body_reason = _scan_text(body, guard)
        if body_reason:
            warnings.append(f"body_quarantine: {body_reason}")
            return {"_error": f"body contained {body_reason}", "_warnings": warnings}

        tenant = state.get("tenant", {}) or {}
        footer = tenant.get("legal_footer")
        if isinstance(footer, str) and footer.strip() and footer.strip().lower() not in body.lower():
            candidate = body.rstrip() + " " + footer.strip()
            if channel != "sms" or len(candidate) <= 320:
                body = candidate
                warnings.append("legal_footer_repair: appended missing legal footer")

        body = _ensure_opt_out(body, channel)
        msg["body"] = body

        facts = (state.get("enriched_context", {}) or {}).get("facts", {}) or {}
        interests = [str(i).lower() for i in (facts.get("interests") or []) if i]
        low_body = body.lower()
        interest_reflected = (not interests) or any(
            any(tok in low_body for tok in i.split()) for i in interests)
        output["_interest_reflected"] = interest_reflected
        if interests and not interest_reflected:
            warnings.append("interest_not_reflected")

        cta = msg.get("cta")
        if not isinstance(cta, dict):
            cta = {}
        cta["type"] = cta_type
        msg["cta"] = cta

        output["should_send"] = True
        output["next_message"] = msg

        na = output.get("next_action")
        if not isinstance(na, dict) or na.get("type") not in ("start_cadence", "follow_up_in_days", "no_action"):
            na = {"type": "follow_up_in_days", "value": 3}
            warnings.append("next_action_repair: invalid type replaced")
        if na.get("type") == "start_cadence" and not na.get("name"):
            na["name"] = "prospect_welcome_short_horizon"
        output["next_action"] = na
        output["_warnings"] = warnings
        return output

    # --- Evaluation ---
    def evaluate_all(
        self, output: dict[str, Any], record: dict[str, Any], sanitized: dict[str, Any]
    ) -> list[dict[str, Any]]:
        return _criteria_evaluate_all(output, record, sanitized)

    # --- Telemetry: preserve the leasing-rich, non-PII feature vector ---
    def telemetry_features(self, record: dict[str, Any]) -> dict[str, Any]:
        inp = record.get("input", {}) or {}
        profile = inp.get("profile", {}) or {}
        return {
            "domain": self.name,
            "persona": record.get("persona"),
            "lifecycle_stage": record.get("lifecycle_stage"),
            "asset_class": record.get("asset_class") or inp.get("asset_class"),
            "channel_preferences": record.get("channel_preferences"),
            "consent": record.get("consent"),
            "move_date_target": inp.get("move_date_target"),
            "interests": {
                "amenity_interest": profile.get("amenity_interest"),
                "city_interest": profile.get("city_interest"),
            },
        }

    # --- Knowledge / RAG: enabled by default from the bundled leasing KB; override the
    # source with LEASING_KNOWLEDGE_PATH, or disable with LEASING_KNOWLEDGE_PATH="". ---
    def knowledge_base(self):
        import os
        from .. import knowledge as _kb
        existing = _kb.get_knowledge_base("leasing")
        if existing is not None:
            return existing
        env_path = os.getenv("LEASING_KNOWLEDGE_PATH")
        if env_path is not None:
            # Explicitly set (possibly empty -> disabled).
            if not env_path.strip():
                return None
            return _kb.build_in_memory_kb_from_env("LEASING_KNOWLEDGE_PATH", name="leasing")
        # Default: the bundled tenant-tagged FAQ bank.
        default_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "data", "knowledge", "leasing_kb.jsonl")
        if os.path.exists(default_path):
            kb = _kb.InMemoryKnowledgeBase.from_jsonl(default_path, name="leasing")
            _kb.register_knowledge_base(kb, name="leasing")
            return kb
        return None

    def knowledge_query(self, record: dict[str, Any], facts: dict[str, Any]) -> str | None:
        parts = []
        if facts.get("asset_class"):
            parts.append(str(facts["asset_class"]))
        for i in (facts.get("interests") or []):
            parts.append(str(i))
        if facts.get("primary_cta"):
            parts.append(str(facts["primary_cta"]))
        prop = (record.get("input", {}) or {}).get("property_name")
        if prop:
            parts.append(str(prop))
        return " ".join(parts) or None

    def knowledge_filter(self, record: dict[str, Any]) -> dict[str, Any] | None:
        tid = record.get("tenant_id") or (record.get("input", {}) or {}).get("tenant_id")
        return {"tenant_id": tid} if tid else None


register_domain(LeasingDomain(), default=True)
