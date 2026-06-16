"""Output parser node (domain-agnostic orchestration).

The LLM is trusted only for *content*. This node extracts JSON, asks the active
:class:`~messaging_agent.domain.Domain` to validate the decision and minimally repair it
for hard compliance (``domain.normalize``), snapshots the model's raw decision (G1),
audits unjustified no-sends (G9), and requests a corrective retry on unusable output.
Once retries or the shared time budget are exhausted the LLM node aborts to a safe
no-send (LLM-only: never a fabricated message).

All domain-specific compliance logic (channel/timing/opt-out/subject repair, output
PII/fair-housing scanning) lives in the domain's ``normalize`` method, keeping this node
reusable across domains.
"""
from __future__ import annotations

import json
import re

from ..domain import get_domain
from ..state import MessagingAgentState

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    candidates = []
    m = _FENCE_RE.search(text)
    if m:
        candidates.append(m.group(1))
    candidates.append(text)
    # Greedy outermost-object fallback.
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            continue
    return None


_REPAIR_MARKERS = ("_repair", "_quarantine")


def _suppression_audit(parsed: dict, state: MessagingAgentState) -> dict:
    """Flag (and audit) an LLM no-send that is NOT explained by a compliance veto (G9).

    A no-send is *justified* when the recipient has no consented channel. If the model
    declines to send while a consented channel exists, that is a potential silent
    drop-off: we record it and emit a telemetry event so false no-sends are observable
    rather than invisible. Returns a small audit dict stored on state.
    """
    guard = state.get("constraints", {}) or {}
    record = state.get("validated_record", state.get("sanitized_record", {})) or {}
    allowed = guard.get("allowed_channels") or get_domain(state).consented_channels(record)
    unjustified = bool(allowed)  # consent exists yet the model chose not to send
    audit = {
        "no_send": True,
        "unjustified": unjustified,
        "allowed_channels": allowed,
        "reason": parsed.get("reasoning") or "",
    }
    if unjustified:
        try:
            from .. import telemetry
            telemetry.store.emit_event("suppressed_send", {
                "task_id": state.get("task_id") or record.get("task_id"),
                "tenant_id": guard.get("tenant_id"),
                "allowed_channels": allowed,
                "llm_reason": audit["reason"],
            })
        except Exception:
            pass
    return audit


def _repairs_from(warnings: list[str], baseline: list[str]) -> list[str]:
    """Repair/quarantine warnings added during this normalization pass (G1)."""
    new = warnings[len(baseline):] if len(warnings) >= len(baseline) else warnings
    return [w for w in new if any(m in w for m in _REPAIR_MARKERS)]


async def output_parser(state: MessagingAgentState) -> MessagingAgentState:
    raw = state.get("raw_llm_output", "")
    baseline_warnings = list(state.get("warnings", []))
    parsed = _extract_json(raw)
    if parsed is None:
        return {**state, "parse_status": "retry",
                "parse_error": "response was not valid, parseable JSON",
                "retry_count": state.get("retry_count", 0) + 1}

    # G1: snapshot the LLM's RAW decision (pre-repair) so evaluation can score the
    # model's own correctness separately from post-repair compliance.
    llm_raw_decision = {
        "should_send": parsed.get("should_send"),
        "channel": (parsed.get("next_message") or {}).get("channel")
        if isinstance(parsed.get("next_message"), dict) else None,
        "cta_type": ((parsed.get("next_message") or {}).get("cta") or {}).get("type")
        if isinstance(parsed.get("next_message"), dict) else None,
        "send_at": (parsed.get("next_message") or {}).get("send_at")
        if isinstance(parsed.get("next_message"), dict) else None,
        "next_action_type": (parsed.get("next_action") or {}).get("type")
        if isinstance(parsed.get("next_action"), dict) else None,
    }

    normalized = get_domain(state).normalize(parsed, state)
    error = normalized.pop("_error", None)
    warnings = normalized.pop("_warnings", state.get("warnings", []))
    interest_reflected = normalized.pop("_interest_reflected", None)
    repairs = _repairs_from(warnings, baseline_warnings)

    # A legitimate no-send decision is a valid, successful output.
    if normalized.get("should_send") is False:
        suppression = _suppression_audit(parsed, state)
        return {**state, "parsed_output": normalized, "warnings": warnings,
                "llm_raw_decision": llm_raw_decision, "repairs": repairs,
                "suppression": suppression,
                "should_send": False, "reasoning": normalized.get("reasoning", ""),
                "parse_status": "success"}

    msg = normalized.get("next_message") or {}
    if error or not normalized or not msg.get("body"):
        return {**state, "warnings": warnings, "parse_status": "retry",
                "parse_error": error or "output did not contain a usable message body",
                "retry_count": state.get("retry_count", 0) + 1}

    # --- G6: optional runtime fair-housing LLM-judge veto (default OFF). The regex
    # floor in normalize() always runs; when enabled this adds a model-based check that
    # can only force a SAFE no-send (never invent or alter copy). Failures are no-ops. ---
    from .. import config as _cfg
    if _cfg.RUNTIME_FAIRHOUSING_JUDGE:
        from ..evals import judge as _judge
        from . import llm as _llmnode
        verdict = await _judge.fairness_gate(msg.get("body", ""), client=_llmnode._client)
        if verdict.get("violation"):
            reason = ("Suppressed by runtime fair-housing judge: "
                      + (verdict.get("reason") or "potential protected-class violation"))
            warnings = warnings + [f"fairhousing_judge_quarantine: {verdict.get('reason')!r}"]
            no_send = {"should_send": False, "next_message": None,
                       "next_action": {"type": "no_action"}, "reasoning": reason}
            return {**state, "parsed_output": no_send, "warnings": warnings,
                    "llm_raw_decision": llm_raw_decision, "repairs": repairs,
                    "should_send": False, "reasoning": reason, "parse_status": "success"}

    # G13: surface the (non-blocking) interest-weaving signal; emit telemetry when an
    # instructed interest was not reflected so soft misses are observable, not invisible.
    if interest_reflected is False:
        try:
            from .. import telemetry
            telemetry.store.emit_event("interest_not_reflected", {
                "task_id": state.get("task_id"),
                "tenant_id": (state.get("constraints", {}) or {}).get("tenant_id"),
            })
        except Exception:
            pass

    return {**state, "parsed_output": normalized, "warnings": warnings,
            "llm_raw_decision": llm_raw_decision, "repairs": repairs,
            "interest_reflected": interest_reflected,
            "parse_status": "success"}
