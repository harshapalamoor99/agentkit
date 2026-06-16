"""Intake node: parse, validate, and check consent."""
from __future__ import annotations

import json
from typing import Any

from ..domain import get_domain
from ..state import MessagingAgentState


def _parse(state: MessagingAgentState) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    record = state.get("record")
    if record is None:
        raw = state.get("raw_line", "")
        try:
            record = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            warnings.append(f"unparseable_record: {exc}")
            return {}, warnings
    if not isinstance(record, dict):
        warnings.append("record_not_object")
        return {}, warnings
    return record, warnings


def _validate(record: dict[str, Any]) -> list[str]:
    return get_domain({"validated_record": record}).validate(record)


def intake_agent(state: MessagingAgentState) -> MessagingAgentState:
    record, warnings = _parse(state)
    if not record:
        return {
            **state,
            "validated_record": {},
            "warnings": warnings,
            "should_send": False,
            "abort_reason": "BAD_RECORD",
            "reasoning": "Input could not be parsed into a valid record.",
        }

    warnings += _validate(record)
    has_consent = bool(get_domain({"validated_record": record}).consented_channels(record))
    return {
        **state,
        "task_id": record.get("task_id", state.get("task_id", "unknown")),
        "validated_record": record,
        "warnings": warnings,
        "should_send": has_consent,
        "abort_reason": "" if has_consent else "NO_CONSENT",
        "reasoning": "" if has_consent
        else "User has not opted in to any channel; sending would violate consent.",
    }
