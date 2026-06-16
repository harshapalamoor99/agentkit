"""Sanitize node: produce a cleaned record with only safe personalization data."""
from __future__ import annotations

from .. import safety_rules
from ..state import MessagingAgentState


def sanitize_node(state: MessagingAgentState) -> MessagingAgentState:
    record = state.get("validated_record", {})
    return {**state, "sanitized_record": safety_rules.sanitize_record(record)}
