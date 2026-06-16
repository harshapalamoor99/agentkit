"""Safety node: scan untrusted input for injection / jailbreak / toxic content."""
from __future__ import annotations

from .. import safety_rules
from ..state import MessagingAgentState


def safety_agent(state: MessagingAgentState) -> MessagingAgentState:
    record = state.get("validated_record", {})
    violations = safety_rules.scan_record(record)
    severity = safety_rules.classify_severity(violations)
    return {
        **state,
        "safety_violations": violations,
        "safety_severity": severity,
    }
