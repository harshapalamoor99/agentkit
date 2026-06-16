"""Emit terminal node: assemble the final successful output."""
from __future__ import annotations

from datetime import datetime, timezone

from ..state import MessagingAgentState


def emit_result(state: MessagingAgentState) -> MessagingAgentState:
    output = state.get("parsed_output", {})
    final = {
        "task_id": state.get("task_id"),
        "should_send": output.get("should_send", True),
        "next_message": output.get("next_message"),
        "next_action": output.get("next_action"),
        "reasoning": output.get("reasoning", ""),
        "evaluation": state.get("evaluation"),
        "ac_results": state.get("ac_results", []),
        "warnings": state.get("warnings", []),
        "safety_violations": state.get("safety_violations", []),
        "suppression": state.get("suppression"),
        "interest_reflected": state.get("interest_reflected"),
        "used_fallback": state.get("used_fallback", False),
        "knowledge": state.get("knowledge", []),
        "tenant_id": (state.get("tenant") or {}).get("tenant_id"),
        "asset_class": state.get("asset_class"),
        "lineage": state.get("lineage", {}),
        "token_usage": state.get("token_usage"),
        "cost": (state.get("lineage", {}) or {}).get("cost"),
        "latency_ms": state.get("latency_ms", 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return {**state, "final_output": final}
