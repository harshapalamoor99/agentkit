"""Abort terminal node: produce a safe no-send output with reasoning."""
from __future__ import annotations

from datetime import datetime, timezone

from .. import criteria  # noqa: F401  (kept for backward-compatible import surface)
from ..domain import get_domain
from ..state import MessagingAgentState


def abort_handler(state: MessagingAgentState) -> MessagingAgentState:
    reason = state.get("abort_reason", "ABORTED")
    reasoning = state.get("reasoning") or f"Aborted: {reason}"
    output = {
        "should_send": False,
        "next_message": None,
        "next_action": {"type": "no_action"},
        "reasoning": reasoning,
    }
    record = state.get("validated_record", {})
    sanitized = state.get("sanitized_record", record)
    results = get_domain(state).evaluate_all(output, record, sanitized)
    passed = sum(1 for r in results if r["pass"])
    final = {
        "task_id": state.get("task_id"),
        "should_send": False,
        "next_message": None,
        "next_action": {"type": "no_action"},
        "abort_reason": reason,
        "reasoning": reasoning,
        "evaluation": {
            "passed": passed,
            "total": len(results),
            "score": f"{passed}/{len(results)}",
            "critical_fails": [r for r in results if not r["pass"] and r["severity"] == "critical"],
        },
        "ac_results": results,
        "warnings": state.get("warnings", []),
        "safety_violations": state.get("safety_violations", []),
        "tenant_id": (state.get("tenant") or {}).get("tenant_id"),
        "asset_class": state.get("asset_class"),
        "lineage": state.get("lineage", {}),
        "latency_ms": state.get("latency_ms", 0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return {**state, "parsed_output": output, "ac_results": results, "final_output": final}
