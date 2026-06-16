"""Evaluate node: run all acceptance criteria against the produced output."""
from __future__ import annotations

from ..domain import get_domain
from ..state import MessagingAgentState


def evaluator_agent(state: MessagingAgentState) -> MessagingAgentState:
    output = state.get("parsed_output", {})
    record = state.get("validated_record", {})
    sanitized = state.get("sanitized_record", record)

    results = get_domain(state).evaluate_all(output, record, sanitized)
    passed = sum(1 for r in results if r["pass"])
    critical_fails = [r for r in results if not r["pass"] and r["severity"] == "critical"]

    # G1: separate the LLM's RAW decision quality from post-repair compliance.
    repairs = state.get("repairs", []) or []
    return {
        **state,
        "ac_results": results,
        "evaluation": {
            "passed": passed,
            "total": len(results),
            "score": f"{passed}/{len(results)}",
            "critical_fails": critical_fails,
            "all_critical_pass": not critical_fails,
            # The post-repair output that ships (what the ACs above score).
            "repairs": repairs,
            "repairs_count": len(repairs),
            # True iff the model's own output needed no compliance repair — i.e. the
            # LLM, not the deterministic guardrail, produced a compliant decision.
            "llm_raw_compliant": len(repairs) == 0,
            "llm_raw_decision": state.get("llm_raw_decision"),
        },
    }
