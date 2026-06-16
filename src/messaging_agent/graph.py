"""LangGraph assembly for the context-aware messaging agent.

Flow:
    intake ──┬─ ok ───────► safety ──┬─ clean ─────────► context ─► llm ─► parse
             ├─ no_consent ─► abort  ├─ sanitized ─► sanitize ─► context
             └─ bad_record ─► abort  └─ block ─────────► abort

    parse ──┬─ success ─► evaluate ─► emit ─► END
            └─ retry ───► llm   (until MAX_RETRIES or budget; then llm aborts)

There is no fabricated fallback: an unusable output is retried via the LLM node,
which itself routes to `abort` (safe no-send) once retries or the shared time budget
are exhausted (LLM-only).

Each agent is an independently testable function; LangGraph provides the
orchestration, retry-with-state, and (via callbacks/LangSmith) the audit trail.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .nodes.abort import abort_handler
from .nodes.context import context_builder
from .nodes.emit import emit_result
from .nodes.evaluate import evaluator_agent
from .nodes.intake import intake_agent
from .nodes.llm import llm_agent
from .nodes.parse_output import output_parser
from .nodes.safety import safety_agent
from .nodes.sanitize import sanitize_node
from .observability import instrument_node
from .routing import route_intake, route_llm, route_parse, route_safety
from .state import MessagingAgentState


def build_graph():
    graph = StateGraph(MessagingAgentState)

    graph.add_node("intake", instrument_node("intake", intake_agent))
    graph.add_node("safety", instrument_node("safety", safety_agent))
    graph.add_node("sanitize", instrument_node("sanitize", sanitize_node))
    graph.add_node("context", instrument_node("context", context_builder))
    graph.add_node("llm", instrument_node("llm", llm_agent))
    graph.add_node("parse_output", instrument_node("parse_output", output_parser))
    graph.add_node("evaluate", instrument_node("evaluate", evaluator_agent))
    graph.add_node("emit", instrument_node("emit", emit_result))
    graph.add_node("abort", instrument_node("abort", abort_handler))

    graph.set_entry_point("intake")

    graph.add_conditional_edges("intake", route_intake, {
        "ok": "safety",
        "no_consent": "abort",
        "bad_record": "abort",
    })
    graph.add_conditional_edges("safety", route_safety, {
        "clean": "context",
        "sanitized": "sanitize",
        "block": "abort",
    })
    graph.add_edge("sanitize", "context")
    graph.add_edge("context", "llm")
    graph.add_conditional_edges("llm", route_llm, {
        "parse": "parse_output",
        "abort": "abort",
    })
    graph.add_conditional_edges("parse_output", route_parse, {
        "success": "evaluate",
        "retry": "llm",
    })
    graph.add_edge("evaluate", "emit")
    graph.add_edge("emit", END)
    graph.add_edge("abort", END)

    return graph.compile()


app = build_graph()
