"""Conditional-edge routing functions for the LangGraph."""
from __future__ import annotations

from .state import MessagingAgentState


def route_intake(state: MessagingAgentState) -> str:
    reason = state.get("abort_reason")
    if reason == "BAD_RECORD":
        return "bad_record"
    if reason == "NO_CONSENT":
        return "no_consent"
    return "ok"


def route_safety(state: MessagingAgentState) -> str:
    severity = state.get("safety_severity", "clean")
    if severity == "block":
        return "block"
    if severity == "sanitizable":
        return "sanitized"
    return "clean"


def route_parse(state: MessagingAgentState) -> str:
    status = state.get("parse_status")
    if status == "success":
        return "success"
    # Malformed/unusable output -> retry via the LLM node, which itself aborts once
    # retries or the time budget are exhausted (LLM-only: no fabricated fallback).
    return "retry"


def route_llm(state: MessagingAgentState) -> str:
    """After the LLM node: abort on any failure, else parse the output."""
    if state.get("abort_reason"):
        return "abort"
    if state.get("raw_llm_output"):
        return "parse"
    return "abort"
