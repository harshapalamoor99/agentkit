"""Shared state schema for the messaging agent LangGraph."""
from __future__ import annotations

from typing import Any, TypedDict


class MessagingAgentState(TypedDict, total=False):
    # --- Input ---
    task_id: str
    raw_line: str
    record: dict[str, Any]
    dataset: list[dict[str, Any]]  # all records, used for few-shot learning

    # --- Context decision (deterministic constraints) ---
    constraints: dict[str, Any]

    # --- Intake ---
    validated_record: dict[str, Any]
    warnings: list[str]

    # --- Safety / sanitize ---
    safety_violations: list[dict[str, Any]]
    safety_severity: str  # "clean" | "sanitizable" | "block"
    sanitized_record: dict[str, Any]
    pii_notes: list[str]  # AC-8: tokenization audit notes

    # --- Context ---
    enriched_context: dict[str, Any]
    tenant: dict[str, Any]       # AC-7: resolved brand metadata
    asset_class: str             # AC-9: normalized asset class
    knowledge: list[dict[str, Any]]  # RAG: retrieved snippets used to ground the LLM

    # --- LLM ---
    raw_llm_output: str
    parsed_output: dict[str, Any]
    retry_count: int
    parse_status: str  # "success" | "retry" | "failed"
    parse_error: str   # G3: reason the last output was rejected (drives corrective retry)
    llm_raw_decision: dict[str, Any]  # G1: model's pre-repair decision snapshot
    repairs: list[str]  # G1: compliance repairs/quarantines applied to the LLM output
    suppression: dict[str, Any]  # G9: audit of an LLM no-send (justified vs unjustified)
    interest_reflected: bool  # G13: whether an instructed interest was woven into the body
    used_fallback: bool
    llm_deadline: float  # monotonic deadline shared across retries (latency budget)
    token_usage: dict[str, Any]  # AC-12: prompt/completion token metrics
    lineage: dict[str, Any]      # AC-12: decision lineage snapshot

    # --- Evaluation ---
    ac_results: list[dict[str, Any]]
    evaluation: dict[str, Any]

    # --- Control / output ---
    should_send: bool
    abort_reason: str
    reasoning: str
    final_output: dict[str, Any]
    latency_ms: int
    timestamp: str
