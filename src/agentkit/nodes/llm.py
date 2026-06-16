"""LLM node: async generation with a shared per-request time budget.

LLM-ONLY: there is no deterministic message fabrication. A single deadline
(TOTAL_LLM_BUDGET_S) is established on first entry and shared across retries so the
cumulative LLM time never exceeds the budget (guaranteeing p99 < 2s). On no-provider,
budget-exhaustion, timeout, or API error the node sets an `abort_reason` and the graph
routes to a safe no-send abort -- it never invents a message.

A circuit breaker (AC-11) protects the pipeline from a degraded backend: after repeated
failures it opens and calls fast-abort (LLM_CIRCUIT_OPEN) until a cooldown elapses.
Token metrics from each call are captured for the decision lineage (AC-12).
"""
from __future__ import annotations

import asyncio
import time

from .. import config
from .. import prompts
from .. import circuit_breaker
from .. import cost as _cost
from ..llm_client import LLMClient
from ..state import MessagingAgentState


def _make_client():
    """Select the LLM client. When ``AGENTKIT_MOCK_LLM`` is truthy, use the deterministic
    offline reference client (no network/key) — used by the CI eval smoke test and offline
    demos. Otherwise use the real provider-agnostic client."""
    import os
    if os.getenv("AGENTKIT_MOCK_LLM", "").lower() in ("1", "true", "yes"):
        from ..mock_llm import MockLLMClient
        return MockLLMClient()
    return LLMClient()


_client = _make_client()


def _breaker_for(state: MessagingAgentState):
    """Per-(provider, tenant) breaker so one tenant's degraded backend can't trip the
    circuit for every other tenant in the same process (G8)."""
    tenant_id = (state.get("constraints", {}) or {}).get("tenant_id") \
        or (state.get("tenant", {}) or {}).get("tenant_id")
    return circuit_breaker.get_breaker(getattr(_client, "provider", None), tenant_id)


def _lineage_with_usage(state: MessagingAgentState) -> dict:
    """Record the latest call's token usage and fold its cost into a running total that
    accumulates across retries (AC-12)."""
    lineage = dict(state.get("lineage", {}) or {})
    usage = getattr(_client, "last_usage", None)
    if usage:
        lineage["token_usage"] = usage
        lineage["cost"] = _cost.accumulate(lineage.get("cost"), usage)
    return lineage


async def llm_agent(state: MessagingAgentState) -> MessagingAgentState:
    retry = state.get("retry_count", 0)
    ctx = state.get("enriched_context", {})
    breaker = _breaker_for(state)

    if not _client.available:
        return {**state, "raw_llm_output": "", "used_fallback": False,
                "abort_reason": "LLM_UNAVAILABLE", "latency_ms": 0}

    # AC-11: if the breaker is open, fast-abort rather than wait on a dead backend.
    if not breaker.allow():
        return {**state, "raw_llm_output": "", "used_fallback": False,
                "abort_reason": "LLM_CIRCUIT_OPEN", "latency_ms": 0,
                "lineage": {**(state.get("lineage", {}) or {}),
                            "circuit_breaker": breaker.snapshot()}}

    # Establish a shared deadline once, on first entry, and carry it across retries.
    deadline = state.get("llm_deadline")
    if deadline is None:
        deadline = time.monotonic() + config.TOTAL_LLM_BUDGET_S
    remaining = deadline - time.monotonic()

    # Out of retries or out of budget -> abort (no fabrication).
    if retry >= config.MAX_RETRIES or remaining <= 0.05:
        return {**state, "raw_llm_output": "", "used_fallback": False,
                "llm_deadline": deadline,
                "abort_reason": "LLM_RETRIES_EXHAUSTED" if retry >= config.MAX_RETRIES
                else "LLM_BUDGET_EXHAUSTED",
                "latency_ms": state.get("latency_ms", 0)}

    attempt_timeout = min(config.LLM_TIMEOUT_S, remaining)
    t0 = time.time()

    # G3: on a retry, re-prompt with the prior (rejected) output + the specific error
    # so the model can self-correct instead of repeating itself. Temperature stays 0.0:
    # the correction prompt differs from the original, so a deterministic model already
    # produces a different (corrected) completion — no need to trade away reproducibility.
    user_prompt = ctx["user_prompt"]
    temperature = 0.0
    if retry > 0 and state.get("parse_error"):
        user_prompt = prompts.build_correction_prompt(
            ctx["user_prompt"], state.get("raw_llm_output", ""), state.get("parse_error", ""))

    try:
        text = await asyncio.wait_for(
            _client.generate(ctx["system_prompt"], user_prompt,
                             max_tokens=config.LLM_MAX_TOKENS, temperature=temperature),
            timeout=attempt_timeout,
        )
        breaker.record_success()
        return {**state, "raw_llm_output": text, "used_fallback": False,
                "llm_deadline": deadline,
                "token_usage": getattr(_client, "last_usage", None),
                "lineage": _lineage_with_usage(state),
                "latency_ms": int((time.time() - t0) * 1000)}
    except asyncio.TimeoutError:
        breaker.record_failure()
        return {**state, "raw_llm_output": "", "used_fallback": False,
                "llm_deadline": deadline, "abort_reason": "LLM_TIMEOUT",
                "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as exc:
        breaker.record_failure()
        return {**state, "raw_llm_output": "", "used_fallback": False,
                "llm_deadline": deadline,
                "abort_reason": f"LLM_ERROR:{type(exc).__name__}",
                "latency_ms": int((time.time() - t0) * 1000)}
