"""Token + cost accounting for LLM usage.

Turns the raw token counts captured by :class:`~agentkit.llm_client.LLMClient`
(``last_usage``) into a USD cost using a small, overridable price table, and accumulates
usage across retries so the decision lineage carries a single authoritative total.

Prices are expressed in **USD per 1M tokens** (the unit every major provider publishes).
The built-in table covers common defaults; override or extend it without code changes:

    # JSON: {"model-name": {"input": <usd_per_1m>, "output": <usd_per_1m>}, ...}
    export LLM_PRICE_TABLE='{"gpt-4o-mini": {"input": 0.15, "output": 0.60}}'
    export LLM_PRICE_TABLE_PATH=/etc/agentkit/prices.json

Unknown models cost 0.0 (and are flagged via ``priced=False``) rather than raising, so
cost accounting never breaks a decision. Everything here is pure-python, no deps.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any

# USD per 1,000,000 tokens. Conservative public list prices; override via env.
_DEFAULT_PRICES: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o4-mini": {"input": 1.10, "output": 4.40},
    # Anthropic
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku": {"input": 0.80, "output": 4.00},
    "claude-3-haiku": {"input": 0.25, "output": 1.25},
    # Google Gemini
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
}


@lru_cache(maxsize=1)
def _price_table() -> dict[str, dict[str, float]]:
    table = dict(_DEFAULT_PRICES)
    path = os.getenv("LLM_PRICE_TABLE_PATH")
    if path:
        try:
            with open(path, encoding="utf-8") as f:
                table.update(json.load(f))
        except Exception:
            pass
    raw = os.getenv("LLM_PRICE_TABLE")
    if raw:
        try:
            table.update(json.loads(raw))
        except Exception:
            pass
    return table


def reset_price_cache() -> None:
    """Clear the memoized price table (tests / hot-reload after env changes)."""
    _price_table.cache_clear()


def _match_price(model: str | None) -> dict[str, float] | None:
    if not model:
        return None
    table = _price_table()
    if model in table:
        return table[model]
    # Tolerate provider/date suffixes, e.g. "claude-3-5-sonnet-20241022",
    # "gpt-4o-2024-08-06", "models/gemini-2.0-flash".
    base = model.split("/")[-1]
    if base in table:
        return table[base]
    best = None
    for key in table:
        if base.startswith(key) and (best is None or len(key) > len(best)):
            best = key
    return table[best] if best else None


def estimate_cost(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Cost a single ``last_usage`` dict.

    Returns ``{prompt_tokens, completion_tokens, total_tokens, input_cost_usd,
    output_cost_usd, cost_usd, model, priced}``. ``priced`` is False when the model is
    not in the price table (cost falls back to 0.0).
    """
    usage = usage or {}
    model = usage.get("model")
    prompt = int(usage.get("prompt_tokens") or 0)
    completion = int(usage.get("completion_tokens") or 0)
    total = int(usage.get("total_tokens") or (prompt + completion))
    price = _match_price(model)
    if price is None:
        return {
            "model": model, "prompt_tokens": prompt, "completion_tokens": completion,
            "total_tokens": total, "input_cost_usd": 0.0, "output_cost_usd": 0.0,
            "cost_usd": 0.0, "priced": False,
        }
    input_cost = prompt / 1_000_000 * float(price.get("input", 0.0))
    output_cost = completion / 1_000_000 * float(price.get("output", 0.0))
    return {
        "model": model, "prompt_tokens": prompt, "completion_tokens": completion,
        "total_tokens": total,
        "input_cost_usd": round(input_cost, 8),
        "output_cost_usd": round(output_cost, 8),
        "cost_usd": round(input_cost + output_cost, 8),
        "priced": True,
    }


def accumulate(running: dict[str, Any] | None, usage: dict[str, Any] | None) -> dict[str, Any]:
    """Fold one call's ``usage`` into a running total (across LLM retries).

    The running total has the same shape as :func:`estimate_cost` plus a ``calls``
    counter. ``model`` reflects the most recent call. Pass ``running=None`` to start.
    """
    one = estimate_cost(usage)
    if not running:
        return {**one, "calls": 1 if usage else 0}
    return {
        "model": one.get("model") or running.get("model"),
        "prompt_tokens": running.get("prompt_tokens", 0) + one["prompt_tokens"],
        "completion_tokens": running.get("completion_tokens", 0) + one["completion_tokens"],
        "total_tokens": running.get("total_tokens", 0) + one["total_tokens"],
        "input_cost_usd": round(running.get("input_cost_usd", 0.0) + one["input_cost_usd"], 8),
        "output_cost_usd": round(running.get("output_cost_usd", 0.0) + one["output_cost_usd"], 8),
        "cost_usd": round(running.get("cost_usd", 0.0) + one["cost_usd"], 8),
        "priced": running.get("priced", False) and one["priced"],
        "calls": running.get("calls", 0) + (1 if usage else 0),
    }
