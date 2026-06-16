"""LangSmith tracing hook.

Tracing is enabled purely via environment (no code change) so it can be turned on in
prod and off in tests:

    export LANGCHAIN_TRACING_V2=true
    export LANGCHAIN_API_KEY=ls-...
    export LANGCHAIN_PROJECT=agentkit

LangGraph/LangChain pick these up automatically. This module just reports status and
provides a per-invocation run config (tags + metadata) to attach to `app.ainvoke`.
"""
from __future__ import annotations

import os
from typing import Any


def tracing_enabled() -> bool:
    return os.getenv("LANGCHAIN_TRACING_V2", "").lower() in ("1", "true", "yes")


def run_config(task_id: str, **metadata: Any) -> dict[str, Any]:
    """Config dict to pass as `app.ainvoke(state, config=run_config(...))`."""
    return {
        "run_name": f"agentkit:{task_id}",
        "tags": ["agentkit", f"task:{task_id}"],
        "metadata": {"task_id": task_id, **metadata},
    }


def status() -> dict[str, Any]:
    return {
        "tracing_enabled": tracing_enabled(),
        "project": os.getenv("LANGCHAIN_PROJECT", "default"),
        "has_api_key": bool(os.getenv("LANGCHAIN_API_KEY")),
    }
