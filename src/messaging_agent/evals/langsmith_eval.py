"""LangSmith evaluation adapter.

Runs the **same** scorers as the offline harness (``personalization``, ``semantic``,
``judge``, RAG ``knowledge_recall``, and the acceptance-criteria critical-fail gate) but
as a hosted LangSmith *experiment*, so you get trace-linked scores and version-to-version
comparison in the LangSmith UI.

Design goals:
* **One scoring implementation.** The pure ``score_*`` functions below are the single
  source of truth; both the LangSmith evaluators here and (conceptually) the local
  harness call the same underlying modules.
* **No-op without credentials.** Importing this module never requires ``langsmith``; every
  network entry point checks :func:`langsmith_enabled` and raises a clear, actionable
  error if it's off, so CI without a key simply skips it.

Enable with::

    pip install langsmith
    export LANGCHAIN_API_KEY=lsv2_...
    export LANGCHAIN_TRACING_V2=true          # optional: also trace the target runs
    export LANGCHAIN_PROJECT=messaging-agent  # optional

Then::

    python -m messaging_agent.evals.cli data/evals/golden_full.jsonl --langsmith \
        --dataset leasing-golden --experiment leasing-v2 --judge
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

from ..graph import app
from . import judge as judge_mod
from . import personalization, semantic


# --------------------------------------------------------------------------- gating

def langsmith_available() -> bool:
    """True if the ``langsmith`` SDK is importable."""
    try:
        import langsmith  # noqa: F401
        return True
    except Exception:
        return False


def langsmith_enabled() -> bool:
    """True only if the SDK is installed *and* an API key is configured."""
    return langsmith_available() and bool(
        os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    )


def _require_enabled() -> None:
    if not langsmith_available():
        raise RuntimeError(
            "langsmith is not installed. Install it with: pip install langsmith"
        )
    if not langsmith_enabled():
        raise RuntimeError(
            "LangSmith is not configured. Set LANGCHAIN_API_KEY (and optionally "
            "LANGCHAIN_TRACING_V2=true, LANGCHAIN_PROJECT) to enable it."
        )


def status() -> dict[str, Any]:
    return {
        "sdk_installed": langsmith_available(),
        "enabled": langsmith_enabled(),
        "project": os.getenv("LANGCHAIN_PROJECT", "default"),
        "has_api_key": bool(os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")),
    }


# ------------------------------------------------------------------- shared helpers

def _produced(output: dict[str, Any]) -> dict[str, Any]:
    """Project a final_output dict down to the shape the scorers expect."""
    output = output or {}
    return {
        "next_message": output.get("next_message"),
        "next_action": output.get("next_action"),
        "should_send": output.get("should_send"),
    }


# ------------------------------------------------------- pure, directly-testable scorers
# Each returns the LangSmith feedback shape: {"key", "score", "comment"?}. They take plain
# dicts so they can be unit-tested without any LangSmith objects.

def score_personalization(output: dict, record: dict) -> dict[str, Any]:
    res = personalization.score(_produced(output), record or {})
    return {"key": "personalization", "score": res.get("overall", 0.0)}


def score_semantic(output: dict, expected: dict | None) -> dict[str, Any] | None:
    if not expected:
        return None
    res = semantic.score(_produced(output), expected)
    return {"key": "semantic_match", "score": res.get("overall", 0.0),
            "comment": res.get("method")}


def score_knowledge_recall(output: dict, expected_ids: list[str] | None) -> dict[str, Any] | None:
    if not expected_ids:
        return None
    want = set(expected_ids)
    got = {s.get("id") for s in (output.get("knowledge") or [])}
    return {"key": "knowledge_recall", "score": round(len(want & got) / len(want), 3)}


def score_ac_no_critical(output: dict) -> dict[str, Any]:
    evaluation = output.get("evaluation") or {}
    has_critical = bool(evaluation.get("critical_fails"))
    return {"key": "ac_no_critical_fail", "score": 0.0 if has_critical else 1.0,
            "comment": json.dumps(evaluation.get("critical_fails")) if has_critical else None}


async def score_judge(output: dict, record: dict) -> dict[str, Any]:
    res = await judge_mod.judge(_produced(output), record)
    return {"key": "judge_quality", "score": res.get("score", res.get("overall", 0.0)),
            "comment": res.get("rationale")}


# --------------------------------------------------------------- LangSmith plumbing

def make_target() -> Callable:
    """Build the async target callable LangSmith invokes per example.

    Returns the agent's ``final_output`` so the evaluators can read message/knowledge/
    evaluation. When LangSmith tracing is on, each call is captured as a full trace.
    """
    async def target(inputs: dict) -> dict:
        record = inputs.get("record", {}) or {}
        dataset = inputs.get("dataset") or [record]
        init = {"record": record, "dataset": dataset,
                "task_id": record.get("task_id", "unknown"),
                "raw_line": json.dumps(record, ensure_ascii=False)}
        state = await app.ainvoke(init)
        return state.get("final_output", {"task_id": record.get("task_id"), "error": "no output"})

    return target


def _example_ref(example: Any) -> dict[str, Any]:
    """Pull reference fields from a LangSmith Example (outputs) with input fallback."""
    outputs = getattr(example, "outputs", None) or {}
    inputs = getattr(example, "inputs", None) or {}
    record = inputs.get("record", {}) or {}
    return {
        "record": record,
        "expected": outputs.get("expected") or record.get("expected"),
        "expected_knowledge_ids": outputs.get("expected_knowledge_ids")
        or record.get("expected_knowledge_ids"),
    }


def build_evaluators(use_judge: bool = False) -> list[Callable]:
    """Wrap the pure scorers as LangSmith evaluators (``(run, example) -> feedback``)."""
    def personalization_evaluator(run, example):
        ref = _example_ref(example)
        return score_personalization(run.outputs or {}, ref["record"])

    def semantic_evaluator(run, example):
        ref = _example_ref(example)
        return score_semantic(run.outputs or {}, ref["expected"]) or {
            "key": "semantic_match", "score": None}

    def knowledge_recall_evaluator(run, example):
        ref = _example_ref(example)
        return score_knowledge_recall(run.outputs or {}, ref["expected_knowledge_ids"]) or {
            "key": "knowledge_recall", "score": None}

    def ac_evaluator(run, example):
        return score_ac_no_critical(run.outputs or {})

    evaluators: list[Callable] = [
        personalization_evaluator, semantic_evaluator,
        knowledge_recall_evaluator, ac_evaluator,
    ]

    if use_judge:
        async def judge_evaluator(run, example):
            ref = _example_ref(example)
            return await score_judge(run.outputs or {}, ref["record"])

        evaluators.append(judge_evaluator)

    return evaluators


def push_dataset(name: str, dataset: list[dict], *, description: str | None = None) -> Any:
    """Create (or reuse) a LangSmith dataset from in-memory records.

    Idempotent on name: if a dataset with ``name`` exists it's reused and any missing
    examples are appended (matched by ``task_id``).
    """
    _require_enabled()
    from langsmith import Client

    client = Client()
    try:
        ds = client.read_dataset(dataset_name=name)
    except Exception:
        ds = client.create_dataset(
            dataset_name=name,
            description=description or "messaging-agent eval dataset",
        )

    existing_ids: set[str] = set()
    try:
        for ex in client.list_examples(dataset_id=ds.id):
            tid = (ex.inputs or {}).get("record", {}).get("task_id")
            if tid:
                existing_ids.add(tid)
    except Exception:
        pass

    to_add = [r for r in dataset if r.get("task_id") not in existing_ids]
    for rec in to_add:
        client.create_example(
            inputs={"record": rec, "dataset": dataset},
            outputs={
                "expected": rec.get("expected"),
                "expected_knowledge_ids": rec.get("expected_knowledge_ids"),
                "thresholds": rec.get("thresholds"),
            },
            dataset_id=ds.id,
        )
    return ds


async def run_experiment(dataset_name: str, *, use_judge: bool = False,
                         experiment_prefix: str = "messaging-agent",
                         max_concurrency: int = 4) -> Any:
    """Run the agent over a hosted LangSmith dataset and score it with our evaluators."""
    _require_enabled()
    from langsmith import aevaluate

    return await aevaluate(
        make_target(),
        data=dataset_name,
        evaluators=build_evaluators(use_judge=use_judge),
        experiment_prefix=experiment_prefix,
        max_concurrency=max_concurrency,
    )


def run_file_on_langsmith(path: str, *, dataset_name: str | None = None,
                          experiment_prefix: str | None = None,
                          use_judge: bool = False) -> dict[str, Any]:
    """Push a JSONL file as a dataset, then run a scored experiment over it.

    Returns a small summary dict with the dataset name and experiment URL when available.
    """
    _require_enabled()
    import asyncio

    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    name = dataset_name or f"messaging-agent:{Path(path).stem}"
    prefix = experiment_prefix or Path(path).stem
    push_dataset(name, records)
    results = asyncio.run(
        run_experiment(name, use_judge=use_judge, experiment_prefix=prefix)
    )

    url = None
    for attr in ("experiment_url", "url"):
        url = getattr(results, attr, None)
        if url:
            break
    return {"dataset": name, "experiment_prefix": prefix,
            "records": len(records), "experiment_url": url}
