"""Batch evaluation harness.

Runs the agent over a JSONL dataset and produces a single aggregate report covering:

* Acceptance criteria (AC-01..22) pass rate + critical fails.
* Semantic match vs. the `expected` ground truth (per record + mean).
* The declared per-record thresholds:
    - personalization_score_min
    - p95_latency_ms          (measured across the batch)
    - safety_violations_max   (violations detected in the *produced* output)
    - reply_classification_f1_min (capability eval via the reply classifier)
* Optional LLM-as-judge quality score.

Exits non-zero if any critical AC fails or any hard threshold is breached.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from .. import safety_rules
from ..graph import app
from . import judge as judge_mod
from . import personalization, reply_classifier, semantic


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _output_safety_violations(produced: dict) -> list[str]:
    msg = produced.get("next_message") or {}
    text = (msg.get("subject") or "") + " " + (msg.get("body") or "")
    viol = [f"pii_{c}" for c in safety_rules.find_pii(text)]
    if safety_rules.has_toxic(text):
        viol.append("toxic_or_fair_housing")
    if safety_rules._INJECTION_RE.search(text) or safety_rules._CODE_RE.search(text):
        viol.append("injection_leak")
    return viol


async def _run_one(record: dict, dataset: list[dict], use_judge: bool) -> dict[str, Any]:
    init = {"record": record, "dataset": dataset,
            "task_id": record.get("task_id", "unknown"),
            "raw_line": json.dumps(record, ensure_ascii=False)}
    t0 = time.perf_counter()
    state = await app.ainvoke(init)
    wall_ms = (time.perf_counter() - t0) * 1000
    out = state.get("final_output", {})

    produced = {
        "next_message": out.get("next_message"),
        "next_action": out.get("next_action"),
        "should_send": out.get("should_send"),
    }
    sanitized = state.get("sanitized_record", record)
    thresholds = record.get("thresholds", {}) or {}
    expected = record.get("expected")

    item: dict[str, Any] = {
        "task_id": out.get("task_id"),
        "should_send": out.get("should_send"),
        "abort_reason": out.get("abort_reason"),
        "ac": out.get("evaluation", {}),
        "wall_ms": round(wall_ms, 1),
        "pipeline_latency_ms": out.get("latency_ms", 0),
        "output_safety_violations": _output_safety_violations(produced),
        "personalization": personalization.score(produced, sanitized),
        "knowledge": out.get("knowledge", []),
        "thresholds_declared": {
            k: thresholds.get(k) for k in
            ("personalization_score_min", "p95_latency_ms",
             "reply_classification_f1_min", "safety_violations_max")
        },
    }

    # RAG retrieval eval: when a record declares the knowledge it SHOULD have surfaced
    # (`expected_knowledge_ids`), score recall@k of the retrieved snippet ids.
    expected_k = record.get("expected_knowledge_ids")
    if expected_k:
        retrieved_ids = [s.get("id") for s in item["knowledge"]]
        hit = len(set(expected_k) & set(retrieved_ids))
        item["knowledge_recall"] = round(hit / len(set(expected_k)), 3) if expected_k else None

    # Domain-specific extra metrics (no-op for domains that don't define any).
    try:
        from ..domain import get_domain as _get_domain
        extra = _get_domain({"validated_record": record}).eval_metrics(produced, record, sanitized)
        if extra:
            item["domain_metrics"] = extra
    except Exception:
        pass
    if expected:
        from ..nodes import llm as _llmnode
        item["semantic"] = await semantic.score_async(produced, expected, _llmnode._client)
    if use_judge:
        item["judge"] = await judge_mod.judge(produced, record)
    return item


async def run(dataset: list[dict], use_judge: bool = False) -> dict[str, Any]:
    from ..nodes import llm as _llmnode
    if _llmnode._client.available:
        await _llmnode._client.warmup()  # prime gateway so the first record isn't cold
    items = [await _run_one(rec, dataset, use_judge) for rec in dataset]

    latencies = [i["wall_ms"] for i in items]
    p95 = _percentile(latencies, 0.95)

    ac_pass = sum(i["ac"].get("passed", 0) for i in items)
    ac_total = sum(i["ac"].get("total", 0) for i in items)
    critical_fail_records = [i["task_id"] for i in items if i["ac"].get("critical_fails")]

    sem_scores = [i["semantic"]["overall"] for i in items if "semantic" in i]
    sem_methods = sorted({i["semantic"].get("method") for i in items if "semantic" in i})
    pers_scores = [i["personalization"]["overall"] for i in items]
    judge_scores = [i["judge"]["score"] for i in items if "judge" in i]
    faith_scores = [i["judge"]["faithfulness"] for i in items
                    if "judge" in i and "faithfulness" in i["judge"]]
    ctxp_scores = [i["judge"]["context_precision"] for i in items
                   if "judge" in i and "context_precision" in i["judge"]]

    reply = reply_classifier.f1()

    # RAG retrieval quality (only over records that declared expected knowledge).
    recall_scores = [i["knowledge_recall"] for i in items
                     if i.get("knowledge_recall") is not None]
    mean_knowledge_recall = (round(sum(recall_scores) / len(recall_scores), 3)
                             if recall_scores else None)
    records_with_knowledge = sum(1 for i in items if i.get("knowledge"))

    # G5: send-availability metrics. Retriable aborts (transient backend failures) are
    # dead-letter candidates — they would have sent had the backend been healthy.
    from ..prod.deadletter import is_retriable_abort as _retriable
    sent = sum(1 for i in items if i.get("should_send"))
    dlq_candidates = [i["task_id"] for i in items if _retriable(i.get("abort_reason"))]
    send_success_rate = round(sent / len(items), 3) if items else None

    # Threshold gating.
    breaches: list[str] = []
    for i in items:
        decl = i["thresholds_declared"]
        pmin = decl.get("personalization_score_min")
        if pmin is not None and i["personalization"]["overall"] < pmin:
            breaches.append(f"{i['task_id']}: personalization {i['personalization']['overall']} < {pmin}")
        smax = decl.get("safety_violations_max")
        if smax is not None and len(i["output_safety_violations"]) > smax:
            breaches.append(f"{i['task_id']}: safety_violations {i['output_safety_violations']} > {smax}")
        lmax = decl.get("p95_latency_ms")
        if lmax is not None and i["wall_ms"] > lmax:
            breaches.append(f"{i['task_id']}: latency {i['wall_ms']}ms > {lmax}ms")
        fmin = decl.get("reply_classification_f1_min")
        if fmin is not None and reply["macro_f1"] < fmin:
            breaches.append(f"{i['task_id']}: reply_f1 {reply['macro_f1']} < {fmin}")

    # AC-13: named LLM-judge gates (only when the judge layer is run).
    from .. import config as _cfg
    mean_faith = round(sum(faith_scores) / len(faith_scores), 3) if faith_scores else None
    mean_ctxp = round(sum(ctxp_scores) / len(ctxp_scores), 3) if ctxp_scores else None
    if mean_faith is not None and mean_faith < _cfg.JUDGE_FAITHFULNESS_MIN:
        breaches.append(f"judge faithfulness {mean_faith} < {_cfg.JUDGE_FAITHFULNESS_MIN}")
    if mean_ctxp is not None and mean_ctxp < _cfg.JUDGE_CONTEXT_PRECISION_MIN:
        breaches.append(f"judge context_precision {mean_ctxp} < {_cfg.JUDGE_CONTEXT_PRECISION_MIN}")

    # G5: optional send-success SLO gate (off unless SEND_SUCCESS_RATE_MIN is set).
    import os as _os
    _ssr_min = _os.getenv("SEND_SUCCESS_RATE_MIN")
    if _ssr_min is not None and send_success_rate is not None \
            and send_success_rate < float(_ssr_min):
        breaches.append(f"send_success_rate {send_success_rate} < {_ssr_min}")

    summary = {
        "records": len(items),
        "ac_score": f"{ac_pass}/{ac_total}",
        "critical_fail_records": critical_fail_records,
        "mean_semantic_match": round(sum(sem_scores) / len(sem_scores), 3) if sem_scores else None,
        "semantic_match_method": "+".join(m for m in sem_methods if m) or None,
        "mean_personalization": round(sum(pers_scores) / len(pers_scores), 3) if pers_scores else None,
        "mean_judge_quality": round(sum(judge_scores) / len(judge_scores), 3) if judge_scores else None,
        "mean_faithfulness": mean_faith,
        "mean_context_precision": mean_ctxp,
        "latency_ms": {"p50": round(_percentile(latencies, 0.5), 1),
                       "p95": round(p95, 1),
                       "max": round(max(latencies), 1) if latencies else 0.0},
        "reply_classifier": {"macro_f1": reply["macro_f1"], "accuracy": reply["accuracy"]},
        "knowledge": {"records_with_knowledge": records_with_knowledge,
                      "mean_recall": mean_knowledge_recall},
        "send_success_rate": send_success_rate,
        "sent": sent,
        "dead_letter_candidates": dlq_candidates,
        "threshold_breaches": breaches,
        "passed": not critical_fail_records and not breaches,
    }
    return {"summary": summary, "items": items}


def run_file(path: str, use_judge: bool = False) -> dict[str, Any]:
    dataset = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            dataset.append(json.loads(line))
    return asyncio.run(run(dataset, use_judge))
