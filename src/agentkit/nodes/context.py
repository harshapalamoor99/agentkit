"""Context node (domain-agnostic orchestration).

Performs the generic, cross-domain governance — sanitize convergence, PII tokenization
(AC-8), tenant resolution (AC-7) and decision-lineage assembly (AC-12) — then delegates
the domain-specific work (fact extraction, guardrails, few-shot selection and prompt
construction) to the active :class:`~agentkit.domain.Domain`.

This is the convergence point for both the clean and sanitized paths, so it is where
zero-trust data governance is applied before anything reaches the model.
"""
from __future__ import annotations

from .. import pii, safety_rules, tenants
from ..domain import get_domain
from ..state import MessagingAgentState


def context_builder(state: MessagingAgentState) -> MessagingAgentState:
    record = state.get("validated_record", {})
    sanitized = state.get("sanitized_record") or safety_rules.sanitize_record(record)

    # --- AC-8: tokenize raw PII out of the record BEFORE the LLM sees it. ---
    sanitized, pii_notes = pii.tokenize_record(sanitized)

    # --- AC-7: resolve the requesting tenant's brand (and only theirs). ---
    tenant = tenants.resolve_tenant(record)

    # --- Domain-specific facts, guardrails, examples and prompt. ---
    domain = get_domain(state)
    ctx = domain.build_decision_context(
        record=record, sanitized=sanitized, tenant=tenant, dataset=state.get("dataset", []))

    user_prompt = ctx.user_prompt

    # --- RAG: retrieve grounding knowledge (only if the domain exposes a base). ---
    knowledge: list[dict] = []
    kb = domain.knowledge_base()
    query = domain.knowledge_query(record, ctx.facts) if kb else None
    if kb and query:
        try:
            snippets = kb.retrieve(query, k=domain.knowledge_top_k(),
                                   where=domain.knowledge_filter(record))
        except Exception:
            snippets = []
        if snippets:
            block = domain.format_knowledge(snippets)
            if block:
                user_prompt = user_prompt + "\n\n" + block
            knowledge = [s.as_dict() for s in snippets]
            try:
                from .. import telemetry
                telemetry.store.emit_event("knowledge_retrieved", {
                    "task_id": record.get("task_id"),
                    "domain": domain.name,
                    "kb": getattr(kb, "name", None),
                    "query": query,
                    "snippet_ids": [s["id"] for s in knowledge],
                })
            except Exception:
                pass

    enriched = {
        "facts": ctx.facts,
        "system_prompt": ctx.system_prompt,
        "user_prompt": user_prompt,
    }

    # --- AC-12: decision-lineage snapshot (domain lineage + generic PII audit). ---
    lineage = dict(ctx.extras.get("lineage", {}))
    lineage.setdefault("few_shot_example_ids", ctx.example_ids)
    lineage["pii_tokenization"] = pii_notes
    if knowledge:
        lineage["knowledge"] = [{"id": s["id"], "score": s["score"], "source": s["source"]}
                                for s in knowledge]

    return {
        **state,
        "sanitized_record": sanitized,
        "pii_notes": pii_notes,
        "tenant": tenant,
        "asset_class": ctx.extras.get("asset_class"),
        "constraints": ctx.guardrails,
        "enriched_context": enriched,
        "knowledge": knowledge,
        "lineage": lineage,
    }
