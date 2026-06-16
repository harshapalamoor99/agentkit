# Architecture & Design Decisions

This document explains **how** the system is built and, more importantly, **why** —
the trade-offs behind each decision. It is written ADR-style (Architecture Decision
Record): each decision states the context, the choice, the alternatives considered, and
the consequences.

> TL;DR of the thesis: **the LLM makes the business decision; deterministic code makes
> only the legal/compliance decision.** Everything else in the architecture follows from
> taking that separation seriously and making it testable, observable, and reusable.

---

## 1. System overview

The agent ingests a structured record (a prospect/customer + their consent, preferences,
and context) and decides **whether / how / when / what** to communicate, emitting a
validated, compliance-checked message or a safe no-send.

It is built as a **LangGraph state machine** of single-responsibility nodes:

```
intake ──┬─ ok ───────► safety ──┬─ clean ─────────► context ─► llm ─► parse_output
         ├─ no_consent ► abort   ├─ sanitized ─► sanitize ─► context     │
         └─ bad_record ► abort   └─ block ─────────► abort                │
                                                                          ▼
   emit ◄── evaluate ◄── (success) ──────────────────────────── parse_output
     │                                                              │
    END                                            (retry, until budget) ─► llm ─► abort
```

Each node is a pure `state -> state` function, independently unit-tested. LangGraph
supplies orchestration, conditional routing, and retry-with-state.

![LangGraph pipeline — node/edge flow with retry and safe-abort paths](diagrams/pipeline.png)

The reusability seam (ADR-005) and the opt-in capability layer (RAG, multi-agent, evals,
observability) are summarized in two further diagrams:

| Diagram | Shows |
|---------|-------|
| ![Reusability seam](diagrams/domain-plugin.png) | Domain-agnostic engine vs pluggable `Domain` plugins |
| ![Capability layer](diagrams/capabilities.png) | RAG, multi-agent, dual-runner evals, observability/cost |

Full diagram set (PNG + `.dot`/`.mmd` source): [`docs/diagrams/`](diagrams).

---

## 2. Core design decisions

### ADR-001 — The LLM is the decision engine; code is the guardrail

**Context.** The naive way to build outreach automation is a rules engine
(`if persona == "renewal" and channel_pref == "sms": ...`). It is deterministic and
testable but brittle, unscalable across domains, and not "AI."

**Decision.** The LLM decides whether to send, which channel, when, what to say, and the
next action — by reasoning over the record's own fields and few-shot `expected` examples.
Deterministic code is confined to **guardrails** that *validate and minimally repair* the
LLM's output: consent enforcement, business-hours/timezone validity, opt-out presence,
PII tokenization, and fair-housing/safety scanning.

**Alternatives considered.**
- *Pure rules engine* — rejected: doesn't generalize; every new behavior is code.
- *Pure LLM, no guardrails* — rejected: unsafe and non-compliant; an LLM cannot be the
  system-of-record for a legal obligation (consent, opt-out).

**Consequences.** The boundary is the central testable invariant: every acceptance
criterion checks either "did the LLM reason correctly?" (eval) or "did the guardrail hold
regardless of the LLM?" (compliance). It also makes the engine domain-agnostic — see
ADR-005.

### ADR-002 — Graph of small nodes over a monolithic agent loop

**Context.** Agent frameworks often encourage one big loop with tools. That is hard to
test and reason about, and failures are opaque.

**Decision.** Model the flow as an explicit LangGraph `StateGraph` with one node per
responsibility (intake, safety, sanitize, context, llm, parse, evaluate, emit, abort) and
**conditional edges** for routing.

**Consequences.** (+) Each node is a unit-testable function; the control flow is a diagram,
not a call stack. (+) Retry is just an edge back to `llm`. (−) More wiring up front; the
state schema (`state.py`) must be disciplined. Worth it for testability and observability.

### ADR-003 — LLM-only with a shared latency budget; no fabricated fallback

**Context.** A user-facing messaging system needs a hard p99 latency SLA (here, < 2s). LLM
calls have heavy tails. A common hack is to fabricate a deterministic message on timeout —
but that produces **un-reasoned copy** that violates ADR-001 and can be non-compliant.

**Decision.** Establish a single monotonic **deadline** on first LLM entry and share it
across retries (`llm_deadline` in state). Each attempt's timeout is
`min(LLM_TIMEOUT_S, remaining_budget)`. On no-provider, timeout, budget exhaustion, or
error, the node sets an `abort_reason` and the graph routes to a **safe no-send** — never
a fabricated message.

**Consequences.** p99 < 2s is bounded *by construction*, not by hope. The trade-off is an
explicit availability/quality choice: we'd rather not send than send junk. Retriable
aborts (transient backend failures) are surfaced as dead-letter candidates so they can be
retried out-of-band. See `nodes/llm.py` and `tests/test_latency.py`.

### ADR-004 — Circuit breaker, keyed per (provider, tenant)

**Context.** A degraded LLM backend can make every request wait out the full budget,
collapsing throughput. And in a multi-tenant deployment, one tenant's bad backend
shouldn't penalize others.

**Decision.** A circuit breaker (`circuit_breaker.py`) trips after repeated failures and
fast-aborts until a cooldown elapses. Breakers are keyed by **(provider, tenant)** so the
blast radius is isolated.

**Consequences.** Fast failure under degradation; tenant isolation. Adds a small amount of
global state, reset between tests via an autouse fixture.

### ADR-005 — The `Domain` plugin interface (the reusability seam)

**Context.** The original system was a multifamily-housing leasing agent. To make it a
reusable component, the use-case-specific logic had to be separated from the engine
without changing behavior.

**Decision.** Introduce a `Domain` ABC (`domain.py`) that owns everything use-case
specific: channel/consent mapping, decision-context construction, output normalization,
and the acceptance-criteria evaluators. The graph nodes are domain-agnostic and delegate
via `get_domain(state)`. Domains self-register; resolution precedence is record field →
state → `AGENT_DOMAIN` env → first-registered default.

**Alternatives considered.** *Config files / templates* — rejected: can't express the
normalization and evaluation logic a real domain needs. *Subclassing the graph* — rejected:
couples every domain to LangGraph internals.

**Consequences.** A new use case is a single `Domain` subclass + a test file; the engine,
scaling layer, RAG, evals, and observability are inherited for free. The bundled `leasing`
domain reproduces the original agent exactly (behavior-preserving refactor); `support` is a
second reference implementation. See `docs/AUTHORING_A_DOMAIN.md`.

### ADR-006 — RAG as an optional, domain-owned capability with graceful degradation

**Context.** Grounding the LLM in tenant-specific knowledge (FAQs, policies) improves
faithfulness, but a reusable engine can't mandate a vector DB.

**Decision.** A `KnowledgeBase` ABC with a pure-Python TF-IDF `InMemoryKnowledgeBase`
default (zero deps). A domain optionally exposes a knowledge base; the `context` node
retrieves top-k snippets *before* the LLM call, injects them as grounding, and records them
in lineage + telemetry. Metadata filters (`where`) enforce tenant isolation.

**Consequences.** RAG works out-of-the-box offline; production vector backends (FAISS,
pgvector, Chroma) plug in behind the same `retrieve(query, k, where)` interface. Retrieval
quality is measurable — see ADR-008.

### ADR-007 — Compliance as deterministic, auditable repair (not prompting)

**Context.** "Just tell the model to always include an opt-out" is not a compliance
control — it's a suggestion with a failure rate.

**Decision.** Safety/compliance is enforced in code after generation: PII tokenization,
injection/jailbreak/toxicity scanning, opt-out insertion, and an optional fair-housing
LLM veto that can only ever *add* safety (force no-send), never wrongly suppress on
infrastructure failure. Every repair is recorded in the decision lineage.

**Consequences.** Compliance is provable and auditable, independent of model behavior. The
LLM's pre-repair decision is snapshotted so you can see exactly what was changed and why.

### ADR-008 — Evaluation is a first-class, multi-layer artifact

**Context.** "It looks good" is not a quality bar. For an agent, quality spans correctness,
personalization, safety, latency, retrieval, and subjective copy quality.

**Decision.** A batch harness (`evals/harness.py`) scores every record on: acceptance-
criteria pass-rate + critical fails, semantic match vs. ground truth, personalization,
RAG recall@k (vs. `expected_knowledge_ids`), reply-classification F1, p95 latency, and an
optional **LLM-as-judge** with named gated dimensions (faithfulness ≥ 0.95, context
precision ≥ 0.90). The harness **exits non-zero** on any critical failure or threshold
breach, so it gates CI/CD.

**Consequences.** Quality is a number that can regress a build. The same scorers are reused
by the LangSmith adapter (`evals/langsmith_eval.py`) for hosted, trace-linked experiments —
one scoring implementation, two runners (offline JSONL + hosted).

### ADR-009 — Observability: vendor-neutral tracing + in-band cost accounting

**Context.** Production agents need to answer "why did this run do that, how slow, how
expensive?" — and you shouldn't be locked to one SaaS to answer it.

**Decision.** Two complementary layers: (1) **OpenTelemetry** spans per node
(`observability.py`), opt-in and **no-op by default** so tests and light deployments pay
nothing, exportable to any OTLP backend (Jaeger, Tempo, Phoenix, Datadog). (2) **Token →
USD cost accounting** (`cost.py`) folded across retries into the decision lineage and the
emitted output, with an overridable price table. A separate LangSmith hook
(`prod/tracing.py`) provides LLM-native prompt/eval tracing.

**Consequences.** You can correlate agent traces with the rest of your infra (OTel), debug
prompts and run experiments (LangSmith), and enforce per-record budgets (cost) — without
coupling the engine to any one vendor. See the comparison table in
`docs/AUTHORING_A_DOMAIN.md`.

### ADR-010 — Multi-agent exposure: services, a router, and tool adapters

**Context.** A reusable agent should compose into larger systems — both as a callee (tool)
and as a dispatcher across domains.

**Decision.** `AgentService` runs one domain or exposes it `.as_tool()`; `AgentRouter`
dispatches a record to the right domain (explicit field / classifier / default);
`tooling.py` converts a service into OpenAI/Anthropic function-calling schemas with a
`ToolRegistry` for multi-domain dispatch. HTTP surfaces: `GET /api/agents`,
`POST /api/dispatch`.

**Consequences.** The same engine is a standalone service, a function-calling tool for an
external orchestrator, or a router in a supervisor pattern — no new code paths.

---

## 3. Cross-cutting concerns

**State schema discipline.** All inter-node data flows through a single `TypedDict`
(`state.py`). This is the contract; nodes read/write named keys, which keeps the graph
debuggable and the lineage complete.

**Graceful degradation everywhere.** Every optional capability (RAG backend, Kafka
telemetry, Redis cache, OTel exporter, LangSmith, provider SDKs) has an in-memory / no-op
default. The system runs fully offline with zero external infra, and lights up more
capability as env vars/keys are added. This is what makes it both demo-able and
production-ready.

**Provider-agnostic LLM client.** `llm_client.py` auto-detects the provider from env
(LiteLLM gateway / Anthropic / Azure / OpenAI / Gemini), captures token usage uniformly,
and keeps a warm connection to stay under the latency budget.

---

## 4. Testing strategy

- **Hermetic by default.** `conftest.py` strips real provider keys and pins the latency
  budget so the suite is deterministic regardless of a developer's local `.env`. A mock
  LLM produces valid, compliant output so the full pipeline and every acceptance criterion
  run offline.
- **Layered coverage.** Unit tests per node + domain; integration tests over the full
  graph; latency-budget regression tests; RAG recall tests; tool-adapter shape +
  execution tests; cost/observability wiring tests.
- **Behavior-preservation tests.** The leasing domain refactor is covered by the original
  acceptance-criteria and RealPage enterprise tests, proving the generalization changed no
  behavior.
- **CI gate.** GitHub Actions runs the suite on Python 3.11–3.13 plus an offline eval
  smoke test, so every push proves the agent is runnable end-to-end without a key.

---

## 5. Repository map

| Path | Responsibility |
|---|---|
| `src/messaging_agent/graph.py` | LangGraph wiring (entry point: `app`) |
| `src/messaging_agent/state.py` | Shared state schema (the inter-node contract) |
| `src/messaging_agent/domain.py` | `Domain` plugin interface + registry (reusability seam) |
| `src/messaging_agent/domains/` | Bundled domains: `leasing` (default), `support` |
| `src/messaging_agent/nodes/` | One module per graph node |
| `src/messaging_agent/knowledge.py` | RAG layer (KnowledgeBase + TF-IDF retriever) |
| `src/messaging_agent/multiagent.py` | AgentService + AgentRouter |
| `src/messaging_agent/tooling.py` | OpenAI/Anthropic tool-calling adapters |
| `src/messaging_agent/observability.py` | OpenTelemetry node tracing (opt-in, no-op default) |
| `src/messaging_agent/cost.py` | Token → USD cost accounting |
| `src/messaging_agent/evals/` | Harness, judge, semantic/personalization/reply scorers, LangSmith adapter |
| `src/messaging_agent/prod/` | Runner, worker, idempotency, cache, audit, dead-letter, tracing |
| `src/messaging_agent/llm_client.py` | Provider-agnostic async client |

For how to plug in your own domain, see [`AUTHORING_A_DOMAIN.md`](AUTHORING_A_DOMAIN.md).
