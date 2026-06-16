# Authoring a Domain — reuse the engine for your own use case

This project is a **reusable, domain-pluggable agent engine**. The LangGraph pipeline
(`intake → safety → sanitize → context → llm → parse_output → evaluate → emit/abort`),
the provider-agnostic LLM client, the safety/PII layer and the entire `prod/` scaling
stack are **domain-agnostic**. Everything specific to *what* the agent decides about
lives behind one interface: [`messaging_agent.domain.Domain`](../src/messaging_agent/domain.py).

To target a new use case you write a `Domain` subclass and register it. **No change to
the core graph, nodes, LLM client or prod layer is required.**

## The decision boundary stays the same

The engine keeps doing the heavy lifting — orchestration, retry-with-budget, safe-abort,
injection/jailbreak scanning, PII tokenization, decision lineage, telemetry, concurrency.
Your `Domain` supplies the use-case knowledge:

| You implement | Replaces the leasing-specific… |
|---------------|-------------------------------|
| `channel_consent_map()` | sms/email/voice + TCPA consent flags |
| `validate(record)` | required-field warnings |
| `build_decision_context(...)` | facts, guardrails, system/user prompt, few-shot |
| `normalize(output, state)` | compliance validation + minimal repair |
| `evaluate_all(output, record, sanitized)` | the AC-01..23 acceptance criteria |

## Minimal example

```python
from messaging_agent.domain import Domain, DecisionContext, register_domain

class SalesDomain(Domain):
    name = "sales"

    def channel_consent_map(self):
        return {"email": "email_opt_in", "linkedin": "linkedin_opt_in"}

    def validate(self, record):
        return [] if record.get("task_id") else ["missing_task_id"]

    def build_decision_context(self, *, record, sanitized, tenant, dataset):
        facts = {
            "allowed_channels": self.consented_channels(record),
            "first_name": (sanitized.get("input", {}).get("profile", {}) or {}).get("first_name"),
            "primary_cta": "book_demo",
        }
        guardrails = {"allowed_channels": facts["allowed_channels"]}
        system = "You are a B2B sales outreach agent. Output STRICT JSON ..."
        user = f"Lead facts: {facts}\nReturn STRICT JSON now."
        return DecisionContext(facts=facts, guardrails=guardrails,
                               system_prompt=system, user_prompt=user)

    def normalize(self, output, state):
        # validate the LLM's choice; minimally repair for compliance; return the dict.
        # use "_error" to trigger a retry, "_warnings" for the audit trail.
        ...

    def evaluate_all(self, output, record, sanitized):
        return [{"id": "S-01", "severity": "critical", "title": "Channel consented",
                 "pass": True, "detail": ""}]

register_domain(SalesDomain())
```

Put your subclass in `src/messaging_agent/domains/<name>.py` and import it from
`domain._ensure_builtins()` (or register it from your own bootstrap code).

## Selecting a domain

Resolution precedence (see `domain.get_domain`):

1. Per record: `{"domain": "sales", ...}` on the input record.
2. Per process: `AGENT_DOMAIN=sales`.
3. Programmatic: `messaging_agent.set_default_domain("sales")`.
4. Default: the first-registered domain (`leasing`).

```bash
AGENT_DOMAIN=sales PYTHONPATH=src python -m messaging_agent.cli leads.jsonl
```

## Contracts to respect

- **`normalize` return value** (consumed by the `parse_output` node):
  - a normal send → `{"should_send": True, "next_message": {...}, "next_action": {...}}`
  - a legitimate no-send → `{"should_send": False, "next_message": None, "reasoning": "..."}`
  - convention keys: `_error` (str → triggers a corrective retry), `_warnings`
    (`list[str]` audit trail; entries containing `_repair`/`_quarantine` are surfaced as
    compliance repairs), plus any domain observability flags you pop in the node.
- **`evaluate_all` results**: a list of dicts with at least `id`, `severity`
  (`critical`/`high`/`medium`), `title`, `pass` (bool), `detail`. The engine fails the
  record if any `critical` criterion does not pass.
- The output envelope shipped by the `emit` node is `next_message` / `next_action` /
  `reasoning`; keep your schema compatible with it (or extend `emit.py` if you need a
  different shape).

## Reference implementations

- [`domains/leasing.py`](../src/messaging_agent/domains/leasing.py) — the full original
  housing agent (default domain).
- [`domains/support.py`](../src/messaging_agent/domains/support.py) — a compact,
  unrelated example (customer-support follow-ups) exercised end-to-end by
  [`tests/test_support_domain.py`](../tests/test_support_domain.py).

## Knowledge / RAG layer (optional)

A domain can ground the LLM in retrieved documents. Implement two hooks and the
`context` node will retrieve top-k snippets **before** the LLM call, inject them into the
prompt, and record them in the decision lineage + telemetry (`knowledge_retrieved`):

```python
from messaging_agent import InMemoryKnowledgeBase, KnowledgeDoc

_KB = InMemoryKnowledgeBase([
    KnowledgeDoc(id="faq-pets", text="Pets: cats and dogs allowed with a deposit.",
                 metadata={"tenant_id": "t1"}),
])

class MyDomain(Domain):
    ...
    def knowledge_base(self):          # which store to retrieve from
        return _KB
    def knowledge_query(self, record, facts):   # the query for THIS record
        return record["input"].get("question")
    def knowledge_filter(self, record):          # optional isolation (e.g. per tenant)
        return {"tenant_id": record.get("tenant_id")}
```

- `InMemoryKnowledgeBase` is **pure-python TF-IDF** — zero dependencies, deterministic,
  good for dev/CI/demos. Load from JSONL with `InMemoryKnowledgeBase.from_jsonl(path)`.
- Production backends (vector DBs, managed retrieval) plug in behind the same
  `KnowledgeBase.retrieve(query, k, where)` interface; register with
  `register_knowledge_base(kb)` and return it from `knowledge_base()`.
- The bundled `leasing` domain gains optional RAG automatically when
  `LEASING_KNOWLEDGE_PATH=/path/to/kb.jsonl` is set (otherwise no retrieval).
- Retrieval failures degrade gracefully to "no knowledge" — they never abort a record.

The eval harness scores **retrieval recall@k** when a record declares
`"expected_knowledge_ids": [...]` (reported as `knowledge.mean_recall`).

## Domain-aware evals & telemetry

- `eval_metrics(produced, record, sanitized)` — extra per-record metrics merged into the
  harness item under `domain_metrics` (default none).
- `judge_system_prompt()` — override the LLM-as-judge rubric for your domain (default:
  the built-in leasing rubric).
- `telemetry_features(record)` — the non-PII feature vector paired with outcomes in
  closed-loop telemetry. A generic default is provided; override for richer features.

## Exposing as a multi-agent system

Each registered domain is a specialized agent over the one shared graph. Use
[`multiagent.py`](../src/messaging_agent/multiagent.py):

```python
from messaging_agent import AgentService, AgentRouter

# Run one domain agent directly:
out = await AgentService("support").run(record)

# Or route across domains (by record["domain"], a classifier, or a default):
router = AgentRouter(classifier=my_classifier, default_domain="leasing")
out = await router.dispatch(record)   # out["routed_to"] shows the chosen domain

# Surface an agent as a tool for an external orchestrator (function-calling / supervisor):
tool = AgentService("support").as_tool()   # {name, description, input_schema, callable}
```

HTTP: `GET /api/agents` lists the domain agents; `POST /api/dispatch` routes a record.

## Tool-calling adapters (OpenAI / Anthropic)

Expose any domain agent (or the router) as a function-calling tool with
[`tooling.py`](../src/messaging_agent/tooling.py):

```python
from messaging_agent import AgentService, to_openai_tool, run_tool_call

svc = AgentService("leasing")
tools = [to_openai_tool(svc.as_tool())]          # or to_anthropic_tool(...)

resp = openai_client.chat.completions.create(model=..., messages=..., tools=tools)
for call in resp.choices[0].message.tool_calls:
    result_json = await run_tool_call(svc, call.function.name, call.function.arguments)
    # append {"role": "tool", "tool_call_id": call.id, "content": result_json}
```

- `run_tool_call` accepts **either** a JSON-string (OpenAI) **or** a dict (Anthropic) of
  arguments and returns a JSON string ready to hand back to the model.
- `ToolRegistry(["leasing", "support"])` advertises every agent as a separate tool
  (`tools_openai()` / `tools_anthropic()`) and `dispatch(tool_name, args)` runs the right
  one — so a model can pick among domains.
- `router_as_tool(AgentRouter(...))` exposes **one** auto-routing tool instead.

## Bundled data

- `data/knowledge/leasing_kb.jsonl` — tenant-tagged FAQ knowledge base auto-loaded by the
  leasing domain (override with `LEASING_KNOWLEDGE_PATH`, disable with an empty value).
- `data/evals/rag_knowledge.jsonl` — RAG eval records carrying `expected_knowledge_ids`;
  run `python -m messaging_agent.evals.cli data/evals/rag_knowledge.jsonl` to score
  retrieval `knowledge.mean_recall`.

## Observability: tracing + token/cost accounting

**Token/cost** is always on and needs no setup. Every LLM call's token usage is captured
(all providers) and folded — across retries — into the decision lineage. It surfaces on
the emitted output:

```python
out["cost"]            # {model, prompt_tokens, completion_tokens, total_tokens,
                       #  input_cost_usd, output_cost_usd, cost_usd, calls, priced}
out["lineage"]["cost"] # same running total
```

Prices live in [`cost.py`](../src/messaging_agent/cost.py) as USD per 1M tokens. Override
without code changes:

```bash
export LLM_PRICE_TABLE='{"my-model": {"input": 0.5, "output": 1.5}}'
export LLM_PRICE_TABLE_PATH=/etc/messaging_agent/prices.json   # JSON file, merged in
```

Unknown models cost `0.0` and are flagged `priced=False` rather than raising.

**Distributed tracing** is opt-in and degrades to a zero-overhead no-op unless both the
SDK is installed and tracing is enabled — so tests and light deployments are unaffected.
Each graph node (`intake → safety → … → emit`) runs in a span carrying
`agent.domain`, `agent.tenant_id`, `agent.retry_count`, `agent.abort_reason`,
`agent.total_tokens`, `agent.cost_usd`, etc.

```bash
pip install 'messaging-agent[otel]'
export MESSAGING_AGENT_TRACING=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317   # else set MESSAGING_AGENT_TRACE_CONSOLE=true
```

```python
from messaging_agent import tracing_status, span
tracing_status()                       # {tracing_enabled, tracer_active, otlp_endpoint, ...}
with span("my.work", {"k": "v"}):      # wrap your own code in custom spans
    ...
```

This is independent of the LangSmith hook in `prod/tracing.py` (enabled via
`LANGCHAIN_TRACING_V2`); both can run at once.

## Hosted evaluation with LangSmith

The local harness (`evals/harness.py`) is the source of truth for scoring. The LangSmith
adapter ([`evals/langsmith_eval.py`](../src/messaging_agent/evals/langsmith_eval.py))
reuses the **same** scorers (`personalization`, `semantic`, `judge`, RAG
`knowledge_recall`, and the AC critical-fail gate) but runs them as a hosted *experiment*,
giving you trace-linked scores and version-to-version comparison in the LangSmith UI.

It's a no-op without credentials — importing it never requires the SDK, and every network
call checks `langsmith_enabled()` first.

```bash
pip install 'messaging-agent[langsmith]'
export LANGCHAIN_API_KEY=lsv2_...
export LANGCHAIN_TRACING_V2=true            # optional: also trace each target run
export LANGCHAIN_PROJECT=messaging-agent    # optional

# push the JSONL as a dataset + run a scored experiment in one shot:
python -m messaging_agent.evals.cli data/evals/golden_full.jsonl --langsmith \
    --dataset leasing-golden --experiment leasing-v2 --judge
```

Programmatic use:

```python
from messaging_agent.evals import langsmith_eval as lse

lse.push_dataset("leasing-golden", records)          # idempotent by task_id
await lse.run_experiment("leasing-golden", use_judge=True, experiment_prefix="leasing-v2")
```

The pure `score_*` functions (e.g. `score_knowledge_recall(output, expected_ids)`) are
plain dict-in/dict-out and unit-testable without any LangSmith objects;
`build_evaluators()` wraps them into the `(run, example)` evaluator signature LangSmith
calls. This keeps **one** scoring implementation behind both the local and hosted runners.



