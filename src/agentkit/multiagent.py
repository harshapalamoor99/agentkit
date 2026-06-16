"""Multi-agent surface — run the engine as a composable agent and route across domains.

Each registered :class:`~agentkit.domain.Domain` is effectively a *specialized
agent* (its own prompt, channels, compliance, criteria and optional knowledge base),
all sharing the one compiled LangGraph pipeline. This module exposes two things:

* :class:`AgentService` — a thin, reusable wrapper over the compiled graph for a single
  domain. It can be invoked directly (``await svc.run(record)``) or surfaced as a
  tool/sub-agent for an external orchestrator (LangGraph supervisor, A2A, function
  calling) via :meth:`AgentService.as_tool`.
* :class:`AgentRouter` — a supervisor that dispatches an incoming record to the right
  domain agent (by explicit ``record["domain"]``, a custom classifier, or a default),
  giving you a multi-agent system out of the box.

No new infra: this is pure orchestration over the existing graph, so it degrades and
scales exactly like the rest of the engine.
"""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

from .domain import available_domains, get_domain


def _default_app():
    from .graph import app as _agent_app
    return _agent_app


class AgentService:
    """A single-domain agent over the shared compiled graph.

    Parameters
    ----------
    domain_name:
        The domain this agent specializes in. ``None`` means "resolve per record"
        (uses the record's ``domain`` field / ``AGENT_DOMAIN`` / default).
    app:
        Override the compiled graph (mainly for tests).
    """

    def __init__(self, domain_name: str | None = None, *, app=None):
        self.domain_name = domain_name
        self._app = app or _default_app()

    @property
    def name(self) -> str:
        return f"agent:{self.domain_name or 'auto'}"

    @property
    def description(self) -> str:
        if self.domain_name:
            try:
                return (get_domain({"domain": self.domain_name}).__doc__ or "").strip().split("\n")[0] \
                    or f"{self.domain_name} agent"
            except Exception:
                return f"{self.domain_name} agent"
        return "Auto-routing messaging/decision agent"

    async def run(
        self, record: dict[str, Any], *, dataset: list[dict[str, Any]] | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Invoke the agent on one record and return its ``final_output``."""
        rec = dict(record)
        if self.domain_name and "domain" not in rec:
            rec["domain"] = self.domain_name
        init: dict[str, Any] = {
            "task_id": rec.get("task_id", "unknown"),
            "record": rec,
            "dataset": dataset if dataset is not None else [rec],
            "raw_line": json.dumps(rec, ensure_ascii=False),
        }
        state = await self._app.ainvoke(init, config=config) if config else await self._app.ainvoke(init)
        return state.get("final_output", state)

    async def run_batch(
        self, records: list[dict[str, Any]], *, dataset: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        ds = dataset if dataset is not None else records
        return [await self.run(r, dataset=ds) for r in records]

    def as_tool(self) -> dict[str, Any]:
        """Describe this agent as a callable tool for an external orchestrator.

        Returns a provider-neutral descriptor: ``name``, ``description``, a JSON
        ``input_schema`` and an async ``callable`` taking ``{"record": {...}}``. Adapt it
        to OpenAI function-calling, Anthropic tools, or a LangGraph supervisor node.
        """
        async def _invoke(args: dict[str, Any]) -> dict[str, Any]:
            return await self.run(args["record"], dataset=args.get("dataset"))

        return {
            "name": (self.domain_name or "agent") + "_agent",
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": {
                    "record": {"type": "object",
                               "description": "The input record (profile, consent, context...)."},
                    "dataset": {"type": "array", "items": {"type": "object"},
                                "description": "Optional few-shot pool of labelled records."},
                },
                "required": ["record"],
            },
            "callable": _invoke,
        }


# Type of a classifier: record -> domain name (or None to fall through to default).
Classifier = Callable[[dict[str, Any]], "str | None"]


class AgentRouter:
    """Supervisor that dispatches records to the appropriate domain agent.

    Resolution order: explicit ``record['domain']`` -> ``classifier(record)`` ->
    ``default_domain``. Each resolved domain gets a cached :class:`AgentService`.
    """

    def __init__(
        self, *, classifier: Classifier | None = None, default_domain: str | None = None,
        app=None,
    ):
        self._classifier = classifier
        self._default = default_domain
        self._app = app
        self._services: dict[str, AgentService] = {}

    def register(self, domain_name: str) -> AgentService:
        svc = AgentService(domain_name, app=self._app)
        self._services[domain_name] = svc
        return svc

    def service_for(self, domain_name: str) -> AgentService:
        if domain_name not in self._services:
            self.register(domain_name)
        return self._services[domain_name]

    def resolve(self, record: dict[str, Any]) -> str:
        name = record.get("domain")
        if not name and self._classifier is not None:
            try:
                name = self._classifier(record)
            except Exception:
                name = None
        if not name:
            name = self._default
        if not name:
            # Fall back to the engine's own default domain.
            name = get_domain({}).name
        return name

    async def dispatch(
        self, record: dict[str, Any], *, dataset: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        name = self.resolve(record)
        out = await self.service_for(name).run(record, dataset=dataset)
        if isinstance(out, dict):
            out.setdefault("routed_to", name)
        return out

    async def dispatch_batch(
        self, records: list[dict[str, Any]], *, dataset: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        ds = dataset if dataset is not None else records
        return [await self.dispatch(r, dataset=ds) for r in records]

    def agents(self) -> list[dict[str, Any]]:
        """List the available domain agents (for discovery / tool registration)."""
        return [AgentService(name).as_tool() for name in available_domains()]
