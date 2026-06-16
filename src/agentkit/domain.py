"""Pluggable Domain abstraction — the reusable-component seam.

The LangGraph engine (graph/routing/state, the node functions, the LLM client, the
``prod`` scaling layer, PII tokenization and the injection/jailbreak safety scan) is
**domain-agnostic**. Everything that is specific to *what* the agent is deciding about —
the system prompt, which record fields become "facts" for the model, the channel/consent
model, the output schema + compliance repairs, the few-shot ranking signal, and the
acceptance criteria — lives behind this :class:`Domain` interface.

To reuse this engine for a new use case you implement a :class:`Domain` subclass and
register it (``register_domain(MyDomain())``). Select it per-record (``record["domain"]``),
per-process (``AGENT_DOMAIN`` env var), or programmatically (:func:`set_default_domain`).
No change to the core graph or nodes is required.

The shipped, reference implementation is ``domains.leasing.LeasingDomain`` (the original
multifamily-housing leasing agent), which is also the default.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DecisionContext:
    """Everything the ``context`` node needs the domain to compute for the LLM call.

    Returned by :meth:`Domain.build_decision_context`. The core node owns the generic
    envelope (PII tokenization, tenant resolution, lineage assembly) and merges these
    domain-specific pieces into the graph state.
    """

    facts: dict[str, Any]
    guardrails: dict[str, Any]
    system_prompt: str
    user_prompt: str
    example_ids: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    """Optional extra state keys the domain wants persisted (e.g. ``asset_class``)."""


class Domain(ABC):
    """Extension surface that turns the generic engine into a concrete agent.

    Subclass this and implement the abstract methods. Concrete defaults are provided for
    the channel/consent helpers (driven by :meth:`channel_consent_map`) so most domains
    only need to override the prompt/facts/normalize/criteria methods.
    """

    #: Unique key used for registration and selection.
    name: str = "base"

    # --- Consent / channels -------------------------------------------------------
    @abstractmethod
    def channel_consent_map(self) -> dict[str, str]:
        """Map an outbound channel name -> the boolean consent flag in ``record['consent']``."""

    def consented_channels(self, record: dict[str, Any]) -> list[str]:
        """Channels the recipient has explicitly opted into (consent gate input)."""
        consent = record.get("consent", {}) or {}
        return [ch for ch, flag in self.channel_consent_map().items() if consent.get(flag) is True]

    def select_channel(self, record: dict[str, Any]) -> str | None:
        """Highest-priority consented channel (``channel_preferences`` order)."""
        allowed = set(self.consented_channels(record))
        if not allowed:
            return None
        prefs = record.get("channel_preferences") or []
        if isinstance(prefs, list):
            for ch in prefs:
                if ch in allowed:
                    return ch
        for ch in self.channel_consent_map():
            if ch in allowed:
                return ch
        return None

    # --- Intake -------------------------------------------------------------------
    def validate(self, record: dict[str, Any]) -> list[str]:
        """Return non-fatal validation warnings for a record (override per domain)."""
        return []

    # --- Context / prompting ------------------------------------------------------
    @abstractmethod
    def build_decision_context(
        self,
        *,
        record: dict[str, Any],
        sanitized: dict[str, Any],
        tenant: dict[str, Any],
        dataset: list[dict[str, Any]],
    ) -> DecisionContext:
        """Compute facts, guardrails, and the system/user prompt for one record."""

    # --- Output validation / compliance repair ------------------------------------
    @abstractmethod
    def normalize(self, output: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        """Validate the LLM output and minimally repair it for hard compliance.

        Contract (consumed by the ``parse_output`` node): return the repaired output
        dict. Use the private convention keys ``_error`` (str -> trigger a retry),
        ``_warnings`` (list[str] audit trail) and any domain-specific observability
        flags. A legitimate no-send is ``{"should_send": False, ...}``.
        """

    # --- Evaluation ---------------------------------------------------------------
    @abstractmethod
    def evaluate_all(
        self, output: dict[str, Any], record: dict[str, Any], sanitized: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Score the produced output against this domain's acceptance criteria."""

    # --- Knowledge / RAG (optional) -----------------------------------------------
    def knowledge_base(self):  # -> KnowledgeBase | None
        """Return the :class:`~agentkit.knowledge.KnowledgeBase` to retrieve from.

        Default ``None`` (no RAG). Override to ground the LLM in retrieved documents.
        """
        return None

    def knowledge_query(self, record: dict[str, Any], facts: dict[str, Any]) -> str | None:
        """Build the retrieval query for this record (default: no retrieval)."""
        return None

    def knowledge_filter(self, record: dict[str, Any]) -> dict[str, Any] | None:
        """Optional exact-match metadata filter for retrieval (e.g. tenant isolation)."""
        return None

    def knowledge_top_k(self) -> int:
        return 4

    def format_knowledge(self, snippets) -> str:
        """Render retrieved snippets into a prompt block (override to customize)."""
        from .knowledge import render_snippets
        return render_snippets(snippets)

    # --- Telemetry / observability (optional) -------------------------------------
    def telemetry_features(self, record: dict[str, Any]) -> dict[str, Any]:
        """Non-PII feature vector paired with outcomes in closed-loop telemetry.

        Default is a generic, schema-light extraction; override for richer features.
        """
        inp = record.get("input", {}) or {}
        return {
            "domain": self.name,
            "channel_preferences": record.get("channel_preferences"),
            "consent": record.get("consent"),
            "lifecycle_stage": record.get("lifecycle_stage"),
            "tags": record.get("tags"),
            "input_keys": sorted(k for k in inp.keys() if k != "profile"),
        }

    # --- Eval (optional, domain-specific metrics + judge) -------------------------
    def eval_metrics(
        self, produced: dict[str, Any], record: dict[str, Any], sanitized: dict[str, Any]
    ) -> dict[str, Any]:
        """Extra per-record metrics merged into the eval harness item (default none)."""
        return {}

    def judge_system_prompt(self) -> str | None:
        """Override the LLM-as-judge system prompt for this domain (default: built-in)."""
        return None


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

_REGISTRY: dict[str, Domain] = {}
_DEFAULT_NAME: str | None = None


def register_domain(domain: Domain, *, default: bool = False) -> Domain:
    """Register a domain instance under ``domain.name``. Returns it for convenience."""
    global _DEFAULT_NAME
    _REGISTRY[domain.name] = domain
    if default or _DEFAULT_NAME is None:
        _DEFAULT_NAME = domain.name
    return domain


def set_default_domain(name: str) -> None:
    if name not in _REGISTRY:
        _ensure_builtins()
    if name not in _REGISTRY:
        raise KeyError(f"unknown domain {name!r}; registered: {sorted(_REGISTRY)}")
    global _DEFAULT_NAME
    _DEFAULT_NAME = name


def available_domains() -> list[str]:
    _ensure_builtins()
    return sorted(_REGISTRY)


def _ensure_builtins() -> None:
    """Lazily import + register the bundled domains (avoids an import cycle)."""
    if "leasing" not in _REGISTRY:
        from .domains import leasing  # noqa: F401  (registers on import)
    if "support" not in _REGISTRY:
        from .domains import support  # noqa: F401  (registers on import)


def get_domain(state: dict[str, Any] | None = None) -> Domain:
    """Resolve the active domain.

    Precedence: an explicit ``domain`` key on the record/state, then the ``AGENT_DOMAIN``
    environment variable, then the process default (first-registered, i.e. ``leasing``).
    """
    _ensure_builtins()
    name: str | None = None
    if state:
        for key in ("validated_record", "sanitized_record", "record"):
            rec = state.get(key)
            if isinstance(rec, dict) and isinstance(rec.get("domain"), str):
                name = rec["domain"]
                break
        if name is None and isinstance(state.get("domain"), str):
            name = state["domain"]
    if name is None:
        name = os.getenv("AGENT_DOMAIN") or _DEFAULT_NAME
    domain = _REGISTRY.get(name) if name else None
    if domain is None:
        domain = _REGISTRY.get(_DEFAULT_NAME) if _DEFAULT_NAME else None
    if domain is None:
        raise RuntimeError("no domain registered")
    return domain
