"""messaging_agent package.

Loads a local .env (if present) on import so gateway/model configuration is picked up
automatically without manual exports. Real environment variables always take
precedence over .env values.

This package is a **reusable, domain-pluggable agent engine**: the LangGraph pipeline,
LLM client and ``prod`` scaling layer are domain-agnostic, while everything use-case
specific lives behind the :class:`~messaging_agent.domain.Domain` interface. The bundled
``leasing`` domain (default) reproduces the original housing agent; ``support`` is a
second example. See ``docs/AUTHORING_A_DOMAIN.md`` to plug in your own.
"""
from ._env import load_dotenv as _load_dotenv
from .domain import (
    DecisionContext,
    Domain,
    available_domains,
    get_domain,
    register_domain,
    set_default_domain,
)
from .knowledge import (
    InMemoryKnowledgeBase,
    KnowledgeBase,
    KnowledgeDoc,
    KnowledgeSnippet,
    get_knowledge_base,
    register_knowledge_base,
)
from .multiagent import AgentRouter, AgentService
from .cost import accumulate as accumulate_cost
from .cost import estimate_cost
from .observability import instrument_node, span
from .observability import status as tracing_status
from .tooling import (
    ToolRegistry,
    router_as_tool,
    run_tool_call,
    to_anthropic_tool,
    to_openai_tool,
)

_load_dotenv()

__all__ = [
    "Domain",
    "DecisionContext",
    "register_domain",
    "get_domain",
    "set_default_domain",
    "available_domains",
    "KnowledgeBase",
    "KnowledgeDoc",
    "KnowledgeSnippet",
    "InMemoryKnowledgeBase",
    "register_knowledge_base",
    "get_knowledge_base",
    "AgentService",
    "AgentRouter",
    "to_openai_tool",
    "to_anthropic_tool",
    "run_tool_call",
    "ToolRegistry",
    "router_as_tool",
    "estimate_cost",
    "accumulate_cost",
    "span",
    "instrument_node",
    "tracing_status",
]

