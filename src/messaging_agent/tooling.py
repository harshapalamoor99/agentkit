"""Tool-calling adapters — expose domain agents to OpenAI / Anthropic function calling.

`AgentService.as_tool()` returns a provider-neutral descriptor (name, description, JSON
``input_schema``, async ``callable``). This module converts that descriptor into the exact
shapes the OpenAI and Anthropic SDKs expect, and provides :func:`run_tool_call` /
:func:`dispatch_tool_call` to execute a model-issued tool call against the right agent and
return a JSON string the model can consume.

Typical loop (OpenAI):

    from messaging_agent import AgentService
    from messaging_agent.tooling import to_openai_tool, run_tool_call

    svc = AgentService("leasing")
    tools = [to_openai_tool(svc.as_tool())]
    resp = client.chat.completions.create(model=..., messages=..., tools=tools)
    for call in resp.choices[0].message.tool_calls:
        result_json = await run_tool_call(svc, call.function.name, call.function.arguments)
        # append {"role": "tool", "tool_call_id": call.id, "content": result_json}

For a multi-domain setup, register every agent's tool and route by tool name with
:class:`ToolRegistry`.
"""
from __future__ import annotations

import json
from typing import Any

from .multiagent import AgentRouter, AgentService


def to_openai_tool(descriptor: dict[str, Any]) -> dict[str, Any]:
    """Convert an ``as_tool()`` descriptor to an OpenAI Chat Completions tool spec."""
    return {
        "type": "function",
        "function": {
            "name": descriptor["name"],
            "description": descriptor["description"],
            "parameters": descriptor["input_schema"],
        },
    }


def to_anthropic_tool(descriptor: dict[str, Any]) -> dict[str, Any]:
    """Convert an ``as_tool()`` descriptor to an Anthropic Messages tool spec."""
    return {
        "name": descriptor["name"],
        "description": descriptor["description"],
        "input_schema": descriptor["input_schema"],
    }


def _coerce_args(arguments: Any) -> dict[str, Any]:
    """Accept either a JSON string (OpenAI) or a dict (Anthropic) of tool args."""
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        try:
            return json.loads(arguments or "{}")
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"tool arguments were not valid JSON: {exc}") from exc
    if isinstance(arguments, dict):
        return arguments
    raise TypeError(f"unsupported tool arguments type: {type(arguments)!r}")


async def run_tool_call(
    service: AgentService, name: str, arguments: Any, *, as_json: bool = True
) -> str | dict[str, Any]:
    """Execute a model-issued tool call against ``service``.

    ``arguments`` may be a JSON string (OpenAI) or a dict (Anthropic). Returns a JSON
    string by default (ready to hand back to the model), or the raw dict if
    ``as_json=False``. The descriptor name is informational; ``service`` decides routing.
    """
    args = _coerce_args(arguments)
    tool = service.as_tool()
    if name and name != tool["name"]:
        # Name mismatch is non-fatal: the service still owns the call, but surface it.
        result = {"warning": f"tool name {name!r} != {tool['name']!r}; ran anyway",
                  **(await tool["callable"](args))}
    else:
        result = await tool["callable"](args)
    return json.dumps(result, ensure_ascii=False, default=str) if as_json else result


class ToolRegistry:
    """Map tool names -> domain agents so a model can call any of several agents.

    Build it from a list of domain names (or the full set of registered domains) and use
    :meth:`tools_openai` / :meth:`tools_anthropic` to advertise them, then
    :meth:`dispatch` to execute whichever tool the model picked.
    """

    def __init__(self, domain_names: list[str] | None = None, *, app=None):
        from .domain import available_domains
        names = domain_names if domain_names is not None else available_domains()
        self._services: dict[str, AgentService] = {}
        self._by_tool_name: dict[str, AgentService] = {}
        for n in names:
            svc = AgentService(n, app=app)
            self._services[n] = svc
            self._by_tool_name[svc.as_tool()["name"]] = svc

    def descriptors(self) -> list[dict[str, Any]]:
        return [s.as_tool() for s in self._services.values()]

    def tools_openai(self) -> list[dict[str, Any]]:
        return [to_openai_tool(d) for d in self.descriptors()]

    def tools_anthropic(self) -> list[dict[str, Any]]:
        return [to_anthropic_tool(d) for d in self.descriptors()]

    async def dispatch(self, tool_name: str, arguments: Any, *, as_json: bool = True):
        """Run the tool call for ``tool_name`` against its agent."""
        svc = self._by_tool_name.get(tool_name)
        if svc is None:
            err = {"error": f"unknown tool {tool_name!r}",
                   "available_tools": sorted(self._by_tool_name)}
            return json.dumps(err) if as_json else err
        return await run_tool_call(svc, tool_name, arguments, as_json=as_json)


def router_as_tool(router: AgentRouter, *, name: str = "messaging_agent",
                   description: str | None = None) -> dict[str, Any]:
    """Expose an :class:`AgentRouter` itself as a single tool that auto-routes.

    Useful when you want one tool that the model calls with any record and the router
    picks the domain — instead of advertising one tool per domain.
    """
    async def _invoke(args: dict[str, Any]) -> dict[str, Any]:
        return await router.dispatch(args["record"], dataset=args.get("dataset"))

    return {
        "name": name,
        "description": description or
        "Route a record to the appropriate domain agent and return its decision.",
        "input_schema": {
            "type": "object",
            "properties": {
                "record": {"type": "object",
                           "description": "Input record; may include a 'domain' field to force routing."},
                "dataset": {"type": "array", "items": {"type": "object"},
                            "description": "Optional few-shot pool of labelled records."},
            },
            "required": ["record"],
        },
        "callable": _invoke,
    }
