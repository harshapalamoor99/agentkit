"""OpenTelemetry tracing for the agent graph — optional, zero-config, no-op by default.

Every graph node runs inside a span so a full pipeline invocation
(intake → safety → … → emit) becomes one trace. Spans carry decision attributes
(domain, tenant, retry count, abort reason, token cost) so you can slice latency and
spend per domain/tenant in any OTLP backend (Jaeger, Tempo, Honeycomb, Datadog, …).

**Graceful degradation is the whole point**: if ``opentelemetry-sdk`` isn't installed
*or* tracing isn't switched on, this module returns no-op spans with effectively zero
overhead, so tests and lightweight deployments are unaffected. Turn it on with:

    pip install 'messaging-agent[otel]'        # opentelemetry-sdk + otlp exporter
    export MESSAGING_AGENT_TRACING=true
    export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317   # optional; else console

This is independent of the LangSmith hook in ``prod/tracing.py`` (that one is wired via
LangChain env vars); both can run together.
"""
from __future__ import annotations

import functools
import inspect
import os
from contextlib import contextmanager
from typing import Any, Callable

_TRACER: Any = None
_CONFIGURED = False


def tracing_enabled() -> bool:
    return os.getenv("MESSAGING_AGENT_TRACING", "").lower() in ("1", "true", "yes")


def _configure() -> Any:
    """Build (once) a real OTel tracer if possible, else return None (→ no-op)."""
    global _CONFIGURED, _TRACER
    if _CONFIGURED:
        return _TRACER
    _CONFIGURED = True
    if not tracing_enabled():
        _TRACER = None
        return None
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({
            "service.name": os.getenv("OTEL_SERVICE_NAME", "messaging-agent"),
        })
        provider = TracerProvider(resource=resource)
        exporter = _make_exporter()
        if exporter is not None:
            provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _TRACER = trace.get_tracer("messaging_agent")
    except Exception:
        _TRACER = None
    return _TRACER


def _make_exporter():
    """OTLP exporter when an endpoint is set; else console (dev); else None."""
    try:
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            return OTLPSpanExporter()
    except Exception:
        pass
    try:
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        if os.getenv("MESSAGING_AGENT_TRACE_CONSOLE", "").lower() in ("1", "true", "yes"):
            return ConsoleSpanExporter()
    except Exception:
        pass
    return None


class _NoopSpan:
    def set_attribute(self, *_a, **_k):  # noqa: D401
        return None

    def record_exception(self, *_a, **_k):
        return None

    def set_status(self, *_a, **_k):
        return None


@contextmanager
def span(name: str, attributes: dict[str, Any] | None = None):
    """Context manager yielding a span (real or no-op). Always safe to use."""
    tracer = _configure()
    if tracer is None:
        yield _NoopSpan()
        return
    with tracer.start_as_current_span(name) as sp:
        for key, value in (attributes or {}).items():
            if value is not None:
                try:
                    sp.set_attribute(key, value)
                except Exception:
                    pass
        yield sp


def _resolve_domain(state: dict[str, Any]) -> str | None:
    try:
        from .domain import get_domain

        return getattr(get_domain(state), "name", None)
    except Exception:
        return None


def _entry_attributes(name: str, state: dict[str, Any]) -> dict[str, Any]:
    constraints = state.get("constraints", {}) or {}
    tenant = state.get("tenant", {}) or {}
    return {
        "agent.node": name,
        "agent.task_id": state.get("task_id"),
        "agent.domain": _resolve_domain(state),
        "agent.tenant_id": constraints.get("tenant_id") or tenant.get("tenant_id"),
        "agent.retry_count": state.get("retry_count", 0),
    }


def _record_result(sp: Any, result: Any) -> None:
    if not isinstance(result, dict):
        return
    for attr, key in (
        ("agent.abort_reason", "abort_reason"),
        ("agent.parse_status", "parse_status"),
        ("agent.latency_ms", "latency_ms"),
    ):
        val = result.get(key)
        if val is not None:
            sp.set_attribute(attr, val)
    if "should_send" in result:
        sp.set_attribute("agent.should_send", bool(result.get("should_send")))
    cost = (result.get("lineage", {}) or {}).get("cost")
    if isinstance(cost, dict):
        if cost.get("total_tokens") is not None:
            sp.set_attribute("agent.total_tokens", cost["total_tokens"])
        if cost.get("cost_usd") is not None:
            sp.set_attribute("agent.cost_usd", cost["cost_usd"])


def instrument_node(name: str, fn: Callable) -> Callable:
    """Wrap a graph node so each invocation opens a span. Transparent: the wrapper
    returns exactly what the node returns and re-raises any exception (after recording
    it). Handles both sync and async nodes."""
    if inspect.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def _async_wrapper(state, *args, **kwargs):
            with span(f"node.{name}", _entry_attributes(name, state)) as sp:
                try:
                    result = await fn(state, *args, **kwargs)
                except Exception as exc:
                    sp.record_exception(exc)
                    raise
                _record_result(sp, result)
                return result

        return _async_wrapper

    @functools.wraps(fn)
    def _sync_wrapper(state, *args, **kwargs):
        with span(f"node.{name}", _entry_attributes(name, state)) as sp:
            try:
                result = fn(state, *args, **kwargs)
            except Exception as exc:
                sp.record_exception(exc)
                raise
            _record_result(sp, result)
            return result

    return _sync_wrapper


def status() -> dict[str, Any]:
    """Report observability config (mirrors prod/tracing.status())."""
    return {
        "tracing_enabled": tracing_enabled(),
        "tracer_active": _configure() is not None,
        "otlp_endpoint": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        "service_name": os.getenv("OTEL_SERVICE_NAME", "messaging-agent"),
    }
