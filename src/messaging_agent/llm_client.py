"""Provider-agnostic async LLM client.

Auto-detects the provider from environment variables so the user can plug in any
key later without code changes:
  ANTHROPIC_API_KEY            -> Anthropic Claude
  OPENAI_API_KEY               -> OpenAI
  AZURE_OPENAI_API_KEY + ...   -> Azure OpenAI
  GOOGLE_API_KEY / GEMINI_API_KEY -> Google Gemini

Provider precedence when several keys are set: LiteLLM gateway > Anthropic > Azure >
OpenAI > Gemini (see `detect_provider`). If no key is present, `provider` is None and
the agent is LLM-only — callers abort to a safe no-send rather than fabricating output.
"""
from __future__ import annotations

import asyncio
import os
import time


def detect_provider() -> str | None:
    # Precedence (first match wins) when multiple keys are present:
    # LiteLLM gateway > Anthropic > Azure > OpenAI > Gemini.
    if os.getenv("LITELLM_API_KEY") and os.getenv("LITELLM_API_BASE"):
        return "litellm"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return "gemini"
    return None


def default_model(provider: str) -> str:
    return {
        "litellm": os.getenv("LITELLM_MODEL", "gpt-4o-mini"),
        "anthropic": os.getenv("MESSAGING_AGENT_MODEL", "claude-sonnet-4-5"),
        "openai": os.getenv("MESSAGING_AGENT_MODEL", "gpt-4o-mini"),
        "azure": os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini"),
        "gemini": os.getenv("MESSAGING_AGENT_MODEL", "gemini-2.0-flash"),
    }[provider]


def reasoning_effort() -> str | None:
    """Disable model 'thinking' by default (Gemini 2.5 etc.) to stay under the
    latency budget. Override with LLM_REASONING_EFFORT (e.g. 'low'/'medium'/'high')."""
    val = os.getenv("LLM_REASONING_EFFORT", "none")
    return val or None


def embedding_model(provider: str) -> str:
    """Embedding model used for semantic-match eval (G2). OpenAI-compatible default."""
    if provider == "gemini":
        return os.getenv("EMBEDDING_MODEL", "text-embedding-004")
    return os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")


# Reused across calls so we keep a warm HTTP/TLS connection (cold connects cost ~2s).
_litellm_client = None


def _get_litellm_client():
    global _litellm_client
    if _litellm_client is None:
        from openai import AsyncOpenAI

        _litellm_client = AsyncOpenAI(
            api_key=os.environ["LITELLM_API_KEY"],
            base_url=os.environ["LITELLM_API_BASE"],
        )
    return _litellm_client


class LLMClient:
    def __init__(self, provider: str | None = None):
        self.provider = provider or detect_provider()
        self.last_usage: dict | None = None  # AC-12: token metrics from the most recent call
        self._last_call_monotonic: float = 0.0  # for the keep-warm idle check
        self._keepwarm_task: asyncio.Task | None = None

    @property
    def available(self) -> bool:
        return self.provider is not None

    async def warmup(self, rounds: int = 2) -> None:
        """Best-effort: prime the TCP/TLS connection *and* the gateway's model
        cold-start so the first real record isn't slow. The first post-connect call
        to this UAT gateway can be ~1.8s while steady-state is ~0.8s, so we issue a
        couple of priming calls."""
        for _ in range(max(1, rounds)):
            try:
                await self.generate("Return strict JSON.", 'Return {"ok":true}.',
                                    max_tokens=8)
            except Exception:
                return

    async def _keepwarm_loop(self, interval_s: float) -> None:
        """Issue a tiny ping whenever the client has been idle >= interval_s, so the
        gateway connection + model stay hot for the whole (demo) session. Real traffic
        resets the idle timer, so this only fires during pauses. Best-effort + silent."""
        while True:
            try:
                await asyncio.sleep(interval_s)
                idle = time.monotonic() - self._last_call_monotonic
                if idle >= interval_s:
                    try:
                        await self.generate("Return strict JSON.",
                                            'Return {"ok":true}.', max_tokens=8)
                    except Exception:
                        pass
            except asyncio.CancelledError:
                break

    def start_keepwarm(self, interval_s: float | None = None) -> bool:
        """Start the background keep-warm pinger (idempotent). Returns True if started.

        Requires a running event loop and an available provider. Call stop_keepwarm()
        (or rely on process exit) to stop it.
        """
        from . import config
        if not self.available or self._keepwarm_task is not None:
            return False
        interval = config.KEEPWARM_INTERVAL_S if interval_s is None else interval_s
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        self._last_call_monotonic = time.monotonic()
        self._keepwarm_task = loop.create_task(self._keepwarm_loop(interval))
        return True

    async def stop_keepwarm(self) -> None:
        task = self._keepwarm_task
        self._keepwarm_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def generate(self, system: str, user: str, *, max_tokens: int = 700,
                        temperature: float = 0.0) -> str:
        self._last_call_monotonic = time.monotonic()
        if self.provider == "litellm":
            return await self._litellm(system, user, max_tokens, temperature)
        if self.provider == "anthropic":
            return await self._anthropic(system, user, max_tokens, temperature)
        if self.provider == "openai":
            return await self._openai(system, user, max_tokens, temperature)
        if self.provider == "azure":
            return await self._azure(system, user, max_tokens, temperature)
        if self.provider == "gemini":
            return await self._gemini(system, user, max_tokens, temperature)
        raise RuntimeError("No LLM provider configured")

    async def embed(self, texts: list[str]) -> list[list[float]] | None:
        """Return embeddings for `texts`, or None if the provider has no embeddings
        endpoint (e.g. Anthropic) or the call fails. Best-effort: used only by the
        offline semantic-match eval (G2), which falls back to a lexical proxy on None.
        """
        if not texts:
            return []
        try:
            if self.provider in ("litellm", "openai", "azure"):
                return await self._embed_openai_compatible(texts)
            if self.provider == "gemini":
                return await self._embed_gemini(texts)
        except Exception:
            return None
        return None

    async def _embed_openai_compatible(self, texts):
        if self.provider == "litellm":
            client = _get_litellm_client()
        elif self.provider == "openai":
            from openai import AsyncOpenAI
            client = AsyncOpenAI()
        else:
            from openai import AsyncAzureOpenAI
            client = AsyncAzureOpenAI(
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            )
        resp = await client.embeddings.create(
            model=embedding_model(self.provider), input=texts)
        return [d.embedding for d in resp.data]

    async def _embed_gemini(self, texts):
        from google import genai
        client = genai.Client()
        resp = await client.aio.models.embed_content(
            model=embedding_model("gemini"), contents=texts)
        return [e.values for e in resp.embeddings]

    async def _litellm(self, system, user, max_tokens, temperature):
        """OpenAI-compatible LiteLLM proxy (any backend model behind a gateway)."""
        client = _get_litellm_client()
        extra = {}
        eff = reasoning_effort()
        if eff:
            extra["reasoning_effort"] = eff
        kwargs = dict(
            model=default_model("litellm"),
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        try:
            resp = await client.chat.completions.create(
                response_format={"type": "json_object"},
                extra_body=extra or None, **kwargs)
        except Exception:
            # Some gateway-fronted models reject response_format; the prompt already
            # mandates strict JSON, so retry without it rather than fail the call.
            resp = await client.chat.completions.create(extra_body=extra or None, **kwargs)
        self._capture_usage(resp)
        return resp.choices[0].message.content or ""

    def _capture_usage(self, resp) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            self.last_usage = None
            return
        self.last_usage = {
            "model": getattr(resp, "model", None),
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    async def _anthropic(self, system, user, max_tokens, temperature):
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic()
        resp = await client.messages.create(
            model=default_model("anthropic"),
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self._capture_anthropic_usage(resp)
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    def _capture_anthropic_usage(self, resp) -> None:
        usage = getattr(resp, "usage", None)
        if usage is None:
            self.last_usage = None
            return
        prompt = getattr(usage, "input_tokens", None)
        completion = getattr(usage, "output_tokens", None)
        total = (prompt or 0) + (completion or 0) if (prompt or completion) else None
        self.last_usage = {
            "model": getattr(resp, "model", None),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }

    async def _openai(self, system, user, max_tokens, temperature):
        from openai import AsyncOpenAI

        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model=default_model("openai"),
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        self._capture_usage(resp)
        return resp.choices[0].message.content or ""

    async def _azure(self, system, user, max_tokens, temperature):
        from openai import AsyncAzureOpenAI

        client = AsyncAzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01"),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        )
        resp = await client.chat.completions.create(
            model=default_model("azure"),
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        self._capture_usage(resp)
        return resp.choices[0].message.content or ""

    async def _gemini(self, system, user, max_tokens, temperature):
        from google import genai
        from google.genai import types

        client = genai.Client()
        resp = await client.aio.models.generate_content(
            model=default_model("gemini"),
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
                temperature=temperature,
                response_mime_type="application/json",
            ),
        )
        self._capture_gemini_usage(resp)
        return resp.text or ""

    def _capture_gemini_usage(self, resp) -> None:
        usage = getattr(resp, "usage_metadata", None)
        if usage is None:
            self.last_usage = None
            return
        prompt = getattr(usage, "prompt_token_count", None)
        completion = getattr(usage, "candidates_token_count", None)
        total = getattr(usage, "total_token_count", None)
        self.last_usage = {
            "model": default_model("gemini"),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }
