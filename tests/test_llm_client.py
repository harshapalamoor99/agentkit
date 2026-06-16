"""Provider-adapter tests (G4).

These exercise the real LLMClient provider branches (litellm/openai/azure/anthropic/
gemini) WITHOUT network by injecting fake SDK clients. They assert request shaping
(model, max_tokens, response_format, messages) and token-usage capture, and verify
detect_provider precedence. An opt-in live smoke test runs only when a key is present.

Async calls are driven with asyncio.run to match the suite's convention (no
pytest-asyncio dependency).
"""
from __future__ import annotations

import asyncio

import pytest

from agentkit import llm_client
from agentkit.llm_client import LLMClient, detect_provider


def _run(coro):
    return asyncio.run(coro)


# --- Fakes -----------------------------------------------------------------

class _Usage:
    prompt_tokens = 11
    completion_tokens = 7
    total_tokens = 18


class _Msg:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})


class _OpenAIStyleResp:
    model = "fake-model"
    usage = _Usage()

    def __init__(self, content='{"ok":true}'):
        self.choices = [_Msg(content)]


class _FakeCompletions:
    def __init__(self, store):
        self._store = store

    async def create(self, **kwargs):
        self._store["kwargs"] = kwargs
        return _OpenAIStyleResp()


class _FakeOpenAIClient:
    def __init__(self, store, **init):
        self._store = store
        self._store["init"] = init
        self.chat = type("C", (), {"completions": _FakeCompletions(store)})


# --- detect_provider precedence -------------------------------------------

def test_detect_provider_precedence(monkeypatch):
    for k in ("LITELLM_API_KEY", "LITELLM_API_BASE", "ANTHROPIC_API_KEY",
              "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_KEY",
              "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert detect_provider() is None

    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    assert detect_provider() == "openai"  # openai beats gemini

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert detect_provider() == "anthropic"  # anthropic beats openai

    monkeypatch.setenv("LITELLM_API_KEY", "x")
    monkeypatch.setenv("LITELLM_API_BASE", "http://gw/v1")
    assert detect_provider() == "litellm"  # litellm wins overall


# --- litellm ---------------------------------------------------------------

def test_litellm_shaping_and_usage(monkeypatch):
    store: dict = {}
    monkeypatch.setenv("LITELLM_MODEL", "gemini-flash-lite")
    monkeypatch.setattr(llm_client, "_litellm_client", _FakeOpenAIClient(store))

    client = LLMClient(provider="litellm")
    out = _run(client.generate("sys", "usr", max_tokens=400, temperature=0.0))

    assert out == '{"ok":true}'
    assert store["kwargs"]["model"] == "gemini-flash-lite"
    assert store["kwargs"]["max_tokens"] == 400
    assert store["kwargs"]["messages"][0]["role"] == "system"
    assert client.last_usage["total_tokens"] == 18
    assert client.last_usage["model"] == "fake-model"


def test_litellm_retries_without_response_format(monkeypatch):
    """Gateways that reject response_format must trigger a retry without it."""
    calls: list = []

    class _PickyCompletions:
        async def create(self, **kwargs):
            calls.append(kwargs)
            if "response_format" in kwargs:
                raise ValueError("response_format unsupported")
            return _OpenAIStyleResp()

    fake = type("X", (), {"chat": type("C", (), {"completions": _PickyCompletions()})})()
    monkeypatch.setenv("LITELLM_MODEL", "m")
    monkeypatch.setattr(llm_client, "_litellm_client", fake)

    out = _run(LLMClient(provider="litellm").generate("s", "u", max_tokens=10))
    assert out == '{"ok":true}'
    assert len(calls) == 2  # first with response_format (fails), retry without


# --- openai ----------------------------------------------------------------

def test_openai_shaping(monkeypatch):
    import openai
    store: dict = {}
    monkeypatch.setattr(openai, "AsyncOpenAI",
                        lambda *a, **k: _FakeOpenAIClient(store, **k))

    out = _run(LLMClient(provider="openai").generate("sys", "usr", max_tokens=222))
    assert out == '{"ok":true}'
    assert store["kwargs"]["max_tokens"] == 222
    assert store["kwargs"]["response_format"] == {"type": "json_object"}


# --- azure -----------------------------------------------------------------

def test_azure_shaping(monkeypatch):
    import openai
    store: dict = {}
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://az.example/")
    monkeypatch.setattr(openai, "AsyncAzureOpenAI",
                        lambda *a, **k: _FakeOpenAIClient(store, **k))

    out = _run(LLMClient(provider="azure").generate("sys", "usr", max_tokens=99))
    assert out == '{"ok":true}'
    assert store["kwargs"]["max_tokens"] == 99
    assert store["kwargs"]["response_format"] == {"type": "json_object"}


# --- anthropic -------------------------------------------------------------

def test_anthropic_shaping(monkeypatch):
    import anthropic
    store: dict = {}

    class _Block:
        type = "text"
        text = "hello-claude"

    class _Messages:
        async def create(self, **kwargs):
            store["kwargs"] = kwargs
            return type("R", (), {"content": [_Block()]})

    class _FakeAnthropic:
        def __init__(self, *a, **k):
            self.messages = _Messages()

    monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAnthropic)
    out = _run(LLMClient(provider="anthropic").generate("sys", "usr", max_tokens=55))
    assert out == "hello-claude"
    assert store["kwargs"]["system"] == "sys"
    assert store["kwargs"]["max_tokens"] == 55


# --- gemini ----------------------------------------------------------------

def test_gemini_shaping(monkeypatch):
    from google import genai
    store: dict = {}

    class _Models:
        async def generate_content(self, **kwargs):
            store["kwargs"] = kwargs
            return type("R", (), {"text": "hello-gemini"})

    class _FakeClient:
        def __init__(self, *a, **k):
            self.aio = type("A", (), {"models": _Models()})

    monkeypatch.setattr(genai, "Client", _FakeClient)
    out = _run(LLMClient(provider="gemini").generate("sys", "usr", max_tokens=33))
    assert out == "hello-gemini"
    assert store["kwargs"]["model"]


# --- live smoke (opt-in) ---------------------------------------------------

@pytest.mark.skipif(detect_provider() is None,
                    reason="no provider key set; live smoke test skipped")
def test_live_smoke():
    client = LLMClient()
    text = _run(client.generate("Return strict JSON.", 'Return {"ok":true}.',
                                max_tokens=16))
    assert isinstance(text, str) and text.strip()
