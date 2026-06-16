"""Backwards-compatible shim: the mock LLM now lives in the package so it can be used
offline by the eval CLI / CI as well as the test suite. Import from there."""
from agentkit.mock_llm import MockLLMClient, _decide, _extract  # noqa: F401
