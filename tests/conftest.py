import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Keep the whole test suite hermetic: never let a real provider key (incl. a
# LiteLLM gateway) leak in and turn deterministic-fallback tests into live calls.
for _k in ("LITELLM_API_KEY", "LITELLM_API_BASE", "LITELLM_MODEL",
           "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY",
           "AZURE_OPENAI_ENDPOINT", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
    os.environ.pop(_k, None)

# Pin a tight, deterministic latency budget for the suite regardless of a developer's
# local `.env` (which may relax it for live demos, e.g. LLM_TIMEOUT_S=8). The production
# default is 1.8s (see config.py); tests use a tighter budget so the SLA assertions keep
# a wide margin and never flake on a loaded CI runner, while still proving the property
# (a slow/unparseable model aborts safely *under* the shared deadline). `.env` uses
# os.environ.setdefault, so setting these here always wins.
os.environ["LLM_TIMEOUT_S"] = "0.8"
os.environ["TOTAL_LLM_BUDGET_S"] = "1.0"

import pytest  # noqa: E402

from _mock_llm import MockLLMClient  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Circuit breakers are process-level globals (default + per-key registry); reset
    them between tests so a failure-injection test can't leave one OPEN for later cases."""
    import messaging_agent.circuit_breaker as cb
    cb.reset_all()
    yield
    cb.reset_all()


@pytest.fixture(autouse=True)
def _mock_llm():
    """The agent is LLM-only; install a hermetic mock 'model' so the pipeline
    produces a valid message offline. Tests that need specific LLM behavior
    (latency, failures) override messaging_agent.nodes.llm._client themselves."""
    import messaging_agent.nodes.llm as llmnode
    original = llmnode._client
    llmnode._client = MockLLMClient()
    yield
    llmnode._client = original
