"""Static configuration and shared constants."""
from __future__ import annotations

# Channels we know how to address, mapped to their consent flag in record["consent"].
CHANNEL_CONSENT_FLAG = {
    "sms": "sms_opt_in",
    "email": "email_opt_in",
    "voice": "voice_opt_in",
}

# Business hours (local time) within which we *prefer* to schedule a message.
BUSINESS_HOUR_START = 9
BUSINESS_HOUR_END = 18  # exclusive upper bound for "send before 6pm"

# TCPA legal quiet-hours window (AC-4): messages may only be DISPATCHED between 8am
# and 9pm local time. This is the hard legal gate; business hours (9-6) sit inside it.
TCPA_EARLIEST_HOUR = 8
TCPA_LATEST_HOUR = 21  # exclusive: last permissible hour is 20:59

# Per-channel default send hour (still within business hours). Used only as a
# deterministic default; the actual decision is derived from data, not personas.
DEFAULT_SEND_HOUR = {"sms": 9, "email": 10, "voice": 11}

# Valid next_action types (AC-13).
VALID_NEXT_ACTION_TYPES = {"start_cadence", "follow_up_in_days", "no_action"}

# Personas / lifecycle stages that are legitimate. Anything outside this set that
# looks like a role-override attempt is neutralized (AC-19).
KNOWN_PERSONAS = {"prospect", "applicant", "resident", "renewal", "lead", "guest"}

# Hard limits to defend against oversized / garbage input (AC-21).
MAX_NAME_LEN = 80
MAX_TEXT_LEN = 2000
MAX_LIST_ITEMS = 20

# LLM call budget.
# TOTAL_LLM_BUDGET_S caps the *cumulative* time across all retries so the end-to-end
# request stays under the 2s SLA even on the retry path. LLM_TIMEOUT_S caps a single
# attempt. The effective timeout for any attempt is min(LLM_TIMEOUT_S, remaining budget).
# Env-overridable so latency can be tuned per deployment / gateway without code edits.
import os as _os

LLM_TIMEOUT_S = float(_os.getenv("LLM_TIMEOUT_S", "1.6"))
TOTAL_LLM_BUDGET_S = float(_os.getenv("TOTAL_LLM_BUDGET_S", "1.8"))
MAX_RETRIES = int(_os.getenv("MAX_RETRIES", "2"))
# Output token ceiling. With "thinking" disabled the JSON message fits easily; keep
# this small to minimize generation latency. Raise it if a thinking model is used.
LLM_MAX_TOKENS = int(_os.getenv("LLM_MAX_TOKENS", "400"))

# Circuit breaker (AC-11): after this many consecutive LLM failures the breaker opens
# and subsequent calls fast-abort (no fabrication) until the cooldown elapses.
CIRCUIT_BREAKER_THRESHOLD = int(_os.getenv("CIRCUIT_BREAKER_THRESHOLD", "3"))
CIRCUIT_BREAKER_COOLDOWN_S = float(_os.getenv("CIRCUIT_BREAKER_COOLDOWN_S", "30"))

# Prompt template version (AC-12 decision lineage). Bump when the system prompt changes.
PROMPT_TEMPLATE_VERSION = "v3-llm-only-2026.06"

# Named LLM-judge quality gates (AC-13). A deployment whose mean score on either
# dimension drops below these must fail CI.
JUDGE_FAITHFULNESS_MIN = float(_os.getenv("JUDGE_FAITHFULNESS_MIN", "0.95"))
JUDGE_CONTEXT_PRECISION_MIN = float(_os.getenv("JUDGE_CONTEXT_PRECISION_MIN", "0.90"))

# Asset classes that are rent-regulated (AC-9). For these the agent prioritizes
# statutory/recertification messaging and must not push unregulated pricing incentives.
REGULATED_ASSET_CLASSES = {"affordable", "hud", "lihtc", "section8", "section42", "tax_credit"}

# G6: optional runtime fair-housing LLM-judge gate. OFF by default so the always-on
# regex floor is the only runtime cost; when enabled (and a provider + budget exist) the
# parser asks the judge to confirm the body is fair-housing-clean and forces a safe
# no-send on a high-confidence violation. The judge is advisory elsewhere; here it can veto.
RUNTIME_FAIRHOUSING_JUDGE = _os.getenv("RUNTIME_FAIRHOUSING_JUDGE", "0").lower() in ("1", "true", "yes")
FAIRHOUSING_JUDGE_MIN_CONFIDENCE = float(_os.getenv("FAIRHOUSING_JUDGE_MIN_CONFIDENCE", "0.8"))
FAIRHOUSING_JUDGE_TIMEOUT_S = float(_os.getenv("FAIRHOUSING_JUDGE_TIMEOUT_S", "1.0"))

# Keep-warm pinger (demo/low-traffic): the gateway connection + model go cold after a
# few seconds idle, so the first request after a pause pays a ~2-3s cold-start. When
# enabled, a background task issues a tiny ping whenever the client has been idle longer
# than KEEPWARM_INTERVAL_S, holding latency at the warm ~0.8s for the whole session.
# OFF by default (steady prod traffic keeps itself warm); turn on for demos.
KEEPWARM_ENABLED = _os.getenv("KEEPWARM_ENABLED", "0").lower() in ("1", "true", "yes")
KEEPWARM_INTERVAL_S = float(_os.getenv("KEEPWARM_INTERVAL_S", "15"))
