"""Input-side PII tokenization / zero-trust masking (AC-8).

Sensitive financial / identity / screening fields are *never* passed to the LLM in raw
form. Before generation we scrub them into coarse, non-identifying categories (e.g.
exact income -> income_band: "80k-100k", credit score -> credit_tier: "Tier_1",
background result -> screening_status: "passed"). The LLM only ever sees the category,
so it cannot reflect raw PII into a message body. A separate output check (in the parser)
quarantines any message that still contains raw PII-looking numbers.
"""
from __future__ import annotations

import copy
import re
from typing import Any

# Fields that carry raw sensitive PII and must be tokenized out before the LLM sees them.
SENSITIVE_FIELDS = (
    "income", "annual_income", "household_income", "monthly_income",
    "ssn", "social_security", "credit_score", "fico", "fico_score",
    "background_check", "screening", "screening_result", "criminal_history",
    "eviction_history", "bank_balance", "dob", "date_of_birth",
)


def _income_band(value: Any) -> str | None:
    try:
        amount = float(re.sub(r"[^\d.]", "", str(value)))
    except (ValueError, TypeError):
        return None
    if amount <= 0:
        return None
    # Monthly figures (small) are annualized for banding.
    if amount < 1000:
        return None
    if amount < 10000:
        amount *= 12
    bands = [(40000, "under_40k"), (60000, "40k-60k"), (80000, "60k-80k"),
             (100000, "80k-100k"), (150000, "100k-150k")]
    for ceiling, label in bands:
        if amount < ceiling:
            return label
    return "150k_plus"


def _credit_tier(value: Any) -> str | None:
    try:
        score = int(re.sub(r"[^\d]", "", str(value)))
    except (ValueError, TypeError):
        return None
    if not (300 <= score <= 850):
        return None
    if score >= 740:
        return "Tier_1"
    if score >= 670:
        return "Tier_2"
    if score >= 580:
        return "Tier_3"
    return "Tier_4"


def _screening_status(value: Any) -> str:
    s = str(value).strip().lower()
    if s in ("pass", "passed", "approved", "clear", "true", "ok"):
        return "passed"
    if s in ("fail", "failed", "denied", "rejected", "false"):
        return "review_required"
    return "pending"


def tokenize_record(record: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Return (record_with_pii_removed, audit_notes).

    Raw sensitive fields are removed from `input`/`input.profile` and replaced with
    coarse categories under `input.screening_metadata`. The returned record is safe to
    hand to the LLM. Notes document each tokenization for the audit trail (AC-12).
    """
    rec = copy.deepcopy(record)
    notes: list[str] = []
    inp = rec.setdefault("input", {})
    profile = inp.setdefault("profile", {})
    meta: dict[str, Any] = {}

    def consume(container: dict[str, Any], key: str) -> Any:
        return container.pop(key, None) if key in container else None

    for container in (inp, profile):
        for key in list(container.keys()):
            lk = key.lower()
            if lk not in SENSITIVE_FIELDS:
                continue
            raw = consume(container, key)
            if raw is None:
                continue
            if "income" in lk:
                band = _income_band(raw)
                meta["income_band"] = band or "unknown"
                meta["income_verified"] = True
                notes.append(f"pii_tokenized: {key} -> income_band")
            elif lk in ("credit_score", "fico", "fico_score"):
                tier = _credit_tier(raw)
                meta["credit_tier"] = tier or "unknown"
                notes.append(f"pii_tokenized: {key} -> credit_tier")
            elif lk in ("ssn", "social_security"):
                meta["ssn_on_file"] = True
                notes.append(f"pii_tokenized: {key} -> ssn_on_file (raw dropped)")
            elif lk in ("background_check", "screening", "screening_result",
                        "criminal_history", "eviction_history"):
                meta["screening_status"] = _screening_status(raw)
                notes.append(f"pii_tokenized: {key} -> screening_status")
            elif lk in ("dob", "date_of_birth", "bank_balance"):
                notes.append(f"pii_tokenized: {key} -> dropped")

    if meta:
        inp["screening_metadata"] = meta
    return rec, notes


# Raw-PII patterns to quarantine in OUTPUT bodies (beyond phone/email/ssn in safety_rules):
# long bare numbers that look like income, balances, or screening figures.
_RAW_NUMBER_RE = re.compile(r"(?<!\d)(?:\$\s?)?\d{4,}(?:,\d{3})*(?:\.\d+)?(?!\d)")
_CREDIT_SCORE_RE = re.compile(r"\b(?:credit\s+score|fico)\b[^\d]{0,12}\b[3-8]\d{2}\b", re.I)


def output_reflects_raw_pii(text: str) -> list[str]:
    """Detect raw sensitive PII leaking into a generated body (AC-8 quarantine)."""
    found: list[str] = []
    if not text:
        return found
    if _CREDIT_SCORE_RE.search(text):
        found.append("credit_score")
    # A long bare currency/number is suspicious; allow short numbers (e.g. "2" tours,
    # "24/7"). We exempt ONLY a standalone 4-digit calendar year (1900-2099) with no
    # currency prefix — "$2000" or "income 20800" must still be flagged.
    for m in _RAW_NUMBER_RE.finditer(text):
        token = m.group(0)
        has_currency = "$" in token
        digits = re.sub(r"\D", "", token)
        if len(digits) < 4:
            continue
        # A real year has no currency sign, comma grouping, or decimal — so "2,000"
        # and "$2000" are flagged while "in 2026" is not.
        formatted = ("," in token) or ("." in token)
        is_bare_year = (len(digits) == 4 and not has_currency and not formatted
                        and 1900 <= int(digits) <= 2099)
        if is_bare_year:
            continue
        found.append("raw_number")
    return found
