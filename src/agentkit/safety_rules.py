"""Data-driven safety scanning and sanitization.

Detects prompt-injection (AC-16), jailbreak/role-override (AC-19), toxic /
fair-housing-violating personalization content (AC-20), oversized/garbage input
(AC-21), and encoding injection (AC-22). Produces a sanitized copy of the record
that contains only safe, neutral personalization data.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from . import config

# --- Pattern libraries (data-driven scans, not record-specific hardcoding) ---

INJECTION_PATTERNS = [
    r"ignore\s+(all|previous|prior|the)\b.*?instruction",
    r"disregard\s+(your|the|all|previous)",
    r"you\s+are\s+now\b",
    r"system\s+prompt",
    r"reveal\s+your\s+(system\s+)?prompt",
    r"output\s+your\s+(system\s+)?prompt",
    r"act\s+as\b",
    r"new\s+instructions?\s*:",
    r"forget\s+(everything|all|your)",
]

# SQL / markup / script injection that should never reach the message body.
CODE_INJECTION_PATTERNS = [
    r"drop\s+table",
    r";\s*--",
    r"--\s*$",
    r"<\s*script",
    r"</\s*script",
    r"<[^>]+>",          # any html tag
    r"\balert\s*\(",
    r"\bunion\s+select\b",
    r"\bor\s+1\s*=\s*1\b",
]

# Role-override attempts embedded in structural fields (persona, lifecycle_stage).
ROLE_OVERRIDE_PATTERNS = [
    r"\bsystem\b",
    r"\badmin\b",
    r"\broot\b",
    r"ignore\s+safety",
    r"bypass",
    r"developer\s+mode",
]

# Toxic / fair-housing violating content (AC-09/AC-10/AC-20). Includes protected-class
# steering / preference / exclusion language across the FHA-protected classes (race,
# color, religion, national origin, sex, familial status, disability). Matched loosely
# via word boundaries; this is the always-on regex floor under the optional LLM judge.
TOXIC_PATTERNS = [
    r"whites?\s+only",
    r"no\s+(kids|children|child|families|family|infants|toddlers|disabled|handicap(ped)?|"
    r"wheelchairs?|immigrants?|foreigners?|section\s*8)",
    r"\b(christ(ian|ians)?|muslim|islamic|jewish|catholic|hindu|buddhist)\s+only\b",
    r"\b(prefer|preferred|ideal|looking\s+for|seeking)\s+(a\s+)?"
    r"(white|black|asian|hispanic|latino|christian|muslim|jewish|male|female|young|"
    r"single|couple)\b",
    r"\b(no|not\s+for)\s+(blacks?|asians?|hispanics?|latinos?|gays?|lesbians?)\b",
    r"\b(adults?\s+only|no\s+pets\s+no\s+kids|mature\s+adults?\s+only)\b",
    r"\bperfect\s+for\s+(a\s+)?(single|young|christian|professional\s+male)\b",
    r"\bslur\b",            # placeholder token used in the spec's test data
    r"\[slur\]",
    r"hate",
]

# PII patterns (AC-09) — used both to scan inputs and to validate output bodies.
PII_PATTERNS = {
    "phone": r"(?<!\d)(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}(?!\d)",
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
}

# Additional PII patterns scanned ONLY in generated OUTPUT bodies (not used to gate
# inputs, where they would over-trigger on legitimate property metadata). Broadens the
# output leak net beyond phone/email/SSN (G6).
OUTPUT_EXTRA_PII_PATTERNS = {
    "street_address": r"\b\d{1,6}\s+(?:[NSEW]\.?\s+)?[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+"
                      r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|"
                      r"Court|Ct|Way|Circle|Cir|Place|Pl|Terrace|Ter|Parkway|Pkwy)\b\.?",
    "credit_card": r"\b(?:\d[ -]?){13,16}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}

_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)
_CODE_RE = re.compile("|".join(CODE_INJECTION_PATTERNS), re.IGNORECASE)
_ROLE_RE = re.compile("|".join(ROLE_OVERRIDE_PATTERNS), re.IGNORECASE)
_TOXIC_RE = re.compile("|".join(TOXIC_PATTERNS), re.IGNORECASE)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Bidi / zero-width / format characters that can corrupt output (AC-22).
_FORMAT_CHARS_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")


def _coerce_str(value: Any) -> str:
    if isinstance(value, str):
        return value
    return str(value)


def clean_text(value: Any, max_len: int = config.MAX_TEXT_LEN) -> str:
    """Normalize a string: strip control/format chars, NFC-normalize, cap length."""
    s = _coerce_str(value)
    s = s.replace("\x00", "")
    s = _CONTROL_RE.sub("", s)
    s = _FORMAT_CHARS_RE.sub("", s)
    s = unicodedata.normalize("NFC", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s.strip()


def is_unsafe_text(value: Any) -> list[str]:
    """Return a list of violation categories found in a free-text value."""
    s = _coerce_str(value)
    found = []
    if _INJECTION_RE.search(s):
        found.append("prompt_injection")
    if _CODE_RE.search(s):
        found.append("code_injection")
    if _TOXIC_RE.search(s):
        found.append("toxic_or_fair_housing")
    for kind, pat in PII_PATTERNS.items():
        if re.search(pat, s):
            found.append(f"pii_{kind}")
    if len(s) > config.MAX_TEXT_LEN:
        found.append("oversized")
    return found


def scan_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Walk the record's free-text/personalization fields and collect violations."""
    violations: list[dict[str, Any]] = []

    def add(field: str, value: Any, cats: list[str]):
        for c in cats:
            violations.append({"field": field, "value": _coerce_str(value)[:120], "category": c})

    inp = record.get("input", {}) or {}
    profile = inp.get("profile", {}) or {}

    # Structural fields that must never carry instructions (jailbreak surface).
    for field in ("persona", "lifecycle_stage"):
        val = record.get(field)
        if isinstance(val, str):
            if _ROLE_RE.search(val) or _INJECTION_RE.search(val):
                add(field, val, ["role_override"])

    # Free-text personalization fields.
    text_fields = {
        "property_name": inp.get("property_name"),
        "profile.first_name": profile.get("first_name"),
        "profile.city_interest": profile.get("city_interest"),
    }
    for field, val in text_fields.items():
        if val is None:
            continue
        cats = is_unsafe_text(val)
        if cats:
            add(field, val, cats)

    amenities = profile.get("amenity_interest")
    if isinstance(amenities, list):
        for i, item in enumerate(amenities):
            cats = is_unsafe_text(item)
            if cats:
                add(f"profile.amenity_interest[{i}]", item, cats)

    return violations


def classify_severity(violations: list[dict[str, Any]]) -> str:
    """Map violations to a routing severity.

    Injection/markup/toxic/oversized content in *personalization* fields is
    sanitizable — we strip it and still send a safe message. There is no input
    that forces a hard block here; lack of consent (handled in intake) is the
    only thing that stops a send.
    """
    if not violations:
        return "clean"
    return "sanitizable"


def _sanitize_name(value: Any) -> str | None:
    """Return a safe first name, or None if the value is unusable/unsafe."""
    if value is None:
        return None
    if is_unsafe_text(value):
        return None
    name = clean_text(value, max_len=config.MAX_NAME_LEN)
    # A real first name is short and mostly alphabetic.
    if not name or not re.search(r"[A-Za-z]", name) or len(name) > config.MAX_NAME_LEN:
        return None
    return name


def _sanitize_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value[: config.MAX_LIST_ITEMS]:
        if is_unsafe_text(item):
            continue
        cleaned = clean_text(item, max_len=120)
        if cleaned:
            out.append(cleaned)
    return out


def sanitize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Produce a deep-ish copy of the record with all personalization fields cleaned.

    Unsafe values are dropped (not reproduced). Structural override attempts in
    persona/lifecycle_stage are reset to safe neutral values.
    """
    import copy

    rec = copy.deepcopy(record)
    inp = rec.setdefault("input", {})
    profile = inp.setdefault("profile", {})

    # Neutralize jailbreak attempts in structural fields (AC-19).
    persona = rec.get("persona")
    if not isinstance(persona, str) or persona.lower() not in config.KNOWN_PERSONAS:
        if isinstance(persona, str) and _ROLE_RE.search(persona):
            rec["persona"] = "prospect"
        elif isinstance(persona, str):
            rec["persona"] = clean_text(persona, max_len=40) or "prospect"
        else:
            rec["persona"] = "prospect"

    lifecycle = rec.get("lifecycle_stage")
    if isinstance(lifecycle, str) and (_ROLE_RE.search(lifecycle) or _INJECTION_RE.search(lifecycle)):
        rec["lifecycle_stage"] = "unknown"
    elif isinstance(lifecycle, str):
        rec["lifecycle_stage"] = clean_text(lifecycle, max_len=40)

    # Personalization fields.
    safe_name = _sanitize_name(profile.get("first_name"))
    if safe_name:
        profile["first_name"] = safe_name
    else:
        profile.pop("first_name", None)

    if "city_interest" in profile:
        if is_unsafe_text(profile["city_interest"]):
            profile.pop("city_interest", None)
        else:
            profile["city_interest"] = clean_text(profile["city_interest"], max_len=120)

    if "amenity_interest" in profile:
        cleaned = _sanitize_list(profile["amenity_interest"])
        if cleaned:
            profile["amenity_interest"] = cleaned
        else:
            profile.pop("amenity_interest", None)

    if isinstance(inp.get("property_name"), str):
        if is_unsafe_text(inp["property_name"]):
            inp["property_name"] = "our community"
        else:
            inp["property_name"] = clean_text(inp["property_name"], max_len=120)

    return rec


def find_pii(text: str) -> list[str]:
    """Return PII categories present in an output body (AC-09 validation).

    Scans the base PII set plus the output-only extras (street address, card-like and
    IP numbers) that we don't gate inputs on (G6)."""
    found = []
    for kind, pat in {**PII_PATTERNS, **OUTPUT_EXTRA_PII_PATTERNS}.items():
        if re.search(pat, text or ""):
            found.append(kind)
    return found


def has_toxic(text: str) -> bool:
    return bool(_TOXIC_RE.search(text or ""))
