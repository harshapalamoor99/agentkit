"""Geo → timezone resolution for TCPA quiet-hours enforcement (AC-4).

The spec requires deriving the recipient's local timezone "based on their provided zip
code or area code" when an explicit IANA timezone is not present. We resolve in priority
order: explicit IANA `timezone` field → US ZIP prefix → phone area code. This is a
coarse, lookup-table mapping (not a geocoding service) — enough to pick the correct US
zone for quiet-hours math, and trivially swappable for a real geo service in prod.
"""
from __future__ import annotations

import re
from typing import Any

from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# US ZIP first-digit / prefix → IANA zone. ZIPs are allocated roughly west-to-east by
# region; we map by leading digits to the dominant zone for that band.
_ZIP_PREFIX_TZ: list[tuple[str, str]] = [
    ("0", "America/New_York"),   # New England
    ("1", "America/New_York"),   # NY/PA
    ("2", "America/New_York"),   # Mid-Atlantic/Southeast
    ("3", "America/New_York"),   # Southeast (FL/GA/TN-east)
    ("4", "America/New_York"),   # OH/IN/MI/KY
    ("5", "America/Chicago"),    # Upper Midwest
    ("6", "America/Chicago"),    # IL/MO/KS/TX-north
    ("7", "America/Chicago"),    # TX/OK/AR/LA
    ("80", "America/Denver"),    # CO
    ("81", "America/Denver"),
    ("82", "America/Denver"),    # WY
    ("83", "America/Boise"),     # ID
    ("84", "America/Denver"),    # UT
    ("85", "America/Phoenix"),   # AZ (no DST)
    ("86", "America/Phoenix"),
    ("87", "America/Denver"),    # NM
    ("88", "America/Chicago"),   # TX-west/NM border
    ("89", "America/Los_Angeles"),  # NV
    # Non-contiguous / non-DST zones must be matched BEFORE the generic "9" band.
    ("967", "Pacific/Honolulu"),    # HI (no DST, UTC-10)
    ("968", "Pacific/Honolulu"),    # HI
    ("995", "America/Anchorage"),   # AK (UTC-9)
    ("996", "America/Anchorage"),   # AK
    ("997", "America/Anchorage"),   # AK
    ("998", "America/Anchorage"),   # AK
    ("999", "America/Anchorage"),   # AK
    ("9", "America/Los_Angeles"),   # CA/OR/WA (dominant Pacific)
]

# Phone area code → IANA zone (subset of common codes; extend via a real DB in prod).
_AREA_CODE_TZ: dict[str, str] = {
    # Eastern
    "212": "America/New_York", "646": "America/New_York", "718": "America/New_York",
    "202": "America/New_York", "404": "America/New_York", "305": "America/New_York",
    "617": "America/New_York", "215": "America/New_York", "412": "America/New_York",
    # Central
    "312": "America/Chicago", "773": "America/Chicago", "214": "America/Chicago",
    "469": "America/Chicago", "972": "America/Chicago", "713": "America/Chicago",
    "512": "America/Chicago", "615": "America/Chicago", "504": "America/Chicago",
    # Mountain
    "303": "America/Denver", "720": "America/Denver", "801": "America/Denver",
    "505": "America/Denver",
    # Arizona (no DST)
    "602": "America/Phoenix", "480": "America/Phoenix", "520": "America/Phoenix",
    # Pacific
    "213": "America/Los_Angeles", "310": "America/Los_Angeles", "415": "America/Los_Angeles",
    "619": "America/Los_Angeles", "206": "America/Los_Angeles", "503": "America/Los_Angeles",
    "702": "America/Los_Angeles",
}


def _safe_zone(name: str) -> ZoneInfo | None:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return None


def tz_from_zip(zip_code: Any) -> str | None:
    if not isinstance(zip_code, (str, int)):
        return None
    digits = re.sub(r"\D", "", str(zip_code))
    if len(digits) < 3:
        return None
    # Prefer a 2-digit prefix match (more specific), then fall back to 1-digit.
    for prefix, zone in sorted(_ZIP_PREFIX_TZ, key=lambda t: -len(t[0])):
        if digits.startswith(prefix):
            return zone
    return None


def tz_from_phone(phone: Any) -> str | None:
    if not isinstance(phone, (str, int)):
        return None
    digits = re.sub(r"\D", "", str(phone))
    if digits.startswith("1") and len(digits) == 11:
        digits = digits[1:]
    if len(digits) < 3:
        return None
    return _AREA_CODE_TZ.get(digits[:3])


def _is_non_us(record: dict[str, Any]) -> bool:
    """True if the record explicitly declares a non-US country. ZIP/area-code zone
    inference is US-specific, so for a declared non-US prospect we must NOT guess a US
    timezone (and therefore must not assume the US 8am-9pm quiet-hours window)."""
    inp = record.get("input", {}) or {}
    profile = inp.get("profile", {}) or {}
    country = (inp.get("country") or profile.get("country")
               or record.get("country") or "")
    if not isinstance(country, str) or not country.strip():
        return False
    c = country.strip().lower()
    us_aliases = {"us", "usa", "u.s.", "u.s.a.", "united states",
                  "united states of america", "america"}
    return c not in us_aliases


def resolve_timezone(record: dict[str, Any]) -> tuple[ZoneInfo | None, str]:
    """Resolve the recipient's zone with a documented source (AC-4).

    Returns (ZoneInfo|None, source) where source is one of:
    'iana_field' | 'zip' | 'area_code' | 'none'.

    An explicit IANA timezone is always honored. ZIP/area-code inference is US-only and
    is skipped when the record declares a non-US country, so we never apply US quiet
    hours to an international prospect on a guess.
    """
    inp = record.get("input", {}) or {}
    profile = inp.get("profile", {}) or {}

    tz_name = inp.get("timezone")
    if isinstance(tz_name, str) and tz_name:
        zone = _safe_zone(tz_name)
        if zone is not None:
            return zone, "iana_field"

    if _is_non_us(record):
        return None, "none"

    for zip_field in (inp.get("zip_code"), inp.get("zip"), profile.get("zip_code"),
                      profile.get("zip")):
        name = tz_from_zip(zip_field)
        if name:
            return _safe_zone(name), "zip"

    for phone_field in (inp.get("phone"), profile.get("phone"), inp.get("phone_number")):
        name = tz_from_phone(phone_field)
        if name:
            return _safe_zone(name), "area_code"

    return None, "none"
