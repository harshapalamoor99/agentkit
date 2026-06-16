"""Timezone-aware, business-hours send scheduling.

Implements AC-03/AC-04: produce an ISO-8601 timestamp with an explicit UTC offset
derived from the user's IANA timezone, scheduled inside 9am-6pm local time.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import config
from . import geo


def _resolve_tz(tz_name: Any) -> ZoneInfo | None:
    if not isinstance(tz_name, str) or not tz_name:
        return None
    try:
        return ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        return None


def _parse_dt(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def enforce_quiet_hours(iso_ts: str) -> str:
    """AC-4: ensure a dispatch time falls inside the TCPA window (8am-9pm local).

    If the timestamp is before 8am, move it to 8am the same day; if at/after 9pm, move
    it to 8am the next day. Returns the (possibly adjusted) ISO-8601 timestamp; a naive
    or unparseable input is returned unchanged.
    """
    dt = _parse_dt(iso_ts)
    if dt is None or dt.tzinfo is None:
        return iso_ts
    if dt.hour < config.TCPA_EARLIEST_HOUR:
        dt = dt.replace(hour=config.TCPA_EARLIEST_HOUR, minute=0, second=0, microsecond=0)
    elif dt.hour >= config.TCPA_LATEST_HOUR:
        dt = (dt + timedelta(days=1)).replace(
            hour=config.TCPA_EARLIEST_HOUR, minute=0, second=0, microsecond=0)
    return dt.isoformat()


def in_quiet_hours_window(iso_ts: str) -> bool:
    """True if the dispatch time is within the TCPA-permitted 8am-9pm window (AC-4)."""
    dt = _parse_dt(iso_ts)
    if dt is None or dt.tzinfo is None:
        return False
    return config.TCPA_EARLIEST_HOUR <= dt.hour < config.TCPA_LATEST_HOUR


def compute_send_at(record: dict[str, Any], channel: str, now: datetime | None = None) -> str | None:
    """Compute a tz-aware send_at inside business hours.

    Strategy (data-derived, deterministic): anchor on the last interaction (or now),
    move to the next calendar day, and set a channel-appropriate hour within
    business hours. Returns None if no timezone is known (AC-04 requires an explicit
    offset; without a tz we let the caller decide on a fallback).
    """
    inp = record.get("input", {}) or {}
    tz, _src = geo.resolve_timezone(record)
    if tz is None:
        tz = _resolve_tz(inp.get("timezone"))
    if tz is None:
        return None

    anchor = _parse_dt(inp.get("last_interaction"))
    if anchor is None:
        anchor = now or datetime.now(tz)
    anchor = anchor.astimezone(tz)

    hour = config.DEFAULT_SEND_HOUR.get(channel, config.BUSINESS_HOUR_START)
    hour = max(config.BUSINESS_HOUR_START, min(hour, config.BUSINESS_HOUR_END - 1))

    send_day = (anchor + timedelta(days=1)).date()
    send_dt = datetime(send_day.year, send_day.month, send_day.day, hour, 0, 0, tzinfo=tz)
    # Defensive: business hours sit inside the legal window, but enforce it regardless.
    return enforce_quiet_hours(send_dt.isoformat())


def in_business_hours(iso_ts: str) -> bool:
    dt = _parse_dt(iso_ts)
    if dt is None or dt.tzinfo is None:
        return False
    return config.BUSINESS_HOUR_START <= dt.hour < config.BUSINESS_HOUR_END


def is_tz_aware(iso_ts: str) -> bool:
    dt = _parse_dt(iso_ts)
    return dt is not None and dt.utcoffset() is not None
