"""Data-driven channel selection.

Implements AC-01/AC-02: select the highest-priority channel from
`channel_preferences` for which the corresponding consent flag is true. This is
derived entirely from the record's own data — no persona/content-based rules.
"""
from __future__ import annotations

from typing import Any

from . import config


def consented_channels(record: dict[str, Any]) -> list[str]:
    """All channels the user has explicitly opted into."""
    consent = record.get("consent", {}) or {}
    out = []
    for channel, flag in config.CHANNEL_CONSENT_FLAG.items():
        if consent.get(flag) is True:
            out.append(channel)
    return out


def select_channel(record: dict[str, Any]) -> str | None:
    """Return the chosen channel, or None if the user consented to nothing.

    Priority = order in `channel_preferences`, filtered by consent. If preferences
    are empty/missing, fall back to a stable consent-flag order so we still respect
    consent (AC-02) while degrading gracefully (AC-17).
    """
    allowed = set(consented_channels(record))
    if not allowed:
        return None

    prefs = record.get("channel_preferences") or []
    if isinstance(prefs, list):
        for ch in prefs:
            if ch in allowed:
                return ch

    # Empty/invalid preferences: pick by deterministic consent order.
    for ch in config.CHANNEL_CONSENT_FLAG:
        if ch in allowed:
            return ch
    return None
