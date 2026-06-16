"""Personalization scoring (maps to the `personalization_score_min` threshold).

Scores how well a produced message reflects the *available, safe* personalization
signals: the prospect's first name, their stated interests, and the property name.
Only signals that are actually present (and safe, post-sanitization) count toward the
denominator, so a record with no interests isn't penalized for omitting them.
"""
from __future__ import annotations

from typing import Any


def score(produced: dict, sanitized_record: dict) -> dict[str, Any]:
    msg = produced.get("next_message")
    if not msg:
        return {"overall": 0.0, "signals": {}, "note": "no message"}

    text = ((msg.get("subject") or "") + " " + (msg.get("body") or "")).lower()
    inp = (sanitized_record.get("input", {}) or {})
    profile = inp.get("profile", {}) or {}

    signals: dict[str, bool] = {}

    name = profile.get("first_name")
    if name:
        signals["first_name"] = name.lower() in text

    interests: list[str] = []
    if isinstance(profile.get("amenity_interest"), list):
        interests += [str(x) for x in profile["amenity_interest"]]
    if profile.get("city_interest"):
        interests.append(str(profile["city_interest"]))
    if interests:
        signals["interest"] = any(
            any(tok in text for tok in i.lower().split()) for i in interests
        )

    prop = inp.get("property_name")
    if prop:
        signals["property"] = any(tok in text for tok in prop.lower().split() if len(tok) > 3)

    if not signals:
        # Nothing to personalize on (e.g. all fields missing/unsafe). A safe generic
        # message is acceptable; treat as full marks so we don't penalize robustness.
        return {"overall": 1.0, "signals": {}, "note": "no safe signals available"}

    overall = sum(1 for v in signals.values() if v) / len(signals)
    return {"overall": round(overall, 3), "signals": signals}
