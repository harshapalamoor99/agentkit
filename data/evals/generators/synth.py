"""Synthetic eval-set generator.

Produces labeled records in the sample_8613.jsonl schema. The generator embeds a
set of *consistent latent rules*; the agent never sees them — it must infer them
from the data. This is what makes the eval meaningful: if the rules are learnable,
a data-driven agent recovers them; a broken agent won't.

LATENT RULES (encoded here, learned by the agent):
  channel : highest-ranked channel_preference that ALSO has consent.
            If no preferred channel is consented -> do NOT send (suppress).
  timing  : next business-ish day, per-channel hour (sms 9, email 10, voice 11),
            inside quiet hours (never 21:00-08:00 local).
  action  : new        -> start_cadence  <persona>_welcome_{short|long}_horizon
            renewal    -> start_cadence  renewal_offer
            dormant    -> follow_up_in_days value=7
            otherwise  -> follow_up_in_days value=2 (short) / 3 (long)
            suppressed -> hold (reason no_consented_channel)
  horizon : <= 45 days to move = short, else long.
  message : persona/lifecycle templated, personalised with first_name + property,
            opt-out instructions always included; subject null for sms.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone

CHANNELS = ["sms", "email", "voice"]

PROPERTIES = [
    ("Oak Ridge Apartments", "Oak Ridge", "https://oakridge.example/tour", "Richardson, TX", "America/Chicago", "-06:00"),
    ("Maple Court", "Maple Court", "https://maplecourt.example/tour", "Austin, TX", "America/Chicago", "-06:00"),
    ("Bayview Lofts", "Bayview", "https://bayview.example/tour", "San Diego, CA", "America/Los_Angeles", "-08:00"),
    ("Cedar Park Villas", "Cedar Park", "https://cedarpark.example/tour", "Denver, CO", "America/Denver", "-07:00"),
    ("Harborline Residences", "Harborline", "https://harborline.example/tour", "Boston, MA", "America/New_York", "-05:00"),
    ("Sunset Terrace", "Sunset Terrace", "https://sunset.example/tour", "Phoenix, AZ", "America/Phoenix", "-07:00"),
]

NAMES = ["Taylor", "Jordan", "Morgan", "Casey", "Riley", "Avery", "Quinn", "Sam",
         "Jamie", "Drew", "Reese", "Skyler", "Cameron", "Devon", "Harper", "Rowan"]

AMENITIES = [["pool", "fitness"], ["parking", "pet park"], ["rooftop", "lounge"],
             ["business center", "fitness"], ["pool", "spa"], ["fitness", "co-working"]]

PERSONA_LIFECYCLES = [
    ("prospect", "new"),
    ("prospect", "open"),
    ("prospect", "engaged"),
    ("applicant", "open"),
    ("resident", "engaged"),
    ("resident", "renewal"),
    ("resident", "dormant"),
]

HOUR_BY_CHANNEL = {"sms": 9, "email": 10, "voice": 11}


def _consent_for(prefs: list[str], rng: random.Random) -> dict[str, bool]:
    """Generate a consent map producing a varied mix of send / gate / suppress cases."""
    roll = rng.random()
    consent = {f"{c}_opt_in": False for c in CHANNELS}
    if roll < 0.15:
        # suppress: nothing consented
        return consent
    if roll < 0.45:
        # gate: top preference NOT consented, a lower one is
        for c in prefs[1:]:
            consent[f"{c}_opt_in"] = rng.random() < 0.8
        # ensure at least one lower pref consented
        if not any(consent[f"{c}_opt_in"] for c in prefs[1:]) and len(prefs) > 1:
            consent[f"{prefs[1]}_opt_in"] = True
        return consent
    # normal: top preference consented (plus maybe others)
    consent[f"{prefs[0]}_opt_in"] = True
    for c in prefs[1:]:
        consent[f"{c}_opt_in"] = rng.random() < 0.5
    return consent


def _chosen_channel(prefs: list[str], consent: dict[str, bool]) -> str | None:
    for c in prefs:
        if consent.get(f"{c}_opt_in"):
            return c
    return None


def _next_action(persona: str, lifecycle: str, short: bool, should_send: bool) -> dict:
    if not should_send:
        return {"type": "no_action", "reason": "no_consented_channel"}
    if lifecycle == "new":
        return {"type": "start_cadence", "name": f"{persona}_welcome_{'short' if short else 'long'}_horizon"}
    if lifecycle == "renewal":
        return {"type": "start_cadence", "name": "renewal_offer"}
    if lifecycle == "dormant":
        return {"type": "follow_up_in_days", "value": 7}
    return {"type": "follow_up_in_days", "value": 2 if short else 3}


def _send_at(last: datetime, channel: str, offset: str) -> str:
    hour = HOUR_BY_CHANNEL[channel]
    day = (last + timedelta(days=1)).date()
    return f"{day.isoformat()}T{hour:02d}:00:00{offset}"


def _message(persona, lifecycle, channel, name, short_prop, link, amenities, short, primary_cta):
    amen = " & ".join(amenities)
    if persona == "resident" and lifecycle == "renewal":
        subject = None if channel == "sms" else f"Your {short_prop} renewal options"
        if channel == "sms":
            body = (f"Hi {name}—your lease at {short_prop} is up for renewal. "
                    f"Reply 1 to see your renewal offer or 2 to chat. Reply STOP to opt out.")
        else:
            body = (f"Hi {name},\nYour lease at {short_prop} is coming up for renewal. "
                    f"Here are your options and current resident perks ({amen}).\n"
                    f"Review offer → {link}\nTo opt out, reply STOP or click unsubscribe.")
        cta = {"type": "renew_lease", "link": link} if channel != "sms" else {"type": "renew_lease", "options": ["1", "2"]}
        return subject, body, cta
    # prospect / applicant / engaged resident → tour/visit
    horizon_phrase = "this week" if short else "in the next few weeks"
    if channel == "sms":
        subject = None
        body = (f"Hi {name}—welcome to {short_prop}! Tours are available {horizon_phrase}. "
                f"Want to book Thursday or Friday? Reply 1 for Thu, 2 for Fri. Reply STOP to opt out.")
        cta = {"type": "schedule_tour", "options": ["Thu", "Fri"]}
    elif channel == "voice":
        subject = None
        body = (f"Call script: Greet {name}, mention {short_prop} and their interest in {amen}. "
                f"Offer to schedule a tour {horizon_phrase}. Note opt-out: caller may say STOP/do-not-call.")
        cta = {"type": "schedule_tour", "options": ["callback"]}
    else:
        subject = f"Tour {short_prop}—see the {amen} you asked about"
        body = (f"Hi {name},\nThanks for your interest in {short_prop}. Here's a quick look at our "
                f"{amen}. Book a visit {horizon_phrase} to compare floor plans.\n"
                f"Book now → {link}\nTo opt out of emails, click here or reply STOP.")
        cta = {"type": "schedule_tour", "link": link}
    return subject, body, cta


def generate(n: int = 150, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    base = datetime(2025, 12, 6, 12, 0, tzinfo=timezone.utc)
    records: list[dict] = []
    for i in range(n):
        persona, lifecycle = rng.choice(PERSONA_LIFECYCLES)
        prop_name, short_prop, link, city, tz, offset = rng.choice(PROPERTIES)
        name = rng.choice(NAMES)
        amenities = rng.choice(AMENITIES)
        prefs = rng.sample(CHANNELS, k=rng.choice([2, 3]))
        consent = _consent_for(prefs, rng)
        channel = _chosen_channel(prefs, consent)
        should_send = channel is not None

        last = base - timedelta(days=rng.randint(0, 6), hours=rng.randint(0, 12))
        horizon_days = rng.choice([10, 20, 30, 45, 60, 90, 120])
        short = horizon_days <= 45
        move = (last + timedelta(days=horizon_days)).date().isoformat()

        next_action = _next_action(persona, lifecycle, short, should_send)
        primary_cta = "renew_lease" if (persona == "resident" and lifecycle == "renewal") else "book_tour"

        if should_send:
            subject, body, cta = _message(persona, lifecycle, channel, name, short_prop,
                                           link, amenities, short, primary_cta)
            next_message = {
                "channel": channel,
                "send_at": _send_at(last, channel, offset),
                "subject": subject,
                "body": body,
                "cta": cta,
            }
        else:
            next_message = None

        profile: dict = {"first_name": name, "city_interest": city}
        if rng.random() < 0.6:
            profile["amenity_interest"] = amenities

        rec = {
            "task_id": f"syn_{i:04d}_{persona}_{lifecycle}",
            "persona": persona,
            "lifecycle_stage": lifecycle,
            "consent": consent,
            "channel_preferences": prefs,
            "input": {
                "property_name": prop_name,
                "move_date_target": move,
                "last_interaction": last.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "timezone": tz,
                "language": "en",
                "profile": profile,
            },
            "assertions": {
                "required_states": ["consent_verified", "fair_housing_check_passed", "brand_style_applied"],
                "constraints": {
                    "no_pii_leak": True,
                    "no_sensitive_discrimination": True,
                    "include_opt_out_instructions": True,
                    "primary_cta": primary_cta,
                },
            },
            "thresholds": {
                "p95_latency_ms": 2000,
                "personalization_score_min": 0.5,
                "reply_classification_f1_min": 0.9,
                "safety_violations_max": 0,
            },
            "expected": {"next_message": next_message, "next_action": next_action},
        }
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Hard / wide-coverage golden set
# ---------------------------------------------------------------------------
# Extra dimensions layered on top of the base latent rules to stress the agent:
#   - heavier suppress + channel-fallback (gate) consent mixes
#   - multi-tenant isolation (brand-scoped retrieval)
#   - affordable-housing asset class (pricing-incentive suppression)
#   - timezone derived from ZIP (confident) and fully-unknown tz (conservative
#     noon send window) in addition to explicit IANA timezones
# Ground-truth labels are produced by the SAME deterministic latent rules as
# `generate`, so the set is reproducible and drift-guarded.
TENANTS = ["greystar", "camden", "avalonbay"]

# A representative ZIP whose ZIP3 prefix resolves (via geo.py) to each property tz.
ZIP_BY_TZ = {
    "America/Chicago": "75080",
    "America/Los_Angeles": "92101",
    "America/Denver": "80202",
    "America/New_York": "02108",
    "America/Phoenix": "85001",
}


def _consent_hard(prefs: list[str], rng: random.Random) -> dict[str, bool]:
    """Consent map biased toward the hard cases: ~25% suppress, ~40% channel-fallback."""
    roll = rng.random()
    consent = {f"{c}_opt_in": False for c in CHANNELS}
    if roll < 0.25:
        # suppress: nothing consented
        return consent
    if roll < 0.65:
        # gate / channel-fallback: top preference NOT consented, a lower one is
        for c in prefs[1:]:
            consent[f"{c}_opt_in"] = rng.random() < 0.85
        if not any(consent[f"{c}_opt_in"] for c in prefs[1:]) and len(prefs) > 1:
            consent[f"{prefs[1]}_opt_in"] = True
        return consent
    # normal: top preference consented (plus maybe others)
    consent[f"{prefs[0]}_opt_in"] = True
    for c in prefs[1:]:
        consent[f"{c}_opt_in"] = rng.random() < 0.5
    return consent


def generate_hard(n: int = 160, seed: int = 23) -> list[dict]:
    """Generate a hard, wide-coverage golden set spanning consent-fallback,
    suppression, multi-tenant isolation, affordable housing, and derived /
    unknown timezones — every record labeled by the deterministic latent rules.
    """
    rng = random.Random(seed)
    base = datetime(2025, 12, 6, 12, 0, tzinfo=timezone.utc)
    records: list[dict] = []
    for i in range(n):
        persona, lifecycle = rng.choice(PERSONA_LIFECYCLES)
        prop_name, short_prop, link, city, tz, offset = rng.choice(PROPERTIES)
        name = rng.choice(NAMES)
        amenities = rng.choice(AMENITIES)
        prefs = rng.sample(CHANNELS, k=rng.choice([2, 3]))
        consent = _consent_hard(prefs, rng)
        channel = _chosen_channel(prefs, consent)
        should_send = channel is not None

        last = base - timedelta(days=rng.randint(0, 6), hours=rng.randint(0, 12))
        horizon_days = rng.choice([7, 14, 21, 30, 45, 60, 75, 90, 120, 150])
        short = horizon_days <= 45
        move = (last + timedelta(days=horizon_days)).date().isoformat()

        next_action = _next_action(persona, lifecycle, short, should_send)
        primary_cta = "renew_lease" if (persona == "resident" and lifecycle == "renewal") else "book_tour"

        asset_class = "affordable" if rng.random() < 0.25 else "market_rate"
        tenant_id = rng.choice(TENANTS) if rng.random() < 0.35 else None

        input_block: dict = {
            "property_name": prop_name,
            "move_date_target": move,
            "last_interaction": last.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "language": "en",
        }
        send_offset = offset
        send_hour_override: int | None = None
        tz_mode = rng.random()
        if tz_mode < 0.30:
            # tz derived from ZIP (confident) — omit explicit timezone field
            input_block["zip_code"] = ZIP_BY_TZ[tz]
        elif tz_mode < 0.40:
            # tz fully unknown — conservative noon window in America/New_York (winter)
            send_offset = "-05:00"
            send_hour_override = 12
        else:
            input_block["timezone"] = tz

        if should_send:
            subject, body, cta = _message(persona, lifecycle, channel, name, short_prop,
                                           link, amenities, short, primary_cta)
            if send_hour_override is not None:
                day = (last + timedelta(days=1)).date()
                send_at = f"{day.isoformat()}T{send_hour_override:02d}:00:00{send_offset}"
            else:
                send_at = _send_at(last, channel, send_offset)
            next_message = {
                "channel": channel,
                "send_at": send_at,
                "subject": subject,
                "body": body,
                "cta": cta,
            }
        else:
            next_message = None

        profile: dict = {"first_name": name, "city_interest": city}
        if rng.random() < 0.6:
            profile["amenity_interest"] = amenities
        input_block["profile"] = profile

        rec: dict = {
            "task_id": f"hard_{i:04d}_{persona}_{lifecycle}",
            "persona": persona,
            "lifecycle_stage": lifecycle,
            "consent": consent,
            "channel_preferences": prefs,
            "asset_class": asset_class,
            "input": input_block,
            "assertions": {
                "required_states": ["consent_verified", "fair_housing_check_passed", "brand_style_applied"],
                "constraints": {
                    "no_pii_leak": True,
                    "no_sensitive_discrimination": True,
                    "include_opt_out_instructions": True,
                    "primary_cta": primary_cta,
                },
            },
            "thresholds": {
                "p95_latency_ms": 2000,
                "personalization_score_min": 0.5,
                "reply_classification_f1_min": 0.9,
                "safety_violations_max": 0,
            },
            "expected": {"next_message": next_message, "next_action": next_action},
        }
        if tenant_id:
            rec["tenant_id"] = tenant_id
        records.append(rec)
    return records


def write_jsonl(records: list[dict], path: str) -> None:
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def split(records: list, test_frac: float = 0.2, seed: int = 11) -> tuple[list, list]:
    """Deterministic train/test split (order-stable). Works on dicts or Records."""
    idx = list(range(len(records)))
    random.Random(seed).shuffle(idx)
    cut = int(len(records) * (1 - test_frac))
    train_idx = sorted(idx[:cut])
    test_idx = sorted(idx[cut:])
    return [records[i] for i in train_idx], [records[i] for i in test_idx]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="generate synthetic eval set")
    p.add_argument("-n", type=int, default=150)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="comms_agent/eval_data/synthetic.jsonl")
    a = p.parse_args()
    recs = generate(a.n, a.seed)
    write_jsonl(recs, a.out)
    sent = sum(1 for r in recs if r["expected"]["next_message"])
    print(f"wrote {len(recs)} records to {a.out} ({sent} send / {len(recs) - sent} suppress)")
