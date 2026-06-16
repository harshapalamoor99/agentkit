"""LLM-as-judge with named quality dimensions (AC-13).

Scores message-body quality that deterministic checks can't catch, on two named
dimensions that gate the CI/CD pipeline:

  * Faithfulness     — does the copy avoid hallucinating amenities/claims not grounded
                       in the input/expected? (must be >= JUDGE_FAITHFULNESS_MIN, 0.95)
  * Context Precision — did it correctly adapt to the prospect's explicit choices and
                       constraints (interests, channel, horizon)? (>= 0.90)

Plus an overall quality score. Uses the configured LLM if a key is present; otherwise a
transparent heuristic so the harness still produces numbers offline. The judge is an
offline eval gate, never a runtime compliance gate.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from ..llm_client import LLMClient

_client = LLMClient()

_JUDGE_SYSTEM = """\
You are a strict offline QA judge for multifamily leasing outreach copy. You are given
the generated MESSAGE, the INPUT context that produced it, and (optionally) a reference
EXPECTED message. Score three dimensions on a 0.0-1.0 scale:
  - "faithfulness": 1.0 if every concrete claim/amenity in the message is grounded in the
    INPUT or EXPECTED; penalize invented amenities, prices, or facts.
  - "context_precision": 1.0 if the message correctly reflects the prospect's explicit
    interests, chosen channel, and constraints; penalize generic copy that ignores them.
  - "overall": holistic quality (clarity, brand voice, CTA, length, opt-out present).
Respond with STRICT JSON:
{"faithfulness": <float>, "context_precision": <float>, "overall": <float>,
 "issues": ["..."], "rationale": "<one sentence>"}"""


def _heuristic(produced: dict, record: dict | None) -> dict[str, Any]:
    message = produced.get("next_message") or {}
    body = (message.get("body") or "")
    low = body.lower()
    issues: list[str] = []

    # Context precision: are the prospect's interests reflected?
    interests = []
    prof = ((record or {}).get("input", {}) or {}).get("profile", {}) or {}
    if isinstance(prof.get("amenity_interest"), list):
        interests += [str(x).lower() for x in prof["amenity_interest"]]
    if prof.get("city_interest"):
        interests.append(str(prof["city_interest"]).lower())
    if interests:
        hit = any(any(tok in low for tok in i.split()) for i in interests)
        context_precision = 1.0 if hit else 0.6
        if not hit:
            issues.append("interests not reflected")
    else:
        context_precision = 1.0

    # Faithfulness: penalize numbers/prices not present in the input (cheap proxy).
    faithfulness = 1.0
    if re.search(r"\$\s?\d+|\b\d+%\s*off\b", low):
        faithfulness = 0.7
        issues.append("possible invented pricing claim")

    overall = 1.0
    if not re.match(r"\s*hi\b", body, re.I):
        overall -= 0.15
        issues.append("no greeting")
    if not re.search(r"(book|tour|visit|schedule|reply)", low):
        overall -= 0.25
        issues.append("weak CTA")
    if len(body) == 0:
        overall, faithfulness, context_precision = 0.0, 0.0, 0.0

    return {
        "faithfulness": round(max(0.0, faithfulness), 3),
        "context_precision": round(max(0.0, context_precision), 3),
        "overall": round(max(0.0, overall), 3),
        "score": round(max(0.0, overall), 3),
        "issues": issues,
        "rationale": "heuristic score (no LLM judge configured)",
        "judge": "heuristic",
    }


def _parse(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) and "faithfulness" in obj else None
    except (json.JSONDecodeError, ValueError):
        return None


def _clamp(v: Any) -> float:
    try:
        return float(max(0.0, min(1.0, float(v))))
    except (TypeError, ValueError):
        return 0.0


async def judge(produced: dict, record: dict | None = None,
                timeout_s: float = 8.0) -> dict[str, Any]:
    message = produced.get("next_message")
    if not message:
        return {"faithfulness": 1.0, "context_precision": 1.0, "overall": 1.0,
                "score": 1.0, "issues": [], "rationale": "no message (no-send)", "judge": "n/a"}

    if not _client.available:
        return _heuristic(produced, record)

    # Domains may override the judge rubric for their own quality dimensions.
    system = _JUDGE_SYSTEM
    if record is not None:
        try:
            from ..domain import get_domain
            override = get_domain({"validated_record": record}).judge_system_prompt()
            if override:
                system = override
        except Exception:
            pass

    payload = {"message": message}
    if record is not None:
        payload["input"] = {k: record.get(k) for k in ("persona", "lifecycle_stage", "input")}
        if record.get("expected"):
            payload["expected"] = record["expected"].get("next_message")
    prompt = "Evaluate:\n" + json.dumps(payload, ensure_ascii=False)
    try:
        text = await asyncio.wait_for(
            _client.generate(system, prompt, max_tokens=300, temperature=0.0),
            timeout=timeout_s,
        )
        parsed = _parse(text)
        if parsed is None:
            return _heuristic(produced, record)
        faith = _clamp(parsed.get("faithfulness"))
        ctx = _clamp(parsed.get("context_precision"))
        overall = _clamp(parsed.get("overall", (faith + ctx) / 2))
        return {"faithfulness": faith, "context_precision": ctx, "overall": overall,
                "score": overall, "issues": parsed.get("issues", []),
                "rationale": parsed.get("rationale", ""), "judge": "llm"}
    except (asyncio.TimeoutError, Exception):
        return _heuristic(produced, record)


_FAIRHOUSING_SYSTEM = """\
You are a US Fair Housing Act compliance reviewer. Given a leasing OUTREACH MESSAGE,
decide whether it contains discriminatory, steering, or protected-class preference/
exclusion language (race, color, religion, national origin, sex, familial status,
disability). Be precise: ordinary amenity/community copy is NOT a violation.
Respond STRICT JSON: {"violation": <bool>, "confidence": <0.0-1.0>, "reason": "<short>"}"""


async def fairness_gate(body: str, *, client=None, timeout_s: float | None = None,
                        min_confidence: float | None = None) -> dict[str, Any]:
    """Optional runtime fair-housing veto (G6).

    Returns {"violation": bool, "confidence": float, "reason": str}. On no provider,
    timeout, error, or low confidence it returns a non-violating verdict so it can only
    ever ADD safety (force no-send), never wrongly suppress on infrastructure failure.
    """
    from .. import config as _cfg
    c = client if client is not None else _client
    min_conf = _cfg.FAIRHOUSING_JUDGE_MIN_CONFIDENCE if min_confidence is None else min_confidence
    tmo = _cfg.FAIRHOUSING_JUDGE_TIMEOUT_S if timeout_s is None else timeout_s
    clean = {"violation": False, "confidence": 0.0, "reason": ""}
    if not body or c is None or not getattr(c, "available", False):
        return clean
    try:
        text = await asyncio.wait_for(
            c.generate(_FAIRHOUSING_SYSTEM, "MESSAGE:\n" + body, max_tokens=120,
                       temperature=0.0),
            timeout=tmo,
        )
        parsed = _parse_fairness(text)
        if parsed is None:
            return clean
        violation = bool(parsed.get("violation")) and _clamp(parsed.get("confidence")) >= min_conf
        return {"violation": violation, "confidence": _clamp(parsed.get("confidence")),
                "reason": str(parsed.get("reason") or "")}
    except (asyncio.TimeoutError, Exception):
        return clean


def _parse_fairness(text: str) -> dict[str, Any] | None:
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) and "violation" in obj else None
    except (json.JSONDecodeError, ValueError):
        return None
