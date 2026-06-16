"""Semantic match scoring against the `expected` ground truth.

The core task asks for output that *semantically matches* the expected result.
We score this on two levels:

* **Structural fields** (channel, next_action.type, cta.type) — exact match, since
  these are categorical decisions.
* **Free text** (subject, body) — semantic similarity. If an embedding-capable LLM
  provider is configured this can be swapped for cosine similarity; by default we use
  a deterministic lexical proxy (token Jaccard + key-content overlap) so the harness
  runs offline and reproducibly.

Returns a 0..1 score plus a per-field breakdown for debuggability.
"""
from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "to", "of", "and", "or", "for", "you", "your", "is", "are",
         "this", "we", "our", "on", "in", "at", "with", "i", "it", "now", "—", "-"}


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOP}


def text_similarity(a: str, b: str) -> float:
    """Jaccard token overlap — a deterministic semantic proxy in [0, 1]."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _cosine(u: list[float], v: list[float]) -> float:
    if not u or not v or len(u) != len(v):
        return 0.0
    import math
    dot = sum(x * y for x, y in zip(u, v))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    if nu == 0 or nv == 0:
        return 0.0
    # Cosine is in [-1,1]; clamp to [0,1] for a similarity score.
    return max(0.0, min(1.0, dot / (nu * nv)))


async def embedding_similarity(a: str, b: str, client) -> float | None:
    """Cosine similarity of provider embeddings, or None if unavailable (G2)."""
    if client is None or not getattr(client, "available", False):
        return None
    embed = getattr(client, "embed", None)
    if embed is None:
        return None
    vecs = await embed([a or "", b or ""])
    if not vecs or len(vecs) != 2:
        return None
    return round(_cosine(vecs[0], vecs[1]), 3)


def _get(msg: dict | None, *path, default=None):
    cur: Any = msg or {}
    for p in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(p, default)
    return cur


def score(produced: dict, expected: dict) -> dict[str, Any]:
    """Compare produced output against an `expected` block (lexical similarity).

    `produced` and `expected` both look like {next_message:{...}, next_action:{...}}.
    """
    return _score_with(produced, expected, text_similarity, method="lexical")


async def score_async(produced: dict, expected: dict, client=None) -> dict[str, Any]:
    """Embedding-backed semantic match when a provider is available; otherwise the
    deterministic lexical proxy. The returned `method` field records which was used so
    headline numbers are never silently conflated (G2)."""
    body_p = (produced.get("next_message") or {}).get("body") or ""
    body_e = (expected.get("next_message") or {}).get("body") or ""
    emb = await embedding_similarity(body_p, body_e, client)
    if emb is None:
        return _score_with(produced, expected, text_similarity, method="lexical")

    cache: dict[tuple[str, str], float] = {}

    async def _sim_pairs(pairs):
        for a, b in pairs:
            v = await embedding_similarity(a, b, client)
            cache[(a, b)] = text_similarity(a, b) if v is None else v

    def sim(a: str, b: str) -> float:
        key = (a or "", b or "")
        if key in cache:
            return cache[key]
        return text_similarity(a, b)

    # Pre-compute embedding sims for the free-text fields.
    pm, em = produced.get("next_message"), expected.get("next_message")
    if pm is not None and em is not None:
        pairs = [(pm.get("body") or "", em.get("body") or "")]
        if em.get("subject"):
            pairs.append((pm.get("subject") or "", em.get("subject") or ""))
        await _sim_pairs(pairs)
    return _score_with(produced, expected, sim, method="embedding")


def _score_with(produced: dict, expected: dict, sim, method: str) -> dict[str, Any]:
    pm = produced.get("next_message")
    em = expected.get("next_message")

    fields: dict[str, Any] = {}

    # If ground truth expected a message but none was produced (or vice versa).
    if em is None:
        fields["message_presence"] = 1.0 if pm is None else 0.0
    else:
        fields["message_presence"] = 1.0 if pm is not None else 0.0

    if em is not None and pm is not None:
        fields["channel"] = 1.0 if pm.get("channel") == em.get("channel") else 0.0
        fields["cta_type"] = 1.0 if _get(pm, "cta", "type") == _get(em, "cta", "type") else 0.0
        # Subject: only scored when email/expected has a subject.
        if em.get("subject"):
            fields["subject_sim"] = sim(pm.get("subject") or "", em.get("subject") or "")
        fields["body_sim"] = sim(pm.get("body") or "", em.get("body") or "")

    # next_action.
    pa, ea = produced.get("next_action") or {}, expected.get("next_action") or {}
    fields["next_action_type"] = 1.0 if pa.get("type") == ea.get("type") else 0.0
    if ea.get("type") == "follow_up_in_days":
        fields["next_action_value"] = 1.0 if pa.get("value") == ea.get("value") else 0.0

    overall = sum(fields.values()) / len(fields) if fields else 0.0
    return {"overall": round(overall, 3), "method": method,
            "fields": {k: round(v, 3) for k, v in fields.items()}}
