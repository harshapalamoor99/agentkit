"""Inbound-reply intent classifier + F1 evaluation.

The records declare a `reply_classification_f1_min` threshold: the system must be able
to classify a prospect's *reply* (e.g. "STOP", "1", "yes Thursday", "not interested")
so the cadence engine can react. This module provides:

* `classify(text)` — a deterministic, ordered rule classifier over the intents the
  messages actually solicit (opt-out, tour selection, affirm, decline, question).
* `f1(dataset)` — macro-F1 over a labeled fixture, used by the eval harness to check
  the declared threshold. A built-in fixture ships so the capability is always
  measurable; callers can pass their own labeled examples.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

INTENTS = ["opt_out", "tour_select", "affirm", "decline", "question", "other"]

# Ordered rules: first match wins. Opt-out must take precedence (compliance).
_RULES = [
    ("opt_out", re.compile(r"\b(stop|unsubscribe|opt[\s-]?out|remove me|quit|cancel)\b", re.I)),
    ("tour_select", re.compile(r"^\s*(1|2|thu|thur|thursday|fri|friday|option\s*[12])\b", re.I)),
    ("decline", re.compile(r"\b(no|not interested|nope|don'?t|stop sending|leave me|already (leased|signed))\b", re.I)),
    ("affirm", re.compile(r"\b(yes|yeah|yep|sure|sounds good|ok|okay|book|interested|let'?s do)\b", re.I)),
    ("question", re.compile(r"(\?|how much|what time|when|where|price|rent|pet|deposit|available)", re.I)),
]


def classify(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "other"
    for intent, pattern in _RULES:
        if pattern.search(t):
            return intent
    return "other"


# Built-in labeled fixture covering the intents the agent's CTAs elicit.
DEFAULT_FIXTURE: list[dict[str, str]] = [
    {"text": "STOP", "label": "opt_out"},
    {"text": "please unsubscribe me", "label": "opt_out"},
    {"text": "opt out", "label": "opt_out"},
    {"text": "1", "label": "tour_select"},
    {"text": "2", "label": "tour_select"},
    {"text": "Thursday works", "label": "tour_select"},
    {"text": "Fri please", "label": "tour_select"},
    {"text": "yes I'd love to book", "label": "affirm"},
    {"text": "sounds good, interested", "label": "affirm"},
    {"text": "sure let's do it", "label": "affirm"},
    {"text": "no thanks", "label": "decline"},
    {"text": "not interested", "label": "decline"},
    {"text": "we already signed elsewhere", "label": "decline"},
    {"text": "how much is rent?", "label": "question"},
    {"text": "what time are tours available?", "label": "question"},
    {"text": "do you allow pets?", "label": "question"},
    {"text": "lol ok whatever", "label": "affirm"},
    {"text": "asdkjfh", "label": "other"},
    {"text": "🙂", "label": "other"},
]


def f1(dataset: list[dict[str, str]] | None = None) -> dict[str, Any]:
    data = dataset or DEFAULT_FIXTURE
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    correct = 0
    for ex in data:
        pred = classify(ex["text"])
        gold = ex["label"]
        if pred == gold:
            tp[gold] += 1
            correct += 1
        else:
            fp[pred] += 1
            fn[gold] += 1

    per_label = {}
    f1s = []
    labels = {ex["label"] for ex in data}
    for label in labels:
        p = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) else 0.0
        r = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) else 0.0
        lf1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per_label[label] = {"precision": round(p, 3), "recall": round(r, 3), "f1": round(lf1, 3)}
        f1s.append(lf1)

    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    return {
        "macro_f1": round(macro_f1, 3),
        "accuracy": round(correct / len(data), 3),
        "n": len(data),
        "per_label": per_label,
    }
