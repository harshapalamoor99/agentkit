"""Tests for the evaluation harness layer."""
import asyncio
import json
import os

for _k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AZURE_OPENAI_API_KEY",
           "GOOGLE_API_KEY", "GEMINI_API_KEY"):
    os.environ.pop(_k, None)

from messaging_agent.evals import personalization, reply_classifier, semantic  # noqa: E402
from messaging_agent.evals.harness import run, _percentile  # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "evals")


def load(name):
    return [json.loads(l) for l in open(os.path.join(DATA, name), encoding="utf-8") if l.strip()]


def test_semantic_perfect_match():
    expected = {"next_message": {"channel": "sms", "subject": None, "body": "Hi Taylor book a tour reply STOP",
                                 "cta": {"type": "schedule_tour"}},
                "next_action": {"type": "start_cadence"}}
    s = semantic.score(expected, expected)
    assert s["overall"] == 1.0


def test_semantic_channel_mismatch_penalized():
    produced = {"next_message": {"channel": "email", "body": "x", "cta": {"type": "schedule_tour"}},
                "next_action": {"type": "start_cadence"}}
    expected = {"next_message": {"channel": "sms", "body": "x", "cta": {"type": "schedule_tour"}},
                "next_action": {"type": "start_cadence"}}
    s = semantic.score(produced, expected)
    assert s["fields"]["channel"] == 0.0
    assert s["overall"] < 1.0


def test_reply_classifier_f1_meets_threshold():
    assert reply_classifier.f1()["macro_f1"] >= 0.9


def test_reply_classifier_optout_priority():
    assert reply_classifier.classify("yes but STOP") == "opt_out"
    assert reply_classifier.classify("1") == "tour_select"
    assert reply_classifier.classify("not interested") == "decline"


def test_personalization_full_when_signals_used():
    produced = {"next_message": {"subject": None, "body": "Hi Taylor, see the pool at Oak Ridge"}}
    rec = {"input": {"property_name": "Oak Ridge", "profile": {"first_name": "Taylor", "amenity_interest": ["pool"]}}}
    assert personalization.score(produced, rec)["overall"] == 1.0


def test_percentile():
    assert _percentile([1, 2, 3, 4], 0.5) == 2.5


def test_harness_sample_passes():
    report = asyncio.run(run(load("sample_8613.jsonl")))
    s = report["summary"]
    assert s["passed"] is True
    assert not s["critical_fail_records"]
    assert s["mean_semantic_match"] >= 0.8
    assert s["latency_ms"]["p95"] < 2000


def test_harness_adversarial_no_breaches():
    report = asyncio.run(run(load("adversarial.jsonl")))
    assert report["summary"]["passed"] is True
    assert not report["summary"]["threshold_breaches"]
