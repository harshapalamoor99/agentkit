"""Tests for the RealPage production-matrix capabilities (RP-01 .. RP-15).

Output-evaluable criteria are checked by running the full pipeline over data/evals/enterprise.jsonl
and scoring with realpage_criteria. Behavioral/infra criteria (state cancellation, circuit
breaker, telemetry, geo timezone, PII tokenization, quiet hours) get focused unit tests.
"""
import asyncio
import json
import os

import pytest

from agentkit import (circuit_breaker, geo, pii, telemetry, tenants,
                              timing, workflow)
from agentkit.graph import app
from agentkit.realpage_criteria import ALL_RP_CRITERIA, evaluate_all

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "evals")


def load(name):
    out = []
    for line in open(os.path.join(DATA_DIR, name), encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def run(record, dataset):
    init = {"record": record, "dataset": dataset,
            "task_id": record.get("task_id", "t"),
            "raw_line": json.dumps(record, ensure_ascii=False)}
    return asyncio.run(app.ainvoke(init))["final_output"]


@pytest.fixture(scope="module")
def enterprise():
    return load("enterprise.jsonl")


# --- Output-evaluable RP criteria over the enterprise dataset ---

def test_realpage_criteria_pass(enterprise):
    for rec in enterprise:
        out = run(rec, enterprise)
        results = evaluate_all(out, rec)
        crit_fails = [r for r in results if not r["pass"] and r["severity"] == "critical"]
        assert not crit_fails, (rec["task_id"], crit_fails)


def test_all_rp_criteria_have_results(enterprise):
    rec = enterprise[0]
    out = run(rec, enterprise)
    results = evaluate_all(out, rec)
    assert {r["id"] for r in results} == {c.id for c in ALL_RP_CRITERIA}
    assert len(results) == 15


# --- RP-04: TCPA quiet-hours window (8am-9pm local) ---

def test_quiet_hours_reschedule_past_window():
    # 23:30 local -> should reschedule to a compliant morning hour.
    late = "2025-12-08T23:30:00-06:00"
    fixed = timing.enforce_quiet_hours(late)
    assert timing.in_quiet_hours_window(fixed)


def test_quiet_hours_window_boundaries():
    assert timing.in_quiet_hours_window("2025-12-09T09:00:00-06:00")
    assert timing.in_quiet_hours_window("2025-12-09T20:59:00-06:00")
    assert not timing.in_quiet_hours_window("2025-12-09T07:30:00-06:00")
    assert not timing.in_quiet_hours_window("2025-12-09T21:30:00-06:00")


# --- AC-4: geo timezone resolution fallback ---

def test_geo_timezone_from_iana_field():
    tz, src = geo.resolve_timezone({"input": {"timezone": "America/Chicago"}})
    assert tz is not None and "America/Chicago" in str(tz)


def test_geo_timezone_from_zip_prefix():
    rec = {"input": {"profile": {"zip": "75080"}}}  # Richardson, TX
    tz, src = geo.resolve_timezone(rec)
    assert tz is not None


# --- RP-08 / AC-8: PII tokenization & output leak detection ---

def test_pii_tokenization_scrubs_raw_fields():
    rec = {"input": {"profile": {"first_name": "Pat", "annual_income": 82000,
                                 "credit_score": 730, "ssn": "123-45-6789"}}}
    out, notes = pii.tokenize_record(rec)
    prof = out["input"]["profile"]
    assert "annual_income" not in prof and "ssn" not in prof and "credit_score" not in prof
    meta = out["input"]["screening_metadata"]
    assert meta.get("income_verified") is True
    assert "credit_tier" in meta and meta.get("ssn_on_file") is True
    assert notes


def test_output_reflects_raw_pii_detects_numbers():
    assert pii.output_reflects_raw_pii("Your income of 82000 qualifies you")
    assert not pii.output_reflects_raw_pii("Tour Oak Ridge in 2026!")


# --- RP-07/RP-10: tenant / portfolio isolation ---

def test_foreign_property_names_excludes_own_tenant():
    rec = {"tenant_id": "oakridge_pm", "input": {"property_name": "Oak Ridge Apartments"}}
    foreign = tenants.foreign_property_names(rec)
    assert "Summit Heights" in foreign
    assert "Oak Ridge Apartments" not in foreign


def test_cross_tenant_property_not_leaked_in_output(enterprise):
    rec = next(r for r in enterprise if r["task_id"] == "crosstenant_bait_day0")
    out = run(rec, enterprise)
    m = out.get("next_message") or {}
    body = (m.get("body") or "").lower()
    for fp in tenants.foreign_property_names(rec):
        assert fp.lower() not in body


# --- RP-09: asset-class routing (regulated => no pricing incentive) ---

def test_regulated_asset_class_no_pricing_incentive(enterprise):
    rec = next(r for r in enterprise if r["task_id"] == "affordable_recert_day0")
    out = run(rec, enterprise)
    assert out.get("asset_class") in (None, "lihtc") or out["asset_class"] == "lihtc"
    m = out.get("next_message")
    if m and m.get("body"):
        from agentkit.domains.leasing import _PRICING_INCENTIVE_RE
        assert not _PRICING_INCENTIVE_RE.search(m["body"])


# --- RP-06: state mutation / message cancellation ---

def test_workflow_cancels_booking_message_on_tour_booked():
    eng = workflow.WorkflowEngine()
    eng.schedule(workflow.ScheduledMessage(
        message_id="m1", prospect_id="p1", send_at="2026-01-01T10:00:00-06:00",
        payload={"cta": {"type": "schedule_tour"}, "body": "Book a tour!"}))
    result = eng.handle_event("p1", "tour_booked", {})
    assert result["state"] == workflow.TOUR_SCHEDULED
    assert "m1" in result["cancelled_messages"]
    assert eng.pending("p1") == []


def test_workflow_rewrites_message_with_tour_time():
    eng = workflow.WorkflowEngine()
    eng.schedule(workflow.ScheduledMessage(
        message_id="m2", prospect_id="p2", send_at="2026-01-01T10:00:00-06:00",
        payload={"cta": {"type": "schedule_tour"}, "body": "Book a tour!"}))
    result = eng.handle_event("p2", "tour_booked", {"tour_time": "Thu 2pm"})
    assert "m2" in result["updated_messages"]


def test_workflow_cancels_all_on_opt_out():
    eng = workflow.WorkflowEngine()
    eng.schedule(workflow.ScheduledMessage(
        message_id="m3", prospect_id="p3", send_at="2026-01-01T10:00:00-06:00",
        payload={"cta": {"type": "schedule_tour"}, "body": "x"}))
    result = eng.handle_event("p3", "stop", {})
    assert result["state"] == workflow.OPTED_OUT
    assert "m3" in result["cancelled_messages"]


# --- RP-11: circuit breaker ---

def test_circuit_breaker_trips_after_threshold():
    cb = circuit_breaker.CircuitBreaker(threshold=3, cooldown_s=60)
    assert cb.allow()
    for _ in range(3):
        cb.record_failure()
    assert cb.state == "open"
    assert not cb.allow()
    cb.record_success()
    assert cb.allow()


# --- RP-12: decision lineage ---

def test_lineage_present_in_output(enterprise):
    rec = enterprise[0]
    out = run(rec, enterprise)
    lin = out.get("lineage") or {}
    assert "prompt_template_version" in lin
    assert "few_shot_example_ids" in lin
    assert "input_snapshot" in lin


# --- RP-15: closed-loop telemetry ---

def test_telemetry_emit_outcome(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEMETRY_LOG_PATH", str(tmp_path / "telemetry.jsonl"))
    store = telemetry.TelemetryStore()
    rec = {"task_id": "t1", "input": {"profile": {"first_name": "Lee"}}}
    out = {"task_id": "t1", "tenant_id": "oakridge_pm",
           "next_message": {"channel": "sms", "body": "Hi Lee", "cta": {"type": "schedule_tour"}},
           "lineage": {"prompt_template_version": "v3"}}
    event = store.emit_outcome(record=rec, produced_output=out,
                               outcome="tour_booked", metadata={"src": "test"})
    assert event["type"] == "closed_loop_outcome"
    assert event["is_conversion"] is True
    assert event["generated_copy"]["channel"] == "sms"
    assert (tmp_path / "telemetry.jsonl").exists()


# --- RP-03: consent enforcement (no message without consent) ---

def test_no_consent_no_message():
    rec = {"task_id": "noc", "consent": {"email_opt_in": False, "sms_opt_in": False,
                                          "voice_opt_in": False},
           "channel_preferences": ["sms", "email"],
           "input": {"property_name": "Oak Ridge Apartments", "timezone": "America/Chicago",
                     "profile": {"first_name": "Dana"}}}
    out = run(rec, [rec])
    assert out["should_send"] is False
    assert not out.get("next_message")
