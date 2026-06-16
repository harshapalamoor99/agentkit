"""Closed-loop telemetry capture (AC-15).

When a downstream consumer action occurs (link clicked, tour booked, STOP received) we
emit a telemetry event that PAIRS the original input features and the generated copy with
the final outcome. These paired records are the verified training data for the next
optimization / fine-tuning cycle.

Transport: Kafka (`KAFKA_BOOTSTRAP_SERVERS` + `KAFKA_TELEMETRY_TOPIC`) when available,
otherwise appended as JSONL to `TELEMETRY_LOG_PATH` (default ./telemetry.jsonl) — same
graceful-degradation pattern as the audit store, so it runs locally and in prod
unchanged.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Outcomes we treat as conversion signals for the paired training record.
OUTCOME_EVENTS = {"link_clicked", "tour_booked", "replied", "stop", "opt_out",
                  "unsubscribe", "no_response"}


def _input_features(record: dict[str, Any]) -> dict[str, Any]:
    """Extract the non-PII feature vector that drove the decision.

    Defaults to the active domain's :meth:`telemetry_features` so closed-loop telemetry
    is domain-aware; falls back to the leasing-style extraction below if no domain is
    resolvable (keeps this module importable in isolation).
    """
    try:
        from .domain import get_domain
        return get_domain({"validated_record": record}).telemetry_features(record)
    except Exception:
        pass
    inp = record.get("input", {}) or {}
    profile = inp.get("profile", {}) or {}
    return {
        "persona": record.get("persona"),
        "lifecycle_stage": record.get("lifecycle_stage"),
        "asset_class": record.get("asset_class") or inp.get("asset_class"),
        "channel_preferences": record.get("channel_preferences"),
        "consent": record.get("consent"),
        "move_date_target": inp.get("move_date_target"),
        "interests": {
            "amenity_interest": profile.get("amenity_interest"),
            "city_interest": profile.get("city_interest"),
        },
    }


class _FileTelemetry:
    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()

    def emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")


class _KafkaTelemetry:
    def __init__(self, servers: str, topic: str):
        from kafka import KafkaProducer  # lazy

        self._topic = topic
        self._producer = KafkaProducer(
            bootstrap_servers=servers.split(","),
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
        )

    def emit(self, event: dict[str, Any]) -> None:
        self._producer.send(self._topic, event)
        self._producer.flush()


class TelemetryStore:
    def __init__(self):
        servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
        topic = os.getenv("KAFKA_TELEMETRY_TOPIC", "messaging.telemetry")
        self.backend = "file"
        self._impl: Any = _FileTelemetry(os.getenv("TELEMETRY_LOG_PATH", "telemetry.jsonl"))
        if servers:
            try:
                self._impl = _KafkaTelemetry(servers, topic)
                self.backend = "kafka"
            except Exception:
                pass

    def emit_outcome(self, *, record: dict[str, Any], produced_output: dict[str, Any],
                     outcome: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build and emit the paired training record (features + copy + outcome)."""
        msg = produced_output.get("next_message") or {}
        event = {
            "type": "closed_loop_outcome",
            "task_id": produced_output.get("task_id") or record.get("task_id"),
            "tenant_id": produced_output.get("tenant_id"),
            "input_features": _input_features(record),
            "generated_copy": {
                "channel": msg.get("channel"),
                "subject": msg.get("subject"),
                "body": msg.get("body"),
                "cta": msg.get("cta"),
            },
            "outcome": (outcome or "").strip().lower(),
            "is_conversion": (outcome or "").strip().lower() in ("link_clicked", "tour_booked", "replied"),
            "lineage": produced_output.get("lineage"),
            "metadata": metadata or {},
            "emitted_at": datetime.now(timezone.utc).isoformat(),
        }
        self._impl.emit(event)
        return event


    def emit_event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Emit a generic telemetry event (e.g. an unjustified send-suppression, G9)."""
        event = {"type": event_type,
                 "emitted_at": datetime.now(timezone.utc).isoformat(),
                 **payload}
        self._impl.emit(event)
        return event


# Process-level store.
store = TelemetryStore()
