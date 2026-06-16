"""Stateful cadence workflow + message cancellation (AC-6).

Models the part of the orchestration engine (Temporal-style) that the take-home omits:
scheduled follow-up messages can be CANCELLED or REWRITTEN when a real-time event makes
them obsolete. e.g. a `prospect_long_horizon_day3` follow-up is queued, then the prospect
books a tour via the Day-0 CTA — the workflow transitions to `tour_scheduled` and the
pending Day-3 message is auto-cancelled (or updated to reference the booked tour) instead
of asking them to book again.

In production this is Temporal/Cadence with durable timers; here it is an in-process
state machine + scheduled-message store with the same semantics, so the behavior is
testable and swappable for a real workflow engine.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# Workflow states for a prospect's outreach lifecycle.
NEW = "new"
ENGAGED = "engaged"
TOUR_SCHEDULED = "tour_scheduled"
OPTED_OUT = "opted_out"

# Event -> resulting state.
_EVENT_STATE = {
    "tour_booked": TOUR_SCHEDULED,
    "tour_scheduled": TOUR_SCHEDULED,
    "link_clicked": ENGAGED,
    "replied": ENGAGED,
    "stop": OPTED_OUT,
    "opt_out": OPTED_OUT,
    "unsubscribe": OPTED_OUT,
}


@dataclass
class ScheduledMessage:
    message_id: str
    prospect_id: str
    send_at: str
    payload: dict[str, Any]
    status: str = "scheduled"  # scheduled | cancelled | updated | sent
    note: str = ""


@dataclass
class WorkflowState:
    prospect_id: str
    state: str = NEW
    history: list[dict[str, Any]] = field(default_factory=list)


class WorkflowEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._states: dict[str, WorkflowState] = {}
        self._scheduled: dict[str, list[ScheduledMessage]] = {}

    # --- scheduling ---
    def schedule(self, msg: ScheduledMessage) -> ScheduledMessage:
        with self._lock:
            self._scheduled.setdefault(msg.prospect_id, []).append(msg)
            self._states.setdefault(msg.prospect_id, WorkflowState(msg.prospect_id))
        return msg

    def pending(self, prospect_id: str) -> list[ScheduledMessage]:
        return [m for m in self._scheduled.get(prospect_id, []) if m.status == "scheduled"]

    def state_of(self, prospect_id: str) -> str:
        st = self._states.get(prospect_id)
        return st.state if st else NEW

    # --- event handling ---
    def handle_event(self, prospect_id: str, event_type: str,
                     context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Apply a real-time consumer event: transition state and cancel/rewrite any
        now-obsolete pending messages. Returns a summary of what changed."""
        event_type = (event_type or "").strip().lower()
        new_state = _EVENT_STATE.get(event_type)
        context = context or {}
        cancelled, updated = [], []

        with self._lock:
            st = self._states.setdefault(prospect_id, WorkflowState(prospect_id))
            if new_state:
                st.state = new_state
            st.history.append({
                "event": event_type,
                "to_state": st.state,
                "at": datetime.now(timezone.utc).isoformat(),
            })

            for m in self._scheduled.get(prospect_id, []):
                if m.status != "scheduled":
                    continue
                if new_state == TOUR_SCHEDULED and _is_booking_cta(m.payload):
                    # The follow-up asks them to book — obsolete once they've booked.
                    tour_time = context.get("tour_time")
                    if tour_time:
                        m.status = "updated"
                        m.note = f"rewritten to confirm tour at {tour_time}"
                        m.payload = _confirmation_payload(m.payload, tour_time)
                        updated.append(m.message_id)
                    else:
                        m.status = "cancelled"
                        m.note = "cancelled: prospect already booked a tour"
                        cancelled.append(m.message_id)
                elif new_state == OPTED_OUT:
                    m.status = "cancelled"
                    m.note = "cancelled: prospect opted out"
                    cancelled.append(m.message_id)

        return {
            "prospect_id": prospect_id,
            "event": event_type,
            "state": self.state_of(prospect_id),
            "cancelled_messages": cancelled,
            "updated_messages": updated,
        }


def _is_booking_cta(payload: dict[str, Any]) -> bool:
    cta = (payload.get("next_message") or {}).get("cta") or payload.get("cta") or {}
    return cta.get("type") in ("schedule_tour", "book_tour")


def _confirmation_payload(payload: dict[str, Any], tour_time: str) -> dict[str, Any]:
    p = dict(payload)
    msg = dict(p.get("next_message") or {})
    msg["body"] = (f"Looking forward to seeing you at your tour on {tour_time}! "
                   "Reply if you need to reschedule. Reply STOP to opt out.")
    p["next_message"] = msg
    return p


# Process-level engine (swap for Temporal client in prod).
engine = WorkflowEngine()
