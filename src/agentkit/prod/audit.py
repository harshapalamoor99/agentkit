"""Audit store: durable record of every decision for compliance/replay.

Writes to Postgres when `DATABASE_URL` is set; otherwise appends JSONL to
`AUDIT_LOG_PATH` (default ./audit.jsonl). Fair-housing/consent decisions must be
auditable, so this captures the full output plus the AC evaluation per task.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_DDL = """
CREATE TABLE IF NOT EXISTS message_audit (
    id            BIGSERIAL PRIMARY KEY,
    task_id       TEXT,
    should_send   BOOLEAN,
    channel       TEXT,
    ac_score      TEXT,
    critical_fail BOOLEAN,
    used_fallback BOOLEAN,
    output        JSONB,
    created_at    TIMESTAMPTZ DEFAULT now()
);
"""


class _FileAudit:
    def __init__(self, path: str):
        self.path = Path(path)
        self._lock = threading.Lock()

    def write(self, row: dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


class _PostgresAudit:
    def __init__(self, url: str):
        import psycopg  # lazy

        self._conn = psycopg.connect(url, autocommit=True)
        with self._conn.cursor() as cur:
            cur.execute(_DDL)

    def write(self, row: dict[str, Any]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                "INSERT INTO message_audit "
                "(task_id, should_send, channel, ac_score, critical_fail, used_fallback, output) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (row["task_id"], row["should_send"], row["channel"], row["ac_score"],
                 row["critical_fail"], row["used_fallback"], json.dumps(row["output"])),
            )


class AuditStore:
    def __init__(self):
        url = os.getenv("DATABASE_URL")
        self.backend = "file"
        self._impl: Any = _FileAudit(os.getenv("AUDIT_LOG_PATH", "audit.jsonl"))
        if url:
            try:
                self._impl = _PostgresAudit(url)
                self.backend = "postgres"
            except Exception:
                self._impl = _FileAudit(os.getenv("AUDIT_LOG_PATH", "audit.jsonl"))

    def record(self, final_output: dict[str, Any]) -> None:
        ev = final_output.get("evaluation", {}) or {}
        msg = final_output.get("next_message") or {}
        row = {
            "task_id": final_output.get("task_id"),
            "should_send": bool(final_output.get("should_send")),
            "channel": msg.get("channel"),
            "ac_score": ev.get("score"),
            "critical_fail": bool(ev.get("critical_fails")),
            "used_fallback": bool(final_output.get("used_fallback")),
            "output": final_output,
            "audited_at": datetime.now(timezone.utc).isoformat(),
        }
        self._impl.write(row)
