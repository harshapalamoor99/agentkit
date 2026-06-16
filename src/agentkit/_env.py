"""Zero-dependency .env loader.

Loads KEY=VALUE pairs from a `.env` file (project root or CWD) into the environment
*without* overriding variables already set in the real environment — so an explicit
`export` or CI secret always wins over the file. This is what lets the agent pick up
the right gateway model/embedding config (e.g. the fast lite model) automatically,
avoiding the recurring "openai/ prefix" and "wrong embedding model" pitfalls.

Intentionally tiny and dependency-free; not a full dotenv implementation.
"""
from __future__ import annotations

import os
from pathlib import Path


def _candidate_paths() -> list[Path]:
    # Project root is two levels up from this file (src/agentkit/_env.py).
    root = Path(__file__).resolve().parents[2]
    seen: list[Path] = []
    for p in (Path.cwd() / ".env", root / ".env"):
        if p not in seen:
            seen.append(p)
    return seen


def _parse_line(line: str) -> tuple[str, str] | None:
    line = line.strip()
    if not line or line.startswith("#") or "=" not in line:
        return None
    key, _, value = line.partition("=")
    key = key.strip()
    if key.startswith("export "):
        key = key[len("export "):].strip()
    if not key:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    return key, value


def load_dotenv() -> None:
    """Best-effort: populate os.environ from the first .env found. Never raises."""
    for path in _candidate_paths():
        try:
            if not path.is_file():
                continue
            for raw in path.read_text(encoding="utf-8").splitlines():
                parsed = _parse_line(raw)
                if parsed is None:
                    continue
                key, value = parsed
                os.environ.setdefault(key, value)  # real env wins
            return
        except OSError:
            continue
