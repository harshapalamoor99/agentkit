"""CLI runner: process a JSONL file through the messaging-agent graph.

Usage:
    python -m messaging_agent.cli data/evals/sample_8613.jsonl
    python -m messaging_agent.cli data/evals/sample_8613.jsonl --out results.jsonl --verbose
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .graph import app
from .nodes import llm as _llmnode


def load_jsonl(path: str) -> list[dict]:
    records = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Keep the raw line so the agent can handle it as a bad record (AC-17).
            records.append({"_raw_line": line, "_line_no": i})
    return records


async def run_record(record: dict, dataset: list[dict]) -> dict:
    if "_raw_line" in record:
        init = {"raw_line": record["_raw_line"], "dataset": dataset}
    else:
        init = {"record": record, "raw_line": json.dumps(record, ensure_ascii=False),
                "dataset": dataset, "task_id": record.get("task_id", "unknown")}
    result = await app.ainvoke(init)
    return result.get("final_output", {"error": "no output", "state_keys": list(result.keys())})


async def main_async(args) -> int:
    dataset = load_jsonl(args.input)
    client = _llmnode._client  # warm the exact client the graph's llm node uses
    provider = client.provider or "NONE — LLM-only: records will abort (set an API key)"
    print(f"# LLM provider: {provider}", file=sys.stderr)
    if client.available:
        await client.warmup()  # prime the connection so the first record isn't cold

    outputs = []
    for rec in dataset:
        out = await run_record(rec, dataset)
        outputs.append(out)

    if args.out:
        Path(args.out).write_text(
            "\n".join(json.dumps(o, ensure_ascii=False) for o in outputs) + "\n",
            encoding="utf-8")

    total_pass = total = 0
    for out in outputs:
        ev = out.get("evaluation", {})
        p, t = ev.get("passed", 0), ev.get("total", 0)
        total_pass += p
        total += t
        crit = ev.get("critical_fails", [])
        flag = "OK " if not crit else "!! "
        print(f"{flag}{out.get('task_id'):35} send={str(out.get('should_send')):5} "
              f"AC={ev.get('score','?'):6} "
              f"critical_fails={[c['id'] for c in crit]}")
        if args.verbose:
            print(json.dumps(out, ensure_ascii=False, indent=2))

    print(f"\nTOTAL acceptance criteria passed: {total_pass}/{total}")
    any_critical = any(o.get("evaluation", {}).get("critical_fails") for o in outputs)
    return 1 if any_critical else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Context-aware messaging agent")
    parser.add_argument("input", help="Path to JSONL input file")
    parser.add_argument("--out", help="Write per-record JSON output here")
    parser.add_argument("--verbose", action="store_true", help="Print full output per record")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
