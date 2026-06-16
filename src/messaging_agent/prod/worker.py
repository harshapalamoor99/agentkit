"""Kafka worker — the horizontal scaling unit.

Consumes `messaging.requests`, runs each record through the production Pipeline, and
publishes the result to `messaging.results`. Each worker pod runs one of these; scale
by partition count / consumer-group members.

Runs in two modes (auto-selected):
  * **kafka**  — if `kafka-python` is installed and `KAFKA_BOOTSTRAP_SERVERS` is set.
  * **stdin**  — reads one JSON record per line from stdin (great for local/k8s probes
                 and for piping a file in).
This makes the worker runnable with or without a Kafka cluster, with identical logic.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

from .logging_config import get_logger, log_event
from .runner import Pipeline

log = get_logger()

REQUESTS_TOPIC = os.getenv("KAFKA_REQUESTS_TOPIC", "messaging.requests")
RESULTS_TOPIC = os.getenv("KAFKA_RESULTS_TOPIC", "messaging.results")
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "messaging-agent-workers")


def _kafka_available() -> bool:
    if not os.getenv("KAFKA_BOOTSTRAP_SERVERS"):
        return False
    try:
        import kafka  # noqa: F401
        return True
    except ImportError:
        return False


async def _run_stdin(pipe: Pipeline) -> int:
    log_event(log, "worker_start", mode="stdin")
    count = 0
    loop = asyncio.get_event_loop()
    while True:
        line = await loop.run_in_executor(None, sys.stdin.readline)
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            record = {"_raw_line": line}
        result = await pipe.process_one(record)
        sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
        sys.stdout.flush()
        count += 1
    log_event(log, "worker_stop", mode="stdin", processed=count)
    return 0


async def _run_kafka(pipe: Pipeline) -> int:
    from kafka import KafkaConsumer, KafkaProducer  # lazy

    servers = os.environ["KAFKA_BOOTSTRAP_SERVERS"].split(",")
    consumer = KafkaConsumer(
        REQUESTS_TOPIC, group_id=GROUP_ID, bootstrap_servers=servers,
        auto_offset_reset="earliest", enable_auto_commit=False,
        value_deserializer=lambda v: v.decode("utf-8"),
    )
    producer = KafkaProducer(
        bootstrap_servers=servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
    )
    log_event(log, "worker_start", mode="kafka", topic=REQUESTS_TOPIC, group=GROUP_ID)
    try:
        for message in consumer:
            try:
                record = json.loads(message.value)
            except json.JSONDecodeError:
                record = {"_raw_line": message.value}
            result = await pipe.process_one(record)
            producer.send(RESULTS_TOPIC, result)
            producer.flush()
            consumer.commit()  # at-least-once; idempotency store guards duplicates
    finally:
        consumer.close()
        producer.close()
    return 0


def main() -> int:
    pipe = Pipeline(
        concurrency=int(os.getenv("CONCURRENCY", "16")),
        use_cache=os.getenv("USE_CACHE", "1") == "1",
        use_idempotency=os.getenv("USE_IDEMPOTENCY", "1") == "1",
        use_audit=os.getenv("USE_AUDIT", "1") == "1",
    )
    runner = _run_kafka if _kafka_available() else _run_stdin
    return asyncio.run(runner(pipe))


if __name__ == "__main__":
    raise SystemExit(main())
