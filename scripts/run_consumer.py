"""Consume events from Kafka and write them to PostgreSQL.

This is the second half of the ingestion path: the producer publishes, this
persists. Delivery is at-least-once -- offsets are committed only after the
database transaction succeeds -- so every handler is idempotent and a replay is
harmless. See :mod:`streaming.consumer` for the full reasoning.

Usage::

    python scripts/run_consumer.py                        # run until Ctrl-C
    python scripts/run_consumer.py --max-messages 500     # bounded
    python scripts/run_consumer.py --idle-timeout 15      # exit when quiet
    python scripts/run_consumer.py --topics competitor_prices booking_events
    python scripts/run_consumer.py --from-beginning       # replay the topic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Allow execution as a plain script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from database.connection import wait_for_database  # noqa: E402
from monitoring.logging_config import configure_logging, get_logger  # noqa: E402
from streaming.admin import BrokerUnavailable, wait_for_broker  # noqa: E402
from streaming.consumer import EventConsumer  # noqa: E402
from streaming.topics import TopicName, resolve_topic  # noqa: E402

logger = get_logger(__name__)

#: Topics this consumer persists. Predictions are excluded by default: the API
#: has already written them, and re-consuming would duplicate rows.
DEFAULT_TOPICS = [
    TopicName.COMPETITOR_PRICES,
    TopicName.BOOKING_EVENTS,
    TopicName.DEMAND_EVENTS,
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/run_consumer.py",
        description="Persist Kafka events into PostgreSQL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--topics", type=str, nargs="+",
        default=[t.value for t in DEFAULT_TOPICS],
        choices=[t.value for t in TopicName],
        help="Logical topics to subscribe to.",
    )
    parser.add_argument(
        "--max-messages", type=int, default=None,
        help="Stop after this many messages. Default: run until interrupted.",
    )
    parser.add_argument(
        "--idle-timeout", type=float, default=None,
        help="Stop after this many seconds with no new records.",
    )
    parser.add_argument(
        "--from-beginning", action="store_true",
        help="Read the topics from the start by joining a fresh consumer group. "
        "Useful for replaying; combine with --idle-timeout.",
    )
    parser.add_argument(
        "--group-suffix", type=str, default="",
        help="Appended to the configured consumer group id.",
    )
    parser.add_argument(
        "--strict-validation", action="store_true",
        help="Reject events when the hotels table is empty instead of deferring "
        "to the foreign keys. Recommended in production.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    settings = get_settings()
    configure_logging(settings)
    args = _build_parser().parse_args(argv)

    if not settings.kafka.enabled:
        logger.error("KAFKA_ENABLED=false; there is nothing to consume. Aborting.")
        return 2

    topics = [TopicName(name) for name in args.topics]

    try:
        wait_for_database()
        wait_for_broker(settings)
    except BrokerUnavailable as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.error("Dependency unavailable: %s: %s", type(exc).__name__, exc)
        return 1

    # A fresh group id has no committed offsets, so auto_offset_reset=earliest
    # makes it replay the topic from the start. Deriving the suffix from the
    # message count keeps repeated replays independent of each other.
    suffix = args.group_suffix
    if args.from_beginning:
        suffix = f"{suffix}-replay-{int(__import__('time').time())}"

    from ingestion.validator import EventValidator

    consumer = EventConsumer(
        topics,
        settings=settings,
        validator=EventValidator(strict_when_empty=args.strict_validation),
        group_suffix=suffix,
    )

    logger.info(
        "Consuming %s", ", ".join(resolve_topic(t, settings) for t in topics)
    )

    try:
        stats = consumer.run(
            max_messages=args.max_messages,
            idle_timeout_seconds=args.idle_timeout,
        )
    except Exception as exc:
        logger.error("Consumer failed: %s: %s", type(exc).__name__, exc)
        return 1

    print()
    print("=" * 60)
    print("CONSUMER STOPPED")
    print("=" * 60)
    for key, value in stats.as_dict().items():
        print(f"  {key:<20} {value}")
    print("=" * 60)

    # Poison messages and database errors mean the run was not clean, even
    # though the consumer survived them.
    return 0 if stats.db_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
