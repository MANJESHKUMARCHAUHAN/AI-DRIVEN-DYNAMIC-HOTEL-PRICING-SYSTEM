"""Create the project's Kafka topics.

Topics are created explicitly rather than relying on auto-creation, because an
auto-created topic silently takes the broker's defaults -- usually one partition
and whatever retention the cluster happens to have. Partition count and
retention are pricing-pipeline decisions and belong in reviewable code.

Usage::

    python scripts/create_topics.py              # create anything missing
    python scripts/create_topics.py --describe   # show what exists
    python scripts/create_topics.py --dry-run    # print the plan only
    python scripts/create_topics.py --delete     # destroy them (destructive)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Allow execution as a plain script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from monitoring.logging_config import configure_logging, get_logger  # noqa: E402
from streaming.admin import (  # noqa: E402
    BrokerUnavailable,
    create_topics,
    delete_topics,
    describe_topics,
    wait_for_broker,
)
from streaming.topics import build_topic_specs  # noqa: E402

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/create_topics.py",
        description="Create, inspect or delete the pricing pipeline's Kafka topics.",
    )
    parser.add_argument(
        "--describe", action="store_true",
        help="Show the partition layout of the existing topics and exit.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be created without contacting the cluster.",
    )
    parser.add_argument(
        "--delete", action="store_true",
        help="Delete every topic in the catalogue. Destroys all buffered events.",
    )
    parser.add_argument(
        "--wait-retries", type=int, default=30,
        help="How many times to wait for the broker before giving up.",
    )
    return parser


def _print_catalogue(settings) -> None:
    print("=" * 88)
    print("TOPIC CATALOGUE")
    print("=" * 88)
    print(f"  {'topic':<28} {'parts':>5} {'repl':>5} {'retention':>10}  description")
    print("-" * 88)
    for spec in build_topic_specs(settings).values():
        print(
            f"  {spec.name:<28} {spec.partitions:>5} {spec.replication_factor:>5} "
            f"{spec.retention_ms // 3_600_000:>8}h  {spec.description}"
        )
    print("=" * 88)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    settings = get_settings()
    configure_logging(settings)
    args = _build_parser().parse_args(argv)

    if not settings.kafka.enabled:
        logger.error(
            "KAFKA_ENABLED=false. Set it to true in .env before creating topics."
        )
        return 2

    _print_catalogue(settings)

    if args.dry_run:
        for name, status in create_topics(settings, dry_run=True).items():
            print(f"  {name:<28} {status}")
        return 0

    try:
        wait_for_broker(settings, retries=args.wait_retries)
    except BrokerUnavailable as exc:
        logger.error("%s", exc)
        logger.error(
            "Is the broker running? Try: docker compose up -d kafka  "
            "(bootstrap=%s)",
            settings.kafka.bootstrap_servers,
        )
        return 1

    try:
        if args.delete:
            deleted = delete_topics(settings)
            print(f"\nDeleted {len(deleted)} topic(s): {', '.join(deleted) or '(none)'}")
            return 0

        if args.describe:
            rows = describe_topics(settings)
            if not rows:
                print("\nNo project topics exist yet. Run without --describe first.")
                return 1
            print()
            for row in rows:
                print(
                    f"  {row['topic']:<28} partitions={row['partitions']} "
                    f"replicas={row['replicas']} leaders={row['leaders']}"
                )
            return 0

        results = create_topics(settings)
    except Exception as exc:
        logger.error("Topic operation failed: %s: %s", type(exc).__name__, exc)
        return 1

    print()
    for name, status in sorted(results.items()):
        print(f"  {name:<28} {status}")
    print("\nNext: python scripts/run_consumer.py   (then, in another shell)")
    print("      python scripts/run_producer.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
