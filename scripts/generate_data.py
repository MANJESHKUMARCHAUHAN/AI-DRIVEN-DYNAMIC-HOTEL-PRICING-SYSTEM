"""Generate the synthetic historical dataset and write it to ``data/synthetic``.

Writes five CSVs -- hotels, rooms, bookings, competitor prices and demand
signals -- and prints a per-hotel sanity summary so an obviously broken run is
caught here rather than three phases later when a model refuses to learn.

Usage::

    python scripts/generate_data.py                       # use .env settings
    python scripts/generate_data.py --hotels 8 --days 540
    python scripts/generate_data.py --seed 7 --output /tmp/data

The generation is deterministic: the same seed produces identical files.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

# Allow execution as a plain script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings  # noqa: E402
from ingestion.synthetic_dataset import (  # noqa: E402
    HOTEL_CATALOG,
    SyntheticDatasetGenerator,
    summarise,
)
from monitoring.logging_config import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


def _build_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/generate_data.py",
        description="Generate the synthetic hotel dataset used for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--seed", type=int, default=defaults.synthetic_seed,
        help="Master random seed. Same seed, same data.",
    )
    parser.add_argument(
        "--hotels", type=int, default=defaults.synthetic_hotels,
        help=f"Properties to generate, 1-{len(HOTEL_CATALOG)}.",
    )
    parser.add_argument(
        "--days", type=int, default=defaults.synthetic_history_days,
        help="Length of the generated history in days.",
    )
    parser.add_argument(
        "--end-date", type=str, default=None,
        help="Last stay date, YYYY-MM-DD. Defaults to today (UTC).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory. Defaults to DATA_DIR/synthetic.",
    )
    parser.add_argument(
        "--no-summary", action="store_true",
        help="Skip the per-hotel summary table.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    settings = get_settings()
    configure_logging(settings)

    args = _build_parser(settings.ingestion).parse_args(argv)

    end_date: Optional[date] = None
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
        except ValueError:
            logger.error("--end-date must be YYYY-MM-DD, got %r", args.end_date)
            return 2

    output = args.output or settings.paths.synthetic_dir

    try:
        generator = SyntheticDatasetGenerator(
            seed=args.seed,
            n_hotels=args.hotels,
            history_days=args.days,
            end_date=end_date,
            settings=settings,
        )
        dataset = generator.generate()
    except ValueError as exc:
        logger.error("Cannot generate dataset: %s", exc)
        return 2

    dataset.metadata["written_at"] = datetime.now(timezone.utc).isoformat()
    dataset.to_csv(output)

    if not args.no_summary:
        print()
        print("=" * 96)
        print("PER-HOTEL SUMMARY")
        print("=" * 96)
        print(summarise(dataset).to_string())
        print("=" * 96)

    print()
    for name, count in dataset.row_counts().items():
        print(f"  {name:<20} {count:>9,} rows")
    print(f"\nWritten to {output}")
    print("Next: python scripts/seed_database.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
