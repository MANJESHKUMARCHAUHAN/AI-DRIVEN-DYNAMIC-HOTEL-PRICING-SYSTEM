"""Build the feature matrix from the database and store it.

Reads bookings, competitor rates and demand signals, computes the model-ready
features described in :mod:`features.feature_engineering`, and upserts them onto
``demand_features``. Only the derived columns are written; the exogenous signals
the streaming consumer owns are left alone.

Usage::

    python scripts/build_features.py                 # build and store
    python scripts/build_features.py --dry-run       # build, report, store nothing
    python scripts/build_features.py --export out/   # also write a parquet/CSV copy
    python scripts/build_features.py --report        # show the feature contract
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

# Allow execution as a plain script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config import get_settings  # noqa: E402
from database.connection import session_scope, wait_for_database  # noqa: E402
from features.feature_engineering import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    TARGET_COLUMN,
    describe_features,
)
from features.feature_store import FeatureStore, save_feature_list  # noqa: E402
from monitoring.logging_config import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/build_features.py",
        description="Compute and store the model-ready feature matrix.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Build and summarise the matrix without writing to the database.",
    )
    parser.add_argument(
        "--export", type=Path, default=None,
        help="Also write the matrix to this directory as CSV.",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Print the feature contract and exit.",
    )
    parser.add_argument(
        "--no-correlations", action="store_true",
        help="Skip the target-correlation table.",
    )
    return parser


def _print_summary(frame: pd.DataFrame, *, correlations: bool) -> None:
    print()
    print("=" * 78)
    print(f"FEATURE MATRIX  ({FEATURE_VERSION})")
    print("=" * 78)
    print(f"  rows              {len(frame):,}")
    print(f"  features          {len(FEATURE_COLUMNS)}")
    print(f"  hotels            {frame['hotel_id'].nunique()}")
    print(f"  stay dates        {frame['stay_date'].nunique():,}")
    print(f"  date range        {frame['stay_date'].min().date()} .. "
          f"{frame['stay_date'].max().date()}")
    print(f"  horizons          {sorted(frame['days_to_checkin'].unique().tolist())}")
    print(f"  competitor gaps   {frame['competitor_missing'].mean():.1%} of rows")
    print(f"  target mean/std   {frame[TARGET_COLUMN].mean():.3f} / "
          f"{frame[TARGET_COLUMN].std():.3f}")

    if correlations:
        matrix = frame[list(FEATURE_COLUMNS) + [TARGET_COLUMN]].corr()[TARGET_COLUMN]
        ranked = matrix.drop(TARGET_COLUMN).abs().sort_values(ascending=False)
        print("\n  strongest linear relationships with the target:")
        for name, value in ranked.head(12).items():
            sign = "+" if matrix[name] >= 0 else "-"
            print(f"    {name:<22} {sign}{value:.3f}")
        print("\n  (trees also use interactions, so a low linear score here does")
        print("   not mean a feature is useless -- days_to_checkin is the example)")
    print("=" * 78)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    settings = get_settings()
    configure_logging(settings)
    args = _build_parser().parse_args(argv)

    if args.report:
        print(describe_features().to_string(index=False))
        print(f"\n{len(FEATURE_COLUMNS)} features, version {FEATURE_VERSION}")
        return 0

    store = FeatureStore()
    started = time.perf_counter()

    try:
        wait_for_database()
        with session_scope() as session:
            frame = store.build(session)
            if not args.dry_run:
                store.write(frame, session)
                coverage = store.coverage(session)
            else:
                coverage = None
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    except Exception as exc:
        logger.error("Feature build failed: %s: %s", type(exc).__name__, exc)
        return 1

    _print_summary(frame, correlations=not args.no_correlations)

    if args.export:
        args.export.mkdir(parents=True, exist_ok=True)
        path = args.export / f"features_{FEATURE_VERSION}.csv"
        frame.to_csv(path, index=False)
        save_feature_list(args.export)
        print(f"\nExported to {path}")

    if coverage:
        print(f"\nStored: {coverage['computed']:,}/{coverage['rows']:,} rows "
              f"({coverage['coverage']:.1%}), {coverage['labelled']:,} labelled")
        print("Next: python scripts/train_models.py")
    else:
        print("\nDry run: nothing was written.")

    logger.info("Completed in %.1fs", time.perf_counter() - started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
