"""Load the synthetic dataset into PostgreSQL.

Reads the CSVs written by ``scripts/generate_data.py`` (or generates them in
memory with ``--generate``) and bulk-inserts them in dependency order:
hotels -> rooms -> bookings -> competitor prices -> demand features.

Usage::

    python scripts/seed_database.py                 # load data/synthetic
    python scripts/seed_database.py --reset         # drop and recreate schema
    python scripts/seed_database.py --generate      # skip the CSV step
    python scripts/seed_database.py --dry-run       # validate, insert nothing

Re-running is safe: the seeder truncates the tables it owns before loading, so
the database ends up holding exactly one copy of the dataset rather than two.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow execution as a plain script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from sqlalchemy import Engine, Table, func, select, text  # noqa: E402

from config import get_settings  # noqa: E402
from database.connection import get_engine, session_scope, wait_for_database  # noqa: E402
from database.init_db import init_database  # noqa: E402
from database.models import (  # noqa: E402
    Booking,
    CompetitorPrice,
    DemandFeature,
    Hotel,
    Room,
)
from ingestion.synthetic_dataset import (  # noqa: E402
    SyntheticDataset,
    SyntheticDatasetGenerator,
)
from monitoring.logging_config import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

#: Frame name -> target table, in insert order. Parents first: the composite
#: foreign keys on the fact tables will reject rows whose room does not exist.
LOAD_ORDER: List[tuple] = [
    ("hotels", Hotel),
    ("rooms", Room),
    ("bookings", Booking),
    ("competitor_prices", CompetitorPrice),
    ("demand_signals", DemandFeature),
]

#: Rows per INSERT batch. Large enough that round-trip latency stops mattering,
#: small enough that a failure does not roll back twenty minutes of work.
CHUNK_SIZE = 5_000


def _align_columns(frame: pd.DataFrame, table: Table) -> pd.DataFrame:
    """Drop frame columns the table does not have, and report what was dropped.

    The generator emits a few analyst-facing extras (``event_names``) that are
    intentionally not persisted. Silently discarding columns would hide a real
    schema mismatch, so anything dropped is logged.
    """
    table_columns = {c.name for c in table.columns}
    keep = [c for c in frame.columns if c in table_columns]
    dropped = [c for c in frame.columns if c not in table_columns]
    if dropped:
        logger.debug("Ignoring non-persisted column(s): %s", ", ".join(dropped))

    missing_required = [
        c.name
        for c in table.columns
        if c.name not in keep
        and not c.nullable
        and c.default is None
        and c.server_default is None
        and not c.autoincrement
    ]
    if missing_required:
        raise ValueError(
            f"Frame for table {table.name!r} is missing required column(s): "
            + ", ".join(missing_required)
        )
    return frame[keep]


def _to_records(frame: pd.DataFrame) -> List[Dict[str, Any]]:
    """Frame -> list of dicts with NaN converted to ``None``.

    ``DataFrame.to_dict`` leaves ``NaN`` in place, and psycopg2 sends that as
    the float ``nan``, which a ``VARCHAR`` column rejects with a confusing
    error. Converting up front keeps the failure modes honest.
    """
    return frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")


def truncate_tables(engine: Engine) -> None:
    """Empty every table the seeder writes to, children first."""
    tables = [model.__table__ for _, model in reversed(LOAD_ORDER)]
    with engine.begin() as conn:
        if engine.dialect.name == "postgresql":
            names = ", ".join(t.name for t in tables)
            conn.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))
        else:
            for table in tables:
                conn.execute(table.delete())
    logger.info("Truncated %d table(s)", len(tables))


def load_frame(engine: Engine, frame: pd.DataFrame, model: Any) -> int:
    """Bulk-insert one frame. Returns the number of rows written."""
    table: Table = model.__table__
    if frame.empty:
        logger.warning("Frame for %s is empty; nothing to load", table.name)
        return 0

    aligned = _align_columns(frame, table)
    records = _to_records(aligned)

    started = time.perf_counter()
    with engine.begin() as conn:
        for start in range(0, len(records), CHUNK_SIZE):
            conn.execute(table.insert(), records[start : start + CHUNK_SIZE])

    elapsed = time.perf_counter() - started
    logger.info(
        "Loaded %-18s %7d rows in %5.1fs (%s rows/s)",
        table.name,
        len(records),
        elapsed,
        f"{len(records) / elapsed:,.0f}" if elapsed else "inf",
    )
    return len(records)


def seed(
    dataset: SyntheticDataset,
    *,
    engine: Optional[Engine] = None,
    truncate: bool = True,
) -> Dict[str, int]:
    """Load a dataset into the database.

    Args:
        dataset: The frames to load.
        engine: Engine override, for tests.
        truncate: Empty the target tables first. Turning this off will violate
            the unique constraints on a second run -- which is the intended
            behaviour, not a bug to work around.

    Returns:
        Table name -> rows inserted.
    """
    engine = engine or get_engine()
    frames = dataset.frames()

    if truncate:
        truncate_tables(engine)

    written: Dict[str, int] = {}
    for frame_name, model in LOAD_ORDER:
        written[model.__tablename__] = load_frame(engine, frames[frame_name], model)
    return written


def verify(engine: Optional[Engine] = None) -> Dict[str, int]:
    """Count rows per table and assert the reference data is coherent."""
    engine = engine or get_engine()
    counts: Dict[str, int] = {}

    with session_scope() as session:
        for _, model in LOAD_ORDER:
            counts[model.__tablename__] = int(
                session.execute(select(func.count()).select_from(model.__table__)).scalar_one()
            )

        # Every room must belong to a hotel that exists, and every hotel must
        # have rooms. Both are enforced by constraints; checking them here turns
        # "the seed ran" into "the seed produced usable data".
        orphan_rooms = session.execute(
            select(func.count())
            .select_from(Room.__table__)
            .where(~Room.hotel_id.in_(select(Hotel.hotel_id)))
        ).scalar_one()
        if orphan_rooms:
            raise RuntimeError(f"{orphan_rooms} room(s) reference a missing hotel")

        hotels_without_rooms = session.execute(
            select(func.count())
            .select_from(Hotel.__table__)
            .where(~Hotel.hotel_id.in_(select(Room.hotel_id)))
        ).scalar_one()
        if hotels_without_rooms:
            raise RuntimeError(f"{hotels_without_rooms} hotel(s) have no rooms")

    return counts


def _build_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/seed_database.py",
        description="Load the synthetic dataset into PostgreSQL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Directory holding the generated CSVs. Defaults to DATA_DIR/synthetic.",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Generate the dataset in memory instead of reading CSVs.",
    )
    parser.add_argument(
        "--seed", type=int, default=defaults.synthetic_seed,
        help="Seed used when --generate is given.",
    )
    parser.add_argument(
        "--hotels", type=int, default=defaults.synthetic_hotels,
        help="Hotel count used when --generate is given.",
    )
    parser.add_argument(
        "--days", type=int, default=defaults.synthetic_history_days,
        help="History length used when --generate is given.",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Drop and recreate the schema before loading. Destroys all data.",
    )
    parser.add_argument(
        "--no-truncate", action="store_true",
        help="Append instead of replacing. Will fail on unique constraints if "
        "the tables already hold this dataset.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Load and validate the frames but write nothing.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    settings = get_settings()
    configure_logging(settings)

    args = _build_parser(settings.ingestion).parse_args(argv)

    # --- obtain the dataset ------------------------------------------------
    if args.generate:
        dataset = SyntheticDatasetGenerator(
            seed=args.seed,
            n_hotels=args.hotels,
            history_days=args.days,
            settings=settings,
        ).generate()
    else:
        source = args.input or settings.paths.synthetic_dir
        try:
            dataset = SyntheticDataset.from_csv(source)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            return 2
        logger.info("Read dataset from %s: %s", source, dataset.row_counts())

    if args.dry_run:
        for frame_name, model in LOAD_ORDER:
            _align_columns(dataset.frames()[frame_name], model.__table__)
        print("Dry run: all frames align with the schema. Nothing written.")
        for name, count in dataset.row_counts().items():
            print(f"  {name:<20} {count:>9,} rows")
        return 0

    # --- write -------------------------------------------------------------
    try:
        wait_for_database()
        init_database(drop=args.reset, wait=False)
        written = seed(dataset, truncate=not args.no_truncate)
        counts = verify()
    except Exception as exc:
        logger.error("Seeding failed: %s: %s", type(exc).__name__, exc)
        return 1

    print()
    print("=" * 60)
    print("SEED COMPLETE")
    print("=" * 60)
    for table, count in counts.items():
        inserted = written.get(table, 0)
        print(f"  {table:<20} {count:>9,} rows  (inserted {inserted:,})")
    print("=" * 60)
    print("Next: python scripts/create_topics.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
