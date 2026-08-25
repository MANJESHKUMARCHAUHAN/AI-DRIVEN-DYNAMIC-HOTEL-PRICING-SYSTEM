"""Bring the system from nothing to ready, skipping whatever is already done.

This is what Compose's ``init`` service runs, and what ``make bootstrap`` runs
locally. The important property is **idempotence**: ``docker compose up`` should
converge on a working stack, not reset it.

An earlier version was a shell chain that always ran every step, so restarting
one service re-seeded the database and destroyed every pricing decision and
manually submitted competitor rate. Surprising *and* destructive. Each step here
checks whether its work already exists and skips if so.

``--force`` re-runs everything, which is the behaviour you want after changing
the generator or the feature pipeline.

Usage::

    python scripts/bootstrap.py            # do only what is missing
    python scripts/bootstrap.py --force    # rebuild everything
    python scripts/bootstrap.py --check    # report state, change nothing
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

# Allow execution as a plain script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import func, select  # noqa: E402

from config import get_settings  # noqa: E402
from database.connection import session_scope, wait_for_database  # noqa: E402
from database.init_db import init_database, schema_status  # noqa: E402
from database.models import DemandFeature, Hotel  # noqa: E402
from monitoring.logging_config import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)

#: Below this many seeded hotels the database counts as empty.
MIN_HOTELS = 1

#: Below this many computed feature rows the feature store counts as unbuilt.
MIN_FEATURE_ROWS = 500


@dataclass
class Step:
    """One bootstrap step: a name, a test for 'already done', and the work."""

    name: str
    is_done: Callable[[], bool]
    run: Callable[[], int]
    describe_done: str


# --------------------------------------------------------------------------- #
# State checks
# --------------------------------------------------------------------------- #


def schema_exists() -> bool:
    try:
        return not schema_status()["missing"]
    except Exception:
        return False


def database_is_seeded() -> bool:
    try:
        with session_scope() as session:
            hotels = session.execute(
                select(func.count()).select_from(Hotel.__table__)
            ).scalar_one()
        return hotels >= MIN_HOTELS
    except Exception:
        return False


def features_are_built() -> bool:
    try:
        with session_scope() as session:
            rows = session.execute(
                select(func.count())
                .select_from(DemandFeature.__table__)
                .where(DemandFeature.feature_version.is_not(None))
            ).scalar_one()
        return rows >= MIN_FEATURE_ROWS
    except Exception:
        return False


def models_are_trained() -> bool:
    artifact_dir = get_settings().model.model_dir
    return artifact_dir.is_dir() and any(artifact_dir.glob("gbr_v*.joblib"))


def topics_exist() -> bool:
    settings = get_settings()
    if not settings.kafka.enabled:
        return True
    try:
        from streaming.admin import existing_topics

        present = set(existing_topics(settings))
        return set(settings.kafka.topics.values()) <= present
    except Exception:
        return False


def synthetic_data_exists() -> bool:
    directory = get_settings().paths.synthetic_dir
    return (directory / "bookings.csv").is_file()


# --------------------------------------------------------------------------- #
# Work
# --------------------------------------------------------------------------- #


def _run_script(module: str, *args: str) -> int:
    """Run one of the CLI scripts in-process, returning its exit code."""
    import importlib

    entry = importlib.import_module(module)
    return int(entry.main(list(args)) or 0)


def create_schema() -> int:
    init_database(wait=False)
    return 0


def create_topics() -> int:
    if not get_settings().kafka.enabled:
        logger.info("KAFKA_ENABLED=false; skipping topic creation")
        return 0
    return _run_script("scripts.create_topics")


def generate_data() -> int:
    return _run_script("scripts.generate_data", "--no-summary")


def seed_database() -> int:
    return _run_script("scripts.seed_database")


def build_features() -> int:
    return _run_script("scripts.build_features", "--no-correlations")


def train_models() -> int:
    return _run_script("scripts.train_models", "--no-backtest")


STEPS: List[Step] = [
    Step("schema", schema_exists, create_schema, "all 9 tables exist"),
    Step("topics", topics_exist, create_topics, "all 4 topics exist"),
    Step("synthetic data", synthetic_data_exists, generate_data, "CSVs are present"),
    Step("seed", database_is_seeded, seed_database, "hotels are loaded"),
    Step("features", features_are_built, build_features, "the feature store is populated"),
    Step("models", models_are_trained, train_models, "artifacts exist"),
]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/bootstrap.py",
        description="Bring the system to a ready state, skipping completed steps.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run every step, including the destructive ones. Reseeds the "
        "database and discards pricing history.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what is done and what is missing, then exit.",
    )
    parser.add_argument(
        "--skip-training", action="store_true", help="Stop before training models."
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    settings = get_settings()
    configure_logging(settings)
    args = _build_parser().parse_args(argv)

    settings.ensure_directories()

    try:
        wait_for_database()
    except Exception as exc:
        logger.error("Database unreachable: %s", exc)
        return 1

    steps = STEPS[:-1] if args.skip_training else STEPS

    if args.check:
        print()
        print("=" * 66)
        print("BOOTSTRAP STATE")
        print("=" * 66)
        for step in steps:
            done = step.is_done()
            marker = "done" if done else "MISSING"
            detail = step.describe_done if done else "not yet run"
            print(f"  [{marker:>7}] {step.name:<16} {detail}")
        print("=" * 66)
        return 0

    started = time.perf_counter()
    ran: List[str] = []
    skipped: List[str] = []

    for step in steps:
        if not args.force and step.is_done():
            logger.info("Skipping %s -- %s", step.name, step.describe_done)
            skipped.append(step.name)
            continue

        logger.info("--> %s", step.name)
        code = step.run()
        if code != 0:
            logger.error("Step %r failed with exit code %d", step.name, code)
            return code
        ran.append(step.name)

    elapsed = time.perf_counter() - started
    print()
    print("=" * 66)
    print("BOOTSTRAP COMPLETE")
    print("=" * 66)
    print(f"  ran      {', '.join(ran) or '(nothing -- already ready)'}")
    print(f"  skipped  {', '.join(skipped) or '(nothing)'}")
    print(f"  took     {elapsed:.1f}s")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
