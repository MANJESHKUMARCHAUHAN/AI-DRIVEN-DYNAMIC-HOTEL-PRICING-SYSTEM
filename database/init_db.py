"""Schema creation, verification and teardown.

``create_all`` is used rather than Alembic. For a system whose schema is created
once per environment and whose "migration" story is `docker compose down -v`,
Alembic is ceremony without benefit -- but the naming convention in
:mod:`database.models` means adding it later is a drop-in, not a rewrite.

Run directly::

    python -m database.init_db                # create anything missing
    python -m database.init_db --drop         # destroy and recreate
    python -m database.init_db --check        # report only, change nothing
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional

from sqlalchemy import Engine, inspect

from config import get_settings
from database.connection import get_engine, wait_for_database
from database.models import TABLE_NAMES, Base
from monitoring.logging_config import configure_logging, get_logger

logger = get_logger(__name__)


def existing_tables(engine: Optional[Engine] = None) -> List[str]:
    """Table names currently present in the database."""
    engine = engine or get_engine()
    return sorted(inspect(engine).get_table_names())


def schema_status(engine: Optional[Engine] = None) -> Dict[str, List[str]]:
    """Compare the live schema against the ORM metadata.

    Returns:
        ``{"present": [...], "missing": [...], "unexpected": [...]}`` where
        *unexpected* lists tables in the database that the models do not define
        (a stale table from an older revision, usually).
    """
    engine = engine or get_engine()
    live = set(existing_tables(engine))
    declared = set(TABLE_NAMES)
    return {
        "present": sorted(live & declared),
        "missing": sorted(declared - live),
        "unexpected": sorted(live - declared),
    }


def create_schema(engine: Optional[Engine] = None) -> List[str]:
    """Create every missing table and index. Idempotent.

    Returns:
        The names of the tables that were created by this call.
    """
    engine = engine or get_engine()
    before = set(existing_tables(engine))

    Base.metadata.create_all(bind=engine)

    created = sorted(set(existing_tables(engine)) - before)
    if created:
        logger.info("Created %d table(s): %s", len(created), ", ".join(created))
    else:
        logger.info("Schema already up to date (%d tables)", len(TABLE_NAMES))
    return created


def drop_schema(engine: Optional[Engine] = None) -> None:
    """Drop every table this project owns. Destructive.

    Only tables declared in :mod:`database.models` are dropped -- anything else
    living in the same database is left alone.
    """
    engine = engine or get_engine()
    logger.warning("Dropping %d table(s): %s", len(TABLE_NAMES), ", ".join(TABLE_NAMES))
    Base.metadata.drop_all(bind=engine)
    logger.info("Schema dropped")


def init_database(
    *,
    drop: bool = False,
    wait: bool = True,
    engine: Optional[Engine] = None,
) -> Dict[str, List[str]]:
    """Bring the database to the schema the code expects.

    Args:
        drop: Destroy existing tables first. Everything in them is lost.
        wait: Block until the server accepts connections before doing anything.
        engine: Engine override, for tests.

    Returns:
        The schema status after the operation.
    """
    engine = engine or get_engine()
    settings = get_settings()

    if wait:
        wait_for_database(engine)

    logger.info("Initialising schema on %s", settings.database.safe_url)

    if drop:
        drop_schema(engine)

    create_schema(engine)

    status = schema_status(engine)
    if status["missing"]:
        raise RuntimeError(
            "Schema creation incomplete; still missing: "
            + ", ".join(status["missing"])
        )
    if status["unexpected"]:
        logger.warning(
            "Database contains %d table(s) unknown to the models: %s",
            len(status["unexpected"]),
            ", ".join(status["unexpected"]),
        )

    logger.info("Schema ready: %d table(s)", len(status["present"]))
    return status


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m database.init_db",
        description="Create, inspect or reset the pricing database schema.",
    )
    parser.add_argument(
        "--drop",
        action="store_true",
        help="Drop all project tables before creating them. Destroys all data.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report schema status and exit without modifying anything.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Fail immediately instead of waiting for the database to accept "
        "connections.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)
    configure_logging()

    try:
        if args.check:
            wait_for_database(retries=3, delay_seconds=1.0)
            status = schema_status()
            for label, tables in status.items():
                logger.info("%-10s %d %s", label, len(tables), tables or "")
            return 0 if not status["missing"] else 1

        init_database(drop=args.drop, wait=not args.no_wait)
        return 0
    except Exception as exc:
        logger.error("Schema initialisation failed: %s: %s", type(exc).__name__, exc)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI
    sys.exit(main())


__all__ = [
    "create_schema",
    "drop_schema",
    "existing_tables",
    "init_database",
    "main",
    "schema_status",
]
