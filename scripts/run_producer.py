"""Stream competitor price events into Kafka.

Reads from the configured competitor data source and publishes each observation
to ``hotel.competitor_prices``. Two shapes of source, handled differently:

* ``SyntheticCompetitorGenerator`` has its own ``stream()`` that already handles
  sweeps, pacing and bounds.
* A **scraper** exposes ``collect_many()`` over a request set that the caller has
  to build -- see :func:`scraper_stream`. Its outbound pacing comes from
  ``INGESTION_RATE_LIMIT_SECONDS``, not from ``--interval``.

Under Compose the default is ``INGESTION_SOURCE=demo_ota``, so the background
feed is genuine scraping against the bundled demo site.

Base prices are read from the ``rooms`` table when the database is reachable, so
the feed prices against the inventory that actually exists rather than against
the shipped catalogue. If the database is down the generator falls back to the
catalogue and says so; a competitor feed that stops because an unrelated
database is unavailable would be a bad trade.

Usage::

    python scripts/run_producer.py                     # run until Ctrl-C
    python scripts/run_producer.py --max-events 200    # bounded, for demos
    python scripts/run_producer.py --interval 0        # as fast as possible
    python scripts/run_producer.py --passes 1          # one sweep, then exit
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow execution as a plain script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402

from config import get_settings  # noqa: E402
from database.connection import session_scope  # noqa: E402
from database.models import Hotel, Room, RoomType  # noqa: E402
from ingestion.scraper_base import (  # noqa: E402
    ScraperDisabled,
    ScraperError,
    build_requests,
    get_scraper,
)
from ingestion.synthetic_dataset import HOTEL_CATALOG  # noqa: E402
from ingestion.synthetic_generator import (  # noqa: E402
    DEFAULT_HORIZONS,
    SyntheticCompetitorGenerator,
)
from monitoring.logging_config import configure_logging, get_logger  # noqa: E402
from streaming.producer import EventProducer  # noqa: E402
from streaming.topics import TopicName  # noqa: E402

logger = get_logger(__name__)


def load_catalog_from_database() -> Optional[Dict[Tuple[str, RoomType], float]]:
    """Read ``(hotel_id, room_type) -> base_price`` from the ``rooms`` table.

    Returns ``None`` when the database is unreachable or unseeded, so the caller
    can fall back to the shipped catalogue.
    """
    try:
        with session_scope() as session:
            rows = session.execute(
                select(Room.hotel_id, Room.room_type, Room.base_price)
            ).all()
    except SQLAlchemyError as exc:
        logger.warning(
            "Could not read room prices (%s); falling back to the shipped catalogue",
            type(exc).__name__,
        )
        return None

    if not rows:
        logger.warning(
            "The rooms table is empty; falling back to the shipped catalogue. "
            "Run scripts/seed_database.py to populate it."
        )
        return None

    catalog = {
        (hotel_id, room_type if isinstance(room_type, RoomType) else RoomType(room_type)):
            float(base_price)
        for hotel_id, room_type, base_price in rows
    }
    logger.info("Loaded %d room prices from the database", len(catalog))
    return catalog


def load_locations_from_database() -> Optional[List[Tuple[str, str]]]:
    """Read ``(hotel_id, city)`` pairs from the ``hotels`` table.

    Scrapers search by city, so they need the location a hotel is in -- which the
    room-price catalogue does not carry. Returns ``None`` when the database is
    unreachable or unseeded so the caller can fall back to the shipped catalogue.
    """
    try:
        with session_scope() as session:
            rows = session.execute(select(Hotel.hotel_id, Hotel.city)).all()
    except SQLAlchemyError as exc:
        logger.warning(
            "Could not read hotel locations (%s); falling back to the shipped catalogue",
            type(exc).__name__,
        )
        return None

    if not rows:
        logger.warning("The hotels table is empty; falling back to the shipped catalogue.")
        return None
    return [(hotel_id, city) for hotel_id, city in rows]


def scraper_stream(
    source,
    *,
    horizons: List[int],
    hotel_ids: Optional[List[str]],
    passes: Optional[int],
    interval: float,
    stopping: Dict[str, bool],
):
    """Yield payloads from a scraping source, sweep after sweep.

    The synthetic generator has its own ``stream()`` that already handles passes,
    pacing and bounds. A scraper does not: it exposes ``collect_many()`` over a
    request set, and building that set is the caller's job. This function is
    that job.

    Note there is no ``interval`` sleep between individual fetches -- the
    scraper's own rate limiter (``INGESTION_RATE_LIMIT_SECONDS``) already paces
    outbound requests, and pacing twice would silently double the time a sweep
    takes. ``interval`` is applied between *sweeps* instead.
    """
    locations = load_locations_from_database()
    if locations is None:
        locations = [(profile.hotel_id, profile.city) for profile in HOTEL_CATALOG]
    if hotel_ids:
        wanted = set(hotel_ids)
        locations = [pair for pair in locations if pair[0] in wanted]

    if not locations:
        logger.error("No hotels to collect for; nothing to publish.")
        return

    completed = 0
    while passes is None or completed < passes:
        if stopping["flag"]:
            return

        # Rebuilt every sweep on purpose: build_requests() is relative to today,
        # so a long-running producer must not keep scraping the dates it was
        # started with.
        requests = build_requests(locations, horizons=horizons)
        logger.info(
            "Sweep %d: %d request(s) across %d hotel(s)",
            completed + 1,
            len(requests),
            len(locations),
        )

        for payload in source.collect_many(requests):
            if stopping["flag"]:
                return
            yield payload

        completed += 1
        if (passes is None or completed < passes) and interval > 0:
            time.sleep(interval)


def _build_parser(defaults) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/run_producer.py",
        description="Publish competitor price events to Kafka.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-events", type=int, default=None,
        help="Stop after this many events. Default: run until interrupted.",
    )
    parser.add_argument(
        "--passes", type=int, default=None,
        help="Number of full sweeps over the request set. Default: unbounded.",
    )
    parser.add_argument(
        "--interval", type=float, default=defaults.synthetic_interval_seconds,
        help="Seconds between events. 0 publishes as fast as possible.",
    )
    parser.add_argument(
        "--horizons", type=int, nargs="+", default=list(DEFAULT_HORIZONS),
        help="Days ahead to collect rates for.",
    )
    parser.add_argument(
        "--hotels", type=str, nargs="+", default=None,
        help="Restrict to these hotel ids. Default: every hotel in the catalogue.",
    )
    parser.add_argument(
        "--no-database", action="store_true",
        help="Skip the room-price lookup and use the shipped catalogue.",
    )
    parser.add_argument(
        "--report-every", type=int, default=50,
        help="Log a progress line every N events.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    settings = get_settings()
    configure_logging(settings)
    args = _build_parser(settings.ingestion).parse_args(argv)

    if not settings.kafka.enabled:
        logger.error("KAFKA_ENABLED=false; nothing would be published. Aborting.")
        return 2

    # --- pick the data source ---------------------------------------------
    try:
        source = get_scraper(settings)
    except ScraperDisabled as exc:
        logger.error("%s", exc)
        return 2
    except ScraperError as exc:
        logger.error("Cannot build a competitor data source: %s", exc)
        return 2

    if isinstance(source, SyntheticCompetitorGenerator) and not args.no_database:
        catalog = load_catalog_from_database()
        if catalog:
            source.catalog = catalog

    logger.info(
        "Publishing from %r to %s (horizons=%s, interval=%.1fs)",
        source.name,
        settings.kafka.topic_competitor,
        args.horizons,
        args.interval,
    )

    # --- graceful shutdown -------------------------------------------------
    stopping = {"flag": False}

    def _stop(signum, _frame):  # type: ignore[no-untyped-def]
        logger.info("Signal %s received; finishing current event", signum)
        stopping["flag"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            pass

    # --- publish -----------------------------------------------------------
    started = time.perf_counter()
    published = 0

    with EventProducer(settings) as producer:
        if isinstance(source, SyntheticCompetitorGenerator):
            stream = source.stream(
                horizons=args.horizons,
                hotel_ids=args.hotels,
                max_events=args.max_events,
                interval_seconds=args.interval,
                passes=args.passes,
            )
        else:
            # Scrapers have no stream() of their own: they expose collect_many()
            # over a request set that somebody has to build. An earlier version
            # called source.requests(), which only SyntheticCompetitorGenerator
            # has -- so this branch raised AttributeError the moment a real
            # scraper was configured. It went unnoticed because until the demo
            # OTA existed, no scraper could get past robots.txt to reach it.
            stream = scraper_stream(
                source,
                horizons=args.horizons,
                hotel_ids=args.hotels,
                passes=args.passes,
                interval=args.interval,
                stopping=stopping,
            )

        for payload in stream:
            if stopping["flag"]:
                break
            if args.max_events is not None and published >= args.max_events:
                break
            if producer.send(payload, TopicName.COMPETITOR_PRICES, source=source.name):
                published += 1
            if args.report_every and published and published % args.report_every == 0:
                logger.info(
                    "Published %d event(s) (%.1f/s)",
                    published,
                    published / max(time.perf_counter() - started, 1e-9),
                )

        producer.flush(timeout=10.0)
        stats = producer.stats.as_dict()

    elapsed = time.perf_counter() - started
    print()
    print("=" * 60)
    print("PRODUCER STOPPED")
    print("=" * 60)
    print(f"  published        {published:,}")
    print(f"  elapsed          {elapsed:,.1f}s")
    print(f"  throughput       {published / max(elapsed, 1e-9):,.1f} events/s")
    for key, value in stats.items():
        print(f"  {key:<16} {value}")
    print("=" * 60)

    # A run that published nothing is a failure, however cleanly it exited.
    return 0 if published else 1


if __name__ == "__main__":
    raise SystemExit(main())
