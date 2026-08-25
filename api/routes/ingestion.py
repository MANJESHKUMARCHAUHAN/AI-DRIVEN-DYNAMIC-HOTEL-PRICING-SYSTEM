"""Competitor ingestion: what the feed is doing, and running a pass on demand.

Two endpoints:

``GET  /api/v1/ingestion/status``  configuration, crawl permission, row counts
``POST /api/v1/ingestion/run``     collect now, optionally publishing the result

WHY THE DASHBOARD DOES NOT SCRAPE FOR ITSELF
--------------------------------------------
It would be shorter to call ``get_scraper()`` from the Streamlit page. It would
also put a second implementation of "collect competitor rates" in the codebase,
running in a process with no database session management and no structured
logging, and it would mean a rate you saw on screen was not the rate that landed
in the table. Everything the dashboard renders comes through the API for the same
reason every price does: one implementation, one audit trail.

WHY THE RUN IS SYNCHRONOUS
--------------------------
The pass is bounded -- hotels x room types x horizons, paced by the rate limiter
-- and the caller wants the rows back. A background task would need a job table,
a polling endpoint and a way to report partial failure, which is real work for no
gain at this size. ``POST /models/train`` is synchronous for the same reason and
documents the same trade. The request-count guard on ``horizons`` is what keeps
"bounded" true.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.dependencies import session_dependency, settings_dependency
from api.security import require_write
from api.schemas import (
    IngestionRunRequest,
    IngestionRunResponse,
    IngestionSourceInfo,
    IngestionStatusResponse,
    RobotsStatus,
    ScrapedRate,
)
from config import CompetitorSource, Settings
from database.models import CompetitorPrice, Hotel, RoomType
from ingestion.scraper_base import (
    ScrapeRequest,
    ScraperBlocked,
    ScraperDisabled,
    ScraperError,
    get_scraper,
)
from monitoring.logging_config import get_logger
from monitoring.metrics import observe_ingestion
from streaming.events import EventEnvelope
from streaming.handlers import handle_competitor_price
from streaming.producer import get_producer
from streaming.topics import TopicName

logger = get_logger(__name__)

router = APIRouter(tags=["ingestion"])

#: Most rates echoed back in a run response. A full pass over eight hotels, four
#: categories and five horizons is 160 requests and can produce several hundred
#: rates; the dashboard only needs enough to show that it worked.
MAX_RATES_IN_RESPONSE = 300

#: Ceiling on how long a synchronous pass may be *expected* to take, in seconds.
#:
#: The binding constraint is wall-clock, not request count. A pass is paced by
#: ``INGESTION_RATE_LIMIT_SECONDS``, so the same 96 requests are 0 seconds with
#: the limiter off and eight minutes at five seconds apart -- and it is the
#: second one that hangs a browser and trips a proxy.
#:
#: A pure count cap misses this. An earlier version capped at 400 requests,
#: which the schema's 12-horizon limit made unreachable: the full catalogue of
#: eight hotels across four categories and twelve horizons is 384. The guard
#: could never fire, so the only real protection was that nobody tried.
MAX_RUN_SECONDS = 300.0

#: Backstop for when the rate limiter is off and the duration estimate is zero.
#: Bounds memory and response size rather than time.
MAX_REQUESTS_PER_RUN = 400


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _source_info(settings: Settings) -> IngestionSourceInfo:
    """Describe the configured source without touching the network."""
    source = settings.ingestion.source
    is_scraper = source is not CompetitorSource.SYNTHETIC

    base_url: Optional[str] = None
    layout: Optional[str] = None
    if source is CompetitorSource.DEMO_OTA:
        base_url = settings.ingestion.demo_ota_base_url
        layout = settings.ingestion.demo_ota_layout
    elif source is CompetitorSource.BOOKING:
        base_url = "https://www.booking.com"
    elif source is CompetitorSource.EXPEDIA:
        base_url = "https://www.expedia.com"

    return IngestionSourceInfo(
        source=source.value,
        is_scraper=is_scraper,
        is_third_party=source.is_third_party,
        real_scrapers_enabled=settings.ingestion.enable_real_scrapers,
        base_url=base_url,
        layout=layout,
        rate_limit_seconds=settings.ingestion.rate_limit_seconds,
        user_agent=settings.ingestion.user_agent,
    )


def _robots_status(settings: Settings) -> RobotsStatus:
    """Read the target's robots.txt and report what it says.

    Never raises. A source with no robots policy to check, an unreachable site
    and a site that says no are three different answers, and the status endpoint
    has to be able to return all three rather than 500.
    """
    source = settings.ingestion.source
    if source is CompetitorSource.SYNTHETIC:
        return RobotsStatus(
            checked=False,
            detail="The synthetic source makes no network requests, so there is "
            "no crawl policy to honour.",
        )

    try:
        scraper = get_scraper(settings)
    except ScraperDisabled as exc:
        return RobotsStatus(checked=False, detail=str(exc))

    probe_url = f"{scraper.base_url}/search"
    try:
        scraper._check_robots(probe_url)
        return RobotsStatus(
            checked=True,
            allowed=True,
            url=probe_url,
            detail=f"robots.txt at {scraper.base_url}/robots.txt permits "
            f"{settings.ingestion.user_agent!r} to fetch this path.",
        )
    except ScraperBlocked as exc:
        # Expected and correct for Booking.com and Expedia: both disallow the
        # search paths their scrapers need. Reported as a fact, not an error.
        return RobotsStatus(checked=True, allowed=False, url=probe_url, detail=str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        return RobotsStatus(
            checked=True,
            allowed=None,
            url=probe_url,
            detail=f"could not determine crawl permission: {type(exc).__name__}: {exc}",
        )
    finally:
        try:
            scraper.close()
        except Exception:  # pragma: no cover
            pass


def _catalogue(session: Session, hotel_ids: Optional[List[str]]) -> List[Tuple[str, str]]:
    """``(hotel_id, city)`` pairs to collect for."""
    query = select(Hotel.hotel_id, Hotel.city).order_by(Hotel.hotel_id)
    if hotel_ids:
        query = query.where(Hotel.hotel_id.in_(hotel_ids))
    pairs = [(row[0], row[1]) for row in session.execute(query).all()]

    if hotel_ids:
        missing = sorted(set(hotel_ids) - {hotel_id for hotel_id, _ in pairs})
        if missing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"unknown hotel(s): {', '.join(missing)}",
            )
    return pairs


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


@router.get(
    "/ingestion/status",
    response_model=IngestionStatusResponse,
    summary="Competitor feed configuration and health",
    description=(
        "Reports which source is configured, whether it reaches the network, "
        "what the target's `robots.txt` says, and how many observations have "
        "actually landed.\n\n"
        "`robots.allowed = false` is a valid healthy state: it means the site "
        "told us not to and the scraper is obeying."
    ),
)
def ingestion_status(
    session: Session = Depends(session_dependency),
    settings: Settings = Depends(settings_dependency),
) -> IngestionStatusResponse:
    """Everything needed to answer "is competitor data flowing, and from where?"."""
    total = session.execute(
        select(func.count()).select_from(CompetitorPrice.__table__)
    ).scalar_one()

    since = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    last_hour = session.execute(
        select(func.count())
        .select_from(CompetitorPrice.__table__)
        .where(CompetitorPrice.collected_at >= since)
    ).scalar_one()

    latest = session.execute(select(func.max(CompetitorPrice.collected_at))).scalar_one()

    by_source = {
        row[0]: row[1]
        for row in session.execute(
            select(CompetitorPrice.source, func.count())
            .group_by(CompetitorPrice.source)
            .order_by(func.count().desc())
        ).all()
    }

    return IngestionStatusResponse(
        source=_source_info(settings),
        robots=_robots_status(settings),
        observations_total=int(total),
        observations_last_hour=int(last_hour),
        latest_observed_at=latest,
        by_source=by_source,
    )


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


@router.post(
    "/ingestion/run",
    response_model=IngestionRunResponse,
    summary="Run one competitor collection pass now",
    description=(
        "Collects competitor rates using the configured source and, unless "
        "`publish` is false, streams each one to Kafka and persists it through "
        "the same handler the consumer uses -- so a rate collected here and the "
        "same rate arriving off the topic are idempotent.\n\n"
        "Synchronous, and paced by `INGESTION_RATE_LIMIT_SECONDS`. A full pass "
        "over every hotel, category and horizon takes rate-limit x request-count "
        "seconds; narrow it with `hotel_ids`, `room_types` and `horizons`."
    ),
    responses={
        404: {"description": "Unknown hotel"},
        409: {"description": "The configured source is disabled"},
        413: {"description": "The requested pass would make too many requests"},
    },
)
def run_ingestion(
    body: IngestionRunRequest,
    session: Session = Depends(session_dependency),
    settings: Settings = Depends(settings_dependency),
    _scope: str = Depends(require_write),
) -> IngestionRunResponse:
    """Collect competitor rates once and report exactly what happened."""
    started = time.perf_counter()

    hotels = _catalogue(session, body.hotel_ids)
    if not hotels:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no hotels in the catalogue; seed the database first",
        )

    room_types: List[RoomType] = list(body.room_types or list(RoomType))
    horizons: List[int] = list(body.horizons or settings.ingestion.horizons)

    planned = len(hotels) * len(room_types) * len(horizons)
    estimated_seconds = planned * settings.ingestion.rate_limit_seconds

    if planned > MAX_REQUESTS_PER_RUN:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"that pass would make {planned} requests (limit "
                f"{MAX_REQUESTS_PER_RUN}). Narrow hotel_ids, room_types or horizons."
            ),
        )
    if estimated_seconds > MAX_RUN_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"that pass would make {planned} requests at "
                f"{settings.ingestion.rate_limit_seconds:.1f}s apart, about "
                f"{estimated_seconds / 60:.0f} minutes -- longer than the "
                f"{MAX_RUN_SECONDS / 60:.0f}-minute limit for a synchronous run. "
                f"Narrow hotel_ids, room_types or horizons, or lower "
                f"INGESTION_RATE_LIMIT_SECONDS."
            ),
        )

    try:
        scraper = get_scraper(settings)
    except ScraperDisabled as exc:
        # 409 rather than 500: the request is well-formed, the server is simply
        # configured not to do this. The message says which settings to change.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    today = date.today()
    requests_made = 0
    failed = 0
    blocked = False
    collected: List = []
    errors: List[str] = []

    try:
        for hotel_id, city in hotels:
            for horizon in horizons:
                for room_type in room_types:
                    request = ScrapeRequest(
                        hotel_id=hotel_id,
                        city=city,
                        room_type=room_type,
                        check_in_date=today + timedelta(days=horizon),
                    )
                    requests_made += 1
                    try:
                        payloads = scraper.collect(request)
                    except ScraperBlocked as exc:
                        # The site said no. Stop the whole pass -- retrying past
                        # a block is how a soft block becomes a permanent one.
                        blocked = True
                        errors.append(f"blocked: {exc}")
                        logger.warning("Collection stopped, source blocked us: %s", exc)
                        raise StopIteration from None
                    except ScraperError as exc:  # pragma: no cover - collect() absorbs these
                        failed += 1
                        errors.append(f"{type(exc).__name__}: {exc}")
                        continue

                    if not payloads:
                        failed += 1
                    collected.extend(payloads)
    except StopIteration:
        pass
    finally:
        scraper.close()

    published = persisted = duplicates = 0

    if body.publish and collected:
        producer = get_producer()
        for payload in collected:
            envelope = EventEnvelope.wrap(payload, source="ingestion-api")
            if producer.send(payload, TopicName.COMPETITOR_PRICES, envelope=envelope):
                published += 1
            try:
                result = handle_competitor_price(envelope, payload, session)
                if result.outcome.value == "duplicate":
                    duplicates += 1
                else:
                    persisted += 1
            except SQLAlchemyError as exc:
                session.rollback()
                errors.append(f"persist failed: {type(exc).__name__}")
                logger.warning("Could not persist scraped rate: %s", exc)
                break
        else:
            session.commit()

    elapsed = time.perf_counter() - started

    observe_ingestion(source=settings.ingestion.source.value, count=len(collected))

    # A pass that collected nothing at all is a failure even when no single
    # request raised: it means every page parsed to zero rates, which is what a
    # silent redesign looks like.
    succeeded = not blocked and bool(collected)

    if blocked:
        detail = "the source refused us; collection stopped early"
    elif not collected:
        detail = (
            "no rates collected. Every request returned empty -- check the "
            "target is reachable and the markup still matches the selectors "
            "(INGESTION_DEMO_OTA_LAYOUT=v2 does exactly this on purpose)."
        )
    elif body.publish:
        detail = (
            f"{len(collected)} rate(s) collected, {persisted} persisted, "
            f"{duplicates} already known"
        )
    else:
        detail = f"{len(collected)} rate(s) collected; nothing published (preview run)"

    return IngestionRunResponse(
        succeeded=succeeded,
        source=settings.ingestion.source.value,
        requests_made=requests_made,
        rates_collected=len(collected),
        failed=failed,
        published=published,
        persisted=persisted,
        duplicates=duplicates,
        blocked=blocked,
        duration_seconds=round(elapsed, 3),
        rates=[
            ScrapedRate(
                hotel_id=p.hotel_id,
                competitor=p.competitor,
                room_type=p.room_type,
                check_in_date=p.check_in_date,
                price=p.price,
                currency=p.currency,
                is_available=p.is_available,
                source=p.source,
            )
            for p in collected[:MAX_RATES_IN_RESPONSE]
        ],
        errors=errors[:20],
        detail=detail,
    )


__all__ = ["router"]
