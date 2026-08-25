"""Competitor rate endpoints: read the market, and submit an observation.

``POST /competitors/events`` is the manual door into the same pipeline the
scrapers use. The event is validated with the *same* Pydantic payload the Kafka
consumer validates, published to ``hotel.competitor_prices``, and persisted --
in that order, and with the persistence happening whether or not Kafka is
reachable.

That ordering is the point. A rate submitted through the API must be
indistinguishable downstream from one collected by the synthetic generator or a
scraper; if the HTTP path had its own validation and its own write, the two
would drift and only one of them would be tested.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, Path, Query, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.dependencies import require_hotel, require_room, session_dependency
from api.security import require_write
from api.schemas import (
    MAX_PAGE_SIZE,
    CompetitorEventRequest,
    CompetitorEventResponse,
    CompetitorPriceItem,
    CompetitorResponse,
    CompetitorSummary,
)
from database.models import CompetitorPrice, RoomType
from monitoring.logging_config import get_logger
from streaming.events import CompetitorPricePayload, EventEnvelope
from streaming.handlers import handle_competitor_price
from streaming.producer import get_producer
from streaming.topics import TopicName

logger = get_logger(__name__)

router = APIRouter(tags=["competitors"])

#: Default window for the read endpoint: the next month of stay dates.
DEFAULT_LOOKAHEAD_DAYS = 30


@router.get(
    "/competitors/{hotel_id}",
    response_model=CompetitorResponse,
    summary="Competitor rates around a hotel",
    description=(
        "Observed competitor rates for upcoming stay dates, with a per-night "
        "summary. `spread_percent` is the width of the competitive set relative "
        "to its mean -- a wide spread means weak price discipline in the market "
        "and more room to move."
    ),
    responses={404: {"description": "Unknown hotel"}},
)
def get_competitors(
    hotel_id: str = Path(examples=["H001"]),
    room_type: Optional[RoomType] = Query(default=None),
    start_date: Optional[date] = Query(default=None),
    end_date: Optional[date] = Query(default=None),
    limit: int = Query(default=200, ge=1, le=MAX_PAGE_SIZE),
    session: Session = Depends(session_dependency),
) -> CompetitorResponse:
    """Return competitor observations and a per-night summary."""
    require_hotel(session, hotel_id)

    start = start_date or datetime.now(timezone.utc).date()
    end = end_date or (start + timedelta(days=DEFAULT_LOOKAHEAD_DAYS))

    statement = (
        select(CompetitorPrice)
        .where(
            CompetitorPrice.hotel_id == hotel_id,
            CompetitorPrice.check_in_date >= start,
            CompetitorPrice.check_in_date <= end,
        )
        .order_by(CompetitorPrice.check_in_date, CompetitorPrice.collected_at.desc())
        .limit(limit)
    )
    if room_type is not None:
        statement = statement.where(CompetitorPrice.room_type == room_type)

    rows = session.execute(statement).scalars().all()

    observations = [
        CompetitorPriceItem(
            competitor=row.competitor,
            room_type=row.room_type,
            check_in_date=row.check_in_date,
            price=float(row.price),
            currency=row.currency,
            is_available=row.is_available,
            source=row.source,
            collected_at=row.collected_at,
        )
        for row in rows
    ]

    # Aggregate per (night, room type). Only the freshest observation per
    # competitor counts -- an average over three weeks of history is not a
    # description of today's market.
    freshest: dict = {}
    for row in rows:
        key = (row.check_in_date, row.room_type, row.competitor)
        existing = freshest.get(key)
        if existing is None or row.collected_at > existing.collected_at:
            freshest[key] = row

    grouped: dict = {}
    for (check_in, room, _competitor), row in freshest.items():
        grouped.setdefault((check_in, room), []).append(float(row.price))

    summaries: List[CompetitorSummary] = []
    for (check_in, room), prices in sorted(grouped.items(), key=lambda item: item[0][0]):
        mean = sum(prices) / len(prices)
        summaries.append(
            CompetitorSummary(
                check_in_date=check_in,
                room_type=room,
                competitor_rate=round(mean, 2),
                competitor_min_rate=round(min(prices), 2),
                competitor_max_rate=round(max(prices), 2),
                competitor_count=len(prices),
                spread_percent=round((max(prices) - min(prices)) / mean * 100.0, 2),
            )
        )

    return CompetitorResponse(
        hotel_id=hotel_id,
        count=len(observations),
        summaries=summaries,
        observations=observations,
    )


@router.post(
    "/competitors/events",
    response_model=CompetitorEventResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit an observed competitor rate",
    description=(
        "Validates the observation, publishes it to `hotel.competitor_prices` "
        "and persists it. The rate is stored even when Kafka is unavailable, so "
        "a broker outage delays the stream rather than losing the data.\n\n"
        "Returns 202: the observation is accepted for processing, and the "
        "feature pipeline will pick it up on its next run."
    ),
    responses={404: {"description": "Unknown hotel or room type"}},
)
def submit_competitor_event(
    body: CompetitorEventRequest,
    session: Session = Depends(session_dependency),
    _scope: str = Depends(require_write),
) -> CompetitorEventResponse:
    """Accept one competitor rate observation."""
    require_hotel(session, body.hotel_id)
    require_room(session, body.hotel_id, body.room_type)

    # The same payload type the Kafka consumer validates. One definition of
    # "valid competitor rate" for every entry point.
    payload = CompetitorPricePayload(
        hotel_id=body.hotel_id,
        competitor=body.competitor,
        room_type=body.room_type,
        check_in_date=body.check_in_date,
        price=body.price,
        currency=body.currency,
        is_available=body.is_available,
        source=body.source,
    )
    envelope = EventEnvelope.wrap(payload, source="api")

    published = get_producer().send(
        payload, TopicName.COMPETITOR_PRICES, envelope=envelope
    )

    # Persisted through the same handler the consumer uses, so an event that
    # arrives twice -- once here, once off the topic -- is idempotent.
    persisted = False
    detail = "accepted"
    try:
        result = handle_competitor_price(envelope, payload, session)
        session.commit()
        persisted = True
        if result.outcome.value == "duplicate":
            detail = "already recorded"
    except SQLAlchemyError as exc:
        session.rollback()
        detail = f"published but not persisted: {type(exc).__name__}"
        logger.warning("Could not persist competitor event %s: %s", envelope.event_id, exc)

    if not published:
        detail += "; Kafka unavailable, the event was not streamed"

    return CompetitorEventResponse(
        accepted=True,
        event_id=envelope.event_id,
        published_to_kafka=published,
        persisted=persisted,
        detail=detail,
    )


__all__ = ["router"]
