"""Persistence handlers: one per event type, Kafka message to database row.

Every handler is idempotent, because the consumer is at-least-once. Two
mechanisms, chosen per table:

``competitor_prices`` / ``bookings``
    Insert with ``ON CONFLICT (event_id) DO NOTHING``. A redelivered event
    inserts zero rows and reports :attr:`HandlerOutcome.DUPLICATE`.

``demand_features``
    Upsert on the natural grain ``(hotel_id, room_type, stay_date)``. Demand
    signals are a *current view*, not an append-only log, so the last writer
    should win rather than collide.

Both dialects in use support ``ON CONFLICT``, so the same statement builder
serves PostgreSQL in production and SQLite in the tests. That is the whole
reason the tests can prove idempotency without a broker or a container.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from sqlalchemy import Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from database.models import Booking, CompetitorPrice, DemandFeature
from features.calendars import is_weekend, season_of
from monitoring.logging_config import get_logger
from streaming.events import (
    BookingPayload,
    CompetitorPricePayload,
    DemandSignalPayload,
    EventEnvelope,
    EventPayload,
    EventType,
)

logger = get_logger(__name__)


class HandlerOutcome(str, Enum):
    """What a handler did with an event."""

    WRITTEN = "written"
    DUPLICATE = "duplicate"
    IGNORED = "ignored"


@dataclass(frozen=True)
class HandlerResult:
    outcome: HandlerOutcome
    rows: int = 0


#: A handler takes the envelope, the decoded payload and an open session, and
#: reports what it did. It must not commit -- the consumer owns the transaction
#: boundary, because the offset commit has to happen after it.
Handler = Callable[[EventEnvelope, EventPayload, Session], HandlerResult]


def _insert_ignoring_conflicts(
    session: Session, table: Table, values: Dict[str, Any], conflict_column: str
) -> int:
    """``INSERT ... ON CONFLICT (col) DO NOTHING``, portable across both dialects.

    Returns the number of rows actually inserted: 0 means the event had already
    been processed.
    """
    dialect = session.get_bind().dialect.name
    builder = pg_insert if dialect == "postgresql" else sqlite_insert
    statement = builder(table).values(**values).on_conflict_do_nothing(
        index_elements=[conflict_column]
    )
    return int(session.execute(statement).rowcount or 0)


def _upsert(
    session: Session,
    table: Table,
    values: Dict[str, Any],
    conflict_columns: List[str],
    update_columns: List[str],
) -> int:
    """``INSERT ... ON CONFLICT (cols) DO UPDATE SET ...``, portable."""
    dialect = session.get_bind().dialect.name
    builder = pg_insert if dialect == "postgresql" else sqlite_insert
    statement = builder(table).values(**values)
    statement = statement.on_conflict_do_update(
        index_elements=conflict_columns,
        set_={column: statement.excluded[column] for column in update_columns},
    )
    return int(session.execute(statement).rowcount or 0)


# --------------------------------------------------------------------------- #
# Handlers
# --------------------------------------------------------------------------- #


def handle_competitor_price(
    envelope: EventEnvelope, payload: EventPayload, session: Session
) -> HandlerResult:
    """Persist one observed competitor rate."""
    assert isinstance(payload, CompetitorPricePayload)

    inserted = _insert_ignoring_conflicts(
        session,
        CompetitorPrice.__table__,
        {
            "event_id": envelope.event_id,
            "hotel_id": payload.hotel_id,
            "room_type": payload.room_type.value,
            "competitor": payload.competitor.value,
            "check_in_date": payload.check_in_date,
            "price": payload.price,
            "currency": payload.currency,
            "is_available": payload.is_available,
            "source": payload.source,
            # The observation time is the envelope's, not "now": a message that
            # sat in a topic for an hour still describes the rate as it was when
            # it was scraped.
            "collected_at": envelope.timestamp,
        },
        conflict_column="event_id",
    )
    if inserted == 0:
        return HandlerResult(HandlerOutcome.DUPLICATE)
    return HandlerResult(HandlerOutcome.WRITTEN, inserted)


def handle_booking(
    envelope: EventEnvelope, payload: EventPayload, session: Session
) -> HandlerResult:
    """Persist one pickup row: bookings taken today for a future night."""
    assert isinstance(payload, BookingPayload)

    inserted = _insert_ignoring_conflicts(
        session,
        Booking.__table__,
        {
            "event_id": envelope.event_id,
            "hotel_id": payload.hotel_id,
            "room_type": payload.room_type.value,
            "booking_date": payload.booking_date,
            "check_in_date": payload.check_in_date,
            "check_out_date": payload.check_out_date,
            "booking_count": payload.booking_count,
            "cancellation_count": payload.cancellation_count,
            "revenue": payload.revenue,
            "adr": payload.adr,
            "lead_time_days": payload.lead_time_days,
            "channel": payload.channel,
        },
        conflict_column="event_id",
    )
    if inserted == 0:
        return HandlerResult(HandlerOutcome.DUPLICATE)
    return HandlerResult(HandlerOutcome.WRITTEN, inserted)


def handle_demand_signal(
    envelope: EventEnvelope, payload: EventPayload, session: Session
) -> HandlerResult:
    """Upsert the exogenous demand signals for one night.

    Only the exogenous half of the row is touched. The derived features and the
    target belong to the Phase 4 pipeline and must not be clobbered by a signal
    update -- which is exactly why the update column list is explicit rather
    than "every column in the payload".

    The calendar columns are recomputed from :mod:`features.calendars` rather
    than trusted from the message. A producer that disagrees with us about which
    day is a holiday would otherwise poison the feature store.
    """
    assert isinstance(payload, DemandSignalPayload)

    season = season_of(payload.stay_date)
    values = {
        "hotel_id": payload.hotel_id,
        "room_type": payload.room_type.value,
        "stay_date": payload.stay_date,
        "day_of_week": payload.stay_date.weekday(),
        "is_weekend": is_weekend(payload.stay_date),
        "season": season.value,
        "holiday_flag": payload.holiday_flag,
        "holiday_name": payload.holiday_name,
        "local_event_score": payload.local_event_score,
        "weather_score": payload.weather_score,
        "search_demand": payload.search_demand,
    }

    rows = _upsert(
        session,
        DemandFeature.__table__,
        values,
        conflict_columns=["hotel_id", "room_type", "stay_date"],
        update_columns=[
            "local_event_score",
            "weather_score",
            "search_demand",
            "holiday_flag",
            "holiday_name",
        ],
    )
    return HandlerResult(HandlerOutcome.WRITTEN, rows)


def handle_price_prediction(
    envelope: EventEnvelope, payload: EventPayload, session: Session
) -> HandlerResult:
    """Predictions are published *by* the API, which already persisted them.

    The topic exists so downstream systems (the dashboard, a channel manager, a
    data warehouse) can subscribe. Re-inserting them here would duplicate rows
    the API wrote a millisecond earlier, so this consumer deliberately ignores
    them.
    """
    return HandlerResult(HandlerOutcome.IGNORED)


#: The routing table the consumer dispatches on.
DEFAULT_HANDLERS: Dict[EventType, Handler] = {
    EventType.COMPETITOR_PRICE: handle_competitor_price,
    EventType.BOOKING: handle_booking,
    EventType.DEMAND_SIGNAL: handle_demand_signal,
    EventType.PRICE_PREDICTION: handle_price_prediction,
}


def build_handlers(
    overrides: Optional[Dict[EventType, Handler]] = None
) -> Dict[EventType, Handler]:
    """The default routing table, with per-type overrides for tests."""
    handlers = dict(DEFAULT_HANDLERS)
    if overrides:
        handlers.update(overrides)
    return handlers


__all__ = [
    "DEFAULT_HANDLERS",
    "Handler",
    "HandlerOutcome",
    "HandlerResult",
    "build_handlers",
    "handle_booking",
    "handle_competitor_price",
    "handle_demand_signal",
    "handle_price_prediction",
]
