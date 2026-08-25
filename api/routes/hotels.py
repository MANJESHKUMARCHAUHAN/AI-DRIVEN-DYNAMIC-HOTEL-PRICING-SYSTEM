"""Hotel and room reference endpoints.

Read-only, and the place a caller starts: they need a hotel id and a room type
before they can ask for a price. The detail endpoint also returns the last
thirty days of trading, because "what is this property actually doing" is the
first question anyone asks after "does it exist".
"""

from __future__ import annotations

from datetime import timedelta
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, Path, Query
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.dependencies import require_hotel, session_dependency
from api.schemas import MAX_PAGE_SIZE, HotelDetail, HotelSummary, RoomSummary
from database.models import Booking, Hotel, Room
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["hotels"])

#: Trading window reported on the detail endpoint.
PERFORMANCE_WINDOW_DAYS = 30


def _to_summary(hotel: Hotel) -> HotelSummary:
    return HotelSummary(
        hotel_id=hotel.hotel_id,
        hotel_name=hotel.hotel_name,
        city=hotel.city,
        country=hotel.country,
        star_rating=hotel.star_rating,
        total_rooms=hotel.total_rooms,
        segment=hotel.segment.value if hasattr(hotel.segment, "value") else str(hotel.segment),
        currency=hotel.currency,
        is_active=hotel.is_active,
    )


def _recent_performance(session: Session, hotel_id: str, total_rooms: int) -> dict:
    """Occupancy, ADR and RevPAR over the last completed nights.

    RevPAR -- revenue per *available* room -- is included because it is the
    number hotels actually manage against. Occupancy alone rewards giving rooms
    away and ADR alone rewards an empty hotel with one expensive suite sold;
    their product is the only one of the three that can be gamed in neither
    direction.
    """
    window_end = session.execute(
        select(func.max(Booking.check_in_date)).where(Booking.hotel_id == hotel_id)
    ).scalar_one_or_none()
    if window_end is None or not total_rooms:
        return {}

    window_start = window_end - timedelta(days=PERFORMANCE_WINDOW_DAYS)

    sold, revenue, nights = session.execute(
        select(
            func.sum(Booking.booking_count - Booking.cancellation_count),
            func.sum(Booking.revenue),
            func.count(func.distinct(Booking.check_in_date)),
        ).where(
            Booking.hotel_id == hotel_id,
            Booking.check_in_date > window_start,
            Booking.check_in_date <= window_end,
        )
    ).one()

    if not sold or not nights:
        return {}

    room_nights_available = total_rooms * int(nights)
    occupancy = float(sold) / room_nights_available
    adr = float(revenue) / float(sold) if sold else 0.0

    return {
        "occupancy_last_30_days": round(occupancy, 4),
        "adr_last_30_days": round(adr, 2),
        "revpar_last_30_days": round(occupancy * adr, 2),
    }


def _performance_for_all(session: Session, hotels: List[Hotel]) -> Dict[str, dict]:
    """Trading figures for many hotels in two queries rather than 2N.

    The per-hotel path (:func:`_recent_performance`) issues two queries each, so
    the Overview page's eight properties cost sixteen round trips plus eight HTTP
    calls -- the classic N+1, and the most expensive thing the dashboard did.

    This groups by ``hotel_id`` instead. Each hotel keeps its **own** window end,
    because a property with no recent bookings must not have its occupancy
    computed against another property's date range -- that would silently make a
    quiet hotel look busy.
    """
    if not hotels:
        return {}

    hotel_ids = [hotel.hotel_id for hotel in hotels]
    rooms_by_id = {hotel.hotel_id: hotel.total_rooms for hotel in hotels}

    window_ends = {
        hotel_id: end
        for hotel_id, end in session.execute(
            select(Booking.hotel_id, func.max(Booking.check_in_date))
            .where(Booking.hotel_id.in_(hotel_ids))
            .group_by(Booking.hotel_id)
        ).all()
        if end is not None
    }
    if not window_ends:
        return {}

    # One pass over the union of every hotel's window, then filtered per hotel.
    # A single OR-ed predicate per hotel would be correct too and produces a
    # query whose size grows with the catalogue.
    earliest = min(window_ends.values()) - timedelta(days=PERFORMANCE_WINDOW_DAYS)

    rows = session.execute(
        select(
            Booking.hotel_id,
            Booking.check_in_date,
            func.sum(Booking.booking_count - Booking.cancellation_count),
            func.sum(Booking.revenue),
        )
        .where(
            Booking.hotel_id.in_(list(window_ends)),
            Booking.check_in_date > earliest,
        )
        .group_by(Booking.hotel_id, Booking.check_in_date)
    ).all()

    totals: Dict[str, List[float]] = {}
    for hotel_id, check_in, sold, revenue in rows:
        end = window_ends[hotel_id]
        if not (end - timedelta(days=PERFORMANCE_WINDOW_DAYS) < check_in <= end):
            continue
        bucket = totals.setdefault(hotel_id, [0.0, 0.0, 0.0])
        bucket[0] += float(sold or 0)
        bucket[1] += float(revenue or 0)
        bucket[2] += 1  # distinct nights, one row per date by construction

    performance: Dict[str, dict] = {}
    for hotel_id, (sold, revenue, nights) in totals.items():
        total_rooms = rooms_by_id.get(hotel_id) or 0
        if not sold or not nights or not total_rooms:
            continue
        occupancy = sold / (total_rooms * nights)
        adr = revenue / sold
        performance[hotel_id] = {
            "occupancy_last_30_days": round(occupancy, 4),
            "adr_last_30_days": round(adr, 2),
            "revpar_last_30_days": round(occupancy * adr, 2),
        }
    return performance


@router.get(
    "/hotels",
    response_model=List[HotelSummary],
    summary="List hotels",
    description="Every property the pricing engine covers. Filter by city or "
    "star rating to narrow the list.\n\n"
    "Set `include_performance=true` to get 30-day occupancy, ADR and RevPAR in "
    "the same response. The Overview dashboard uses it to render the estate in "
    "one call instead of one per hotel.",
)
def list_hotels(
    city: Optional[str] = Query(default=None, examples=["Mumbai"]),
    star_rating: Optional[int] = Query(default=None, ge=1, le=5),
    active_only: bool = Query(default=True),
    include_performance: bool = Query(
        default=False,
        description="Include 30-day occupancy, ADR and RevPAR. Two extra "
        "aggregate queries for the whole list, not per hotel.",
    ),
    limit: int = Query(default=100, ge=1, le=MAX_PAGE_SIZE),
    session: Session = Depends(session_dependency),
) -> List[HotelSummary]:
    """Return the hotel catalogue.

    Performance is opt-in rather than always-on: most callers want the catalogue
    to fill a dropdown, and making every one of them pay for two aggregate
    queries to populate fields they ignore is the other way to get this wrong.
    """
    statement = select(Hotel).order_by(Hotel.hotel_id).limit(limit)
    if city:
        # Case-insensitive so "mumbai" works as well as "Mumbai".
        statement = statement.where(func.lower(Hotel.city) == city.lower())
    if star_rating is not None:
        statement = statement.where(Hotel.star_rating == star_rating)
    if active_only:
        statement = statement.where(Hotel.is_active.is_(True))

    hotels = list(session.execute(statement).scalars())
    summaries = [_to_summary(hotel) for hotel in hotels]

    if include_performance:
        performance = _performance_for_all(session, hotels)
        for summary in summaries:
            for field, value in performance.get(summary.hotel_id, {}).items():
                setattr(summary, field, value)

    return summaries


@router.get(
    "/hotels/{hotel_id}",
    response_model=HotelDetail,
    summary="Hotel detail with rooms and recent trading",
    responses={404: {"description": "Unknown hotel"}},
)
def get_hotel(
    hotel_id: str = Path(examples=["H001"]),
    session: Session = Depends(session_dependency),
) -> HotelDetail:
    """Return one hotel, its room inventory and its last 30 days of trading."""
    hotel = require_hotel(session, hotel_id)

    rooms = session.execute(
        select(Room).where(Room.hotel_id == hotel_id).order_by(Room.base_price)
    ).scalars().all()

    # The trading fields are excluded from the spread because they are supplied
    # below. `_to_summary` returns them as None now that HotelSummary carries
    # them, and splatting both sources is "multiple values for keyword argument".
    summary = _to_summary(hotel).model_dump(
        exclude={"occupancy_last_30_days", "adr_last_30_days", "revpar_last_30_days"}
    )

    return HotelDetail(
        **summary,
        rooms=[
            RoomSummary(
                room_id=room.room_id,
                room_type=room.room_type,
                capacity=room.capacity,
                room_count=room.room_count,
                base_price=float(room.base_price),
                floor_price=float(room.floor_price) if room.floor_price else None,
                ceiling_price=float(room.ceiling_price) if room.ceiling_price else None,
            )
            for room in rooms
        ],
        **_recent_performance(session, hotel_id, hotel.total_rooms),
    )


__all__ = ["router"]
