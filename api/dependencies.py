"""FastAPI dependencies: sessions, the model registry, the pricing engines.

Everything expensive is process-wide and injected. Loading a Prophet bundle
takes about a second; doing it per request would make the API's latency the
model's load time. Everything cheap is constructed per request so it cannot
accumulate state.

The dependency functions are also the seam the tests use: overriding
:func:`get_registry_dependency` with a stub is how the route tests run without
any artifacts on disk.
"""

from __future__ import annotations

from typing import Iterator

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from config import Settings, get_settings
from database.connection import get_db
from database.models import Hotel, Room, RoomType
from models.model_registry import ModelRegistry, get_registry
from monitoring.logging_config import get_logger
from pricing.demand_engine import DemandEngine
from pricing.pricing_engine import PricingEngine

logger = get_logger(__name__)


def settings_dependency() -> Settings:
    """The settings singleton."""
    return get_settings()


def session_dependency() -> Iterator[Session]:
    """A request-scoped database session."""
    yield from get_db()


def get_registry_dependency() -> ModelRegistry:
    """The process-wide model registry, loaded on first use.

    Loading is lazy rather than mandatory at import so that the module can be
    imported -- and the API can start -- before anything has been trained.
    """
    registry = get_registry()
    registry.ensure_loaded()
    return registry


def demand_engine_dependency(
    registry: ModelRegistry = Depends(get_registry_dependency),
    settings: Settings = Depends(settings_dependency),
) -> DemandEngine:
    """A demand engine bound to whichever models are currently loaded.

    Constructed per request but holding references to the shared models, so it
    always reflects the latest reload without anything having to invalidate a
    cache.
    """
    loaded = registry.loaded
    return DemandEngine(
        prophet_bundle=loaded.prophet, gbr_model=loaded.gbr, settings=settings
    )


def pricing_engine_dependency(
    settings: Settings = Depends(settings_dependency),
) -> PricingEngine:
    """The pricing engine. Stateless, so construction is free."""
    return PricingEngine(settings)


# --------------------------------------------------------------------------- #
# Lookups shared by several routes
# --------------------------------------------------------------------------- #


def require_hotel(session: Session, hotel_id: str) -> Hotel:
    """Fetch a hotel or raise a 404 naming it.

    Raises:
        HTTPException: 404 when the hotel does not exist.
    """
    hotel = session.query(Hotel).filter(Hotel.hotel_id == hotel_id).one_or_none()
    if hotel is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No hotel with id {hotel_id!r}",
        )
    return hotel


def require_room(session: Session, hotel_id: str, room_type: RoomType) -> Room:
    """Fetch a room category or raise a 404 listing what the hotel does sell.

    The message matters: "unknown room type" is unhelpful, "this hotel sells
    standard, deluxe, premium" tells the caller what to send instead.
    """
    room = (
        session.query(Room)
        .filter(Room.hotel_id == hotel_id, Room.room_type == room_type)
        .one_or_none()
    )
    if room is None:
        available = [
            r.room_type.value if isinstance(r.room_type, RoomType) else str(r.room_type)
            for r in session.query(Room).filter(Room.hotel_id == hotel_id).all()
        ]
        if not available:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No hotel with id {hotel_id!r}",
            )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Hotel {hotel_id} does not sell "
                f"{room_type.value if isinstance(room_type, RoomType) else room_type!r}. "
                f"It sells: {', '.join(sorted(available))}"
            ),
        )
    return room


__all__ = [
    "demand_engine_dependency",
    "get_registry_dependency",
    "pricing_engine_dependency",
    "require_hotel",
    "require_room",
    "session_dependency",
    "settings_dependency",
]
