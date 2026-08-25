"""Semantic validation of inbound events.

:mod:`streaming.events` already rejects everything that can be judged from the
message alone -- negative prices, malformed dates, unknown room types, missing
fields. What it cannot judge is whether the event refers to something that
*exists*: ``H042`` is a perfectly well-formed hotel id, and we do not operate a
hotel H042.

That check needs the database, which is why it lives here rather than in the
event models. The reference data barely changes, so it is cached and refreshed
on a timer instead of being re-read per message -- at a few hundred events a
second, a per-message SELECT would be the pipeline's bottleneck and the only
thing it would ever discover is "still the same eight hotels".

The two layers together implement requirement 8: reject negative prices, invalid
dates, unknown hotels, invalid room types and missing required fields.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Optional, Set, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.connection import session_scope
from database.models import Hotel, Room, RoomType
from monitoring.logging_config import get_logger
from streaming.events import EventPayload

logger = get_logger(__name__)

#: How long the reference-data cache stays warm. Long enough that validation is
#: effectively free, short enough that a newly seeded hotel starts being
#: accepted without a restart.
CACHE_TTL_SECONDS = 300.0


class EventRejected(ValueError):
    """An event is structurally valid but refers to something that does not exist.

    Carries the rejection ``reason`` as a stable machine-readable code so the
    monitoring layer can count rejection kinds rather than parse messages.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass
class ValidationStats:
    """Counters exposed to monitoring."""

    accepted: int = 0
    rejected: int = 0
    by_reason: Dict[str, int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.by_reason is None:
            self.by_reason = {}

    def record_accept(self) -> None:
        self.accepted += 1

    def record_reject(self, reason: str) -> None:
        self.rejected += 1
        self.by_reason[reason] = self.by_reason.get(reason, 0) + 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "by_reason": dict(self.by_reason),
        }


class ReferenceData:
    """Cached view of which hotels and room types exist.

    Thread-safety note: the cache is replaced by a single assignment of an
    immutable frozenset, so a concurrent reader sees either the old set or the
    new one, never a half-built one. No lock needed.
    """

    def __init__(self, *, ttl_seconds: float = CACHE_TTL_SECONDS) -> None:
        self.ttl_seconds = ttl_seconds
        self._hotels: FrozenSet[str] = frozenset()
        self._rooms: FrozenSet[Tuple[str, str]] = frozenset()
        self._loaded_at: float = 0.0
        #: How many times the cache has been reloaded. Exposed for monitoring,
        #: and the only reliable way to assert a refresh happened: the monotonic
        #: clock on Windows has ~15ms granularity, so two refreshes inside one
        #: tick share a timestamp.
        self.refresh_count = 0

    @property
    def is_stale(self) -> bool:
        # ``>=`` rather than ``>`` so that ttl_seconds=0 means "always reload"
        # instead of "reload only once the coarse clock happens to tick".
        return (time.monotonic() - self._loaded_at) >= self.ttl_seconds

    def refresh(self, session: Optional[Session] = None) -> None:
        """Reload hotel ids and (hotel, room type) pairs from the database."""
        if session is not None:
            self._load(session)
            return
        with session_scope() as owned:
            self._load(owned)

    def _load(self, session: Session) -> None:
        hotels: Set[str] = set(session.execute(select(Hotel.hotel_id)).scalars())
        rooms: Set[Tuple[str, str]] = {
            (hotel_id, room_type.value if isinstance(room_type, RoomType) else str(room_type))
            for hotel_id, room_type in session.execute(
                select(Room.hotel_id, Room.room_type)
            ).all()
        }
        self._hotels = frozenset(hotels)
        self._rooms = frozenset(rooms)
        self._loaded_at = time.monotonic()
        self.refresh_count += 1
        logger.info(
            "Reference data loaded: %d hotel(s), %d hotel/room pair(s)",
            len(self._hotels),
            len(self._rooms),
        )

    def ensure_fresh(self, session: Optional[Session] = None) -> None:
        """Refresh if the cache has expired or was never loaded."""
        if self.is_stale:
            self.refresh(session)

    def has_hotel(self, hotel_id: str) -> bool:
        return hotel_id in self._hotels

    def has_room(self, hotel_id: str, room_type: str) -> bool:
        return (hotel_id, room_type) in self._rooms

    @property
    def hotel_ids(self) -> FrozenSet[str]:
        return self._hotels

    @property
    def is_empty(self) -> bool:
        """True when no reference data exists -- an unseeded database."""
        return not self._hotels


class EventValidator:
    """Checks that an event refers to hotels and rooms that exist.

    Example::

        validator = EventValidator()
        try:
            validator.validate(payload, session)
        except EventRejected as exc:
            logger.warning("dropped: %s", exc.reason)
    """

    def __init__(
        self,
        *,
        reference: Optional[ReferenceData] = None,
        strict_when_empty: bool = False,
    ) -> None:
        """
        Args:
            reference: Cache override, mostly for tests.
            strict_when_empty: What to do when the database holds no hotels at
                all. The default (``False``) accepts events, because the usual
                cause is "the consumer started before the seeder finished" and
                rejecting the whole stream over a start-up race is worse than
                letting the foreign keys catch it. Set ``True`` in production,
                where an empty reference table means something is badly wrong.
        """
        self.reference = reference or ReferenceData()
        self.strict_when_empty = strict_when_empty
        self.stats = ValidationStats()

    def validate(self, payload: EventPayload, session: Optional[Session] = None) -> None:
        """Raise :class:`EventRejected` if the event cannot be persisted.

        Args:
            payload: A structurally valid payload.
            session: Reuse the consumer's session for the refresh, so validation
                does not open a second connection mid-transaction.
        """
        self.reference.ensure_fresh(session)

        hotel_id = getattr(payload, "hotel_id", None)
        if not hotel_id:
            self._reject("missing_hotel_id", type(payload).__name__)

        if self.reference.is_empty:
            if self.strict_when_empty:
                self._reject("no_reference_data", "the hotels table is empty")
            # Permissive path: the foreign keys are still the backstop.
            self.stats.record_accept()
            return

        if not self.reference.has_hotel(hotel_id):
            self._reject("unknown_hotel", hotel_id)

        room_type = getattr(payload, "room_type", None)
        if room_type is not None:
            value = room_type.value if isinstance(room_type, RoomType) else str(room_type)
            if not self.reference.has_room(hotel_id, value):
                self._reject("unknown_room_type", f"{hotel_id}/{value}")

        self.stats.record_accept()

    def _reject(self, reason: str, detail: str) -> None:
        self.stats.record_reject(reason)
        raise EventRejected(reason, detail)

    def is_valid(self, payload: EventPayload, session: Optional[Session] = None) -> bool:
        """Boolean form, for callers that do not want to handle an exception."""
        try:
            self.validate(payload, session)
            return True
        except EventRejected:
            return False


__all__ = [
    "CACHE_TTL_SECONDS",
    "EventRejected",
    "EventValidator",
    "ReferenceData",
    "ValidationStats",
]
