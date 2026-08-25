"""Event contracts: the envelope, the typed payloads, and their wire format.

Every message on every topic is the same envelope with a different ``payload``::

    {
      "event_id":   "3f9a...",          # uuid4, the idempotency key
      "event_type": "competitor_price",
      "version":    1,
      "timestamp":  "2026-09-15T10:30:00Z",
      "source":     "synthetic",
      "payload":    { ... }             # typed, validated
    }

Three decisions worth defending:

**``version`` from day one.** Adding a schema version after the fact means every
consumer has to guess whether a field's absence is "old producer" or "bad
message". Adding it now costs a single integer and lets a consumer reject or
adapt instead of crashing.

**``event_id`` is the idempotency key.** The consumer commits Kafka offsets only
after the database write, which makes delivery at-least-once: after a rebalance
or a crash, messages *will* be reprocessed. ``competitor_prices.event_id`` is
unique, so a redelivery becomes a rejected insert instead of a duplicated
observation quietly dragging the competitor average around.

**Payloads are Pydantic models, not dicts.** Requirement 8 lists exactly what
must be rejected -- negative prices, invalid dates, unknown room types, missing
fields. Encoding those as types means the rejection happens at the boundary,
once, rather than as scattered ``if`` statements in the consumer.

Structural validation lives here. *Semantic* validation -- "is H042 a hotel we
actually operate" -- needs the database and lives in :mod:`ingestion.validator`.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, field_validator

from database.models import Competitor, RoomType, Season

#: Bumped when a payload changes shape in a way consumers must know about.
SCHEMA_VERSION = 1

#: Furthest ahead a check-in date may legitimately sit. A stay date decades out
#: is a parsing bug or a corrupt feed, not a booking.
MAX_HORIZON_DAYS = 730


class EventDecodeError(ValueError):
    """Raised when bytes on a topic are not a valid, decodable event.

    Deliberately distinct from :class:`pydantic.ValidationError`: the consumer
    treats a malformed message as poison (log, count, skip, commit) rather than
    as a transient failure worth retrying, and it needs to tell the two apart.
    """


class EventType(str, Enum):
    """Discriminator carried in every envelope."""

    COMPETITOR_PRICE = "competitor_price"
    BOOKING = "booking"
    DEMAND_SIGNAL = "demand_signal"
    PRICE_PREDICTION = "price_prediction"


# --------------------------------------------------------------------------- #
# Payloads
# --------------------------------------------------------------------------- #


class EventPayload(BaseModel):
    """Base class for every payload.

    ``extra="forbid"`` is intentional. A producer that starts sending an
    unexpected field is either a version mismatch or a bug, and silently
    dropping it would hide both.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: Which topic this payload belongs on. Set by each subclass.
    event_type: ClassVar[EventType]

    #: Kafka partition key. Keying by hotel keeps one property's events ordered
    #: relative to each other, which is all the ordering the pipeline needs.
    def partition_key(self) -> str:
        return getattr(self, "hotel_id")


class CompetitorPricePayload(EventPayload):
    """One competitor rate observed for one night.

    This is the payload requirement 8 specifies, field for field.
    """

    event_type: ClassVar[EventType] = EventType.COMPETITOR_PRICE

    hotel_id: str = Field(min_length=1, max_length=16)
    competitor: Competitor
    room_type: RoomType
    check_in_date: date
    price: float = Field(gt=0, le=1_000_000)
    currency: str = Field(default="INR", min_length=3, max_length=3)
    is_available: bool = True
    source: str = Field(default="synthetic", max_length=16)

    @field_validator("currency")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.upper()

    @field_validator("check_in_date")
    @classmethod
    def _plausible_horizon(cls, value: date) -> date:
        """Reject dates so far out they can only be a parsing accident."""
        horizon = (value - datetime.now(timezone.utc).date()).days
        if horizon > MAX_HORIZON_DAYS:
            raise ValueError(
                f"check_in_date is {horizon} days ahead; the maximum is "
                f"{MAX_HORIZON_DAYS}"
            )
        return value


class BookingPayload(EventPayload):
    """Reservations and cancellations taken on one day for one night."""

    event_type: ClassVar[EventType] = EventType.BOOKING

    hotel_id: str = Field(min_length=1, max_length=16)
    room_type: RoomType
    booking_date: date
    check_in_date: date
    check_out_date: date
    booking_count: int = Field(ge=0)
    cancellation_count: int = Field(default=0, ge=0)
    revenue: float = Field(default=0.0, ge=0)
    adr: float = Field(default=0.0, ge=0)
    channel: str = Field(default="direct", max_length=24)

    @field_validator("check_out_date")
    @classmethod
    def _stay_is_positive(cls, value: date, info) -> date:
        check_in = info.data.get("check_in_date")
        if check_in is not None and value <= check_in:
            raise ValueError("check_out_date must be after check_in_date")
        return value

    @field_validator("cancellation_count")
    @classmethod
    def _cancellations_within_bookings(cls, value: int, info) -> int:
        booked = info.data.get("booking_count")
        if booked is not None and value > booked:
            raise ValueError(
                f"cancellation_count ({value}) exceeds booking_count ({booked})"
            )
        return value

    @property
    def lead_time_days(self) -> int:
        """Days between the booking and the stay. Never negative by construction."""
        return max((self.check_in_date - self.booking_date).days, 0)


class DemandSignalPayload(EventPayload):
    """Exogenous demand signals for one night: search, weather, events.

    These are facts about the world rather than about our hotel, which is what
    makes them available at serving time for dates that have not happened yet.
    """

    event_type: ClassVar[EventType] = EventType.DEMAND_SIGNAL

    hotel_id: str = Field(min_length=1, max_length=16)
    room_type: RoomType
    stay_date: date
    search_demand: float = Field(ge=0.0, le=1.0)
    weather_score: float = Field(default=0.5, ge=0.0, le=1.0)
    local_event_score: float = Field(default=0.0, ge=0.0, le=1.0)
    holiday_flag: bool = False
    holiday_name: Optional[str] = Field(default=None, max_length=64)
    season: Optional[Season] = None


class PricePredictionPayload(EventPayload):
    """A price the API decided to recommend, published for downstream consumers."""

    event_type: ClassVar[EventType] = EventType.PRICE_PREDICTION

    hotel_id: str = Field(min_length=1, max_length=16)
    room_type: RoomType
    check_in_date: date
    prediction_id: str = Field(min_length=1, max_length=36)
    base_price: float = Field(gt=0)
    raw_recommended_price: float = Field(gt=0)
    final_recommended_price: float = Field(gt=0)
    price_change_percent: float
    blended_demand: float = Field(ge=0.0)
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str = Field(max_length=32)
    guardrails_applied: list = Field(default_factory=list)


#: Discriminator value -> payload class. The consumer uses this to decode a
#: message without knowing which topic it came from.
PAYLOAD_TYPES: Dict[EventType, Type[EventPayload]] = {
    EventType.COMPETITOR_PRICE: CompetitorPricePayload,
    EventType.BOOKING: BookingPayload,
    EventType.DEMAND_SIGNAL: DemandSignalPayload,
    EventType.PRICE_PREDICTION: PricePredictionPayload,
}


# --------------------------------------------------------------------------- #
# Envelope
# --------------------------------------------------------------------------- #


class EventEnvelope(BaseModel):
    """The wire format. Identical on every topic; only ``payload`` differs."""

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType
    version: int = Field(default=SCHEMA_VERSION, ge=1)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    source: str = Field(default="synthetic", max_length=32)
    payload: Dict[str, Any]

    @field_validator("timestamp")
    @classmethod
    def _force_utc(cls, value: datetime) -> datetime:
        """Naive timestamps are interpreted as UTC, aware ones are converted.

        A pricing system that is a day out because someone sent local time is a
        real and expensive failure; normalising at the boundary removes the
        whole class of bug.
        """
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    # -- construction ------------------------------------------------------ #

    @classmethod
    def wrap(
        cls,
        payload: EventPayload,
        *,
        source: str = "synthetic",
        event_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> "EventEnvelope":
        """Build an envelope around a typed payload."""
        return cls(
            event_id=event_id or str(uuid.uuid4()),
            event_type=payload.event_type,
            version=SCHEMA_VERSION,
            timestamp=timestamp or datetime.now(timezone.utc),
            source=source,
            payload=payload.model_dump(mode="json"),
        )

    # -- decoding ---------------------------------------------------------- #

    def decode_payload(self) -> EventPayload:
        """Validate and return the typed payload.

        Raises:
            EventDecodeError: If the discriminator is unknown, the schema
                version is not supported, or the payload fails validation.
        """
        if self.version > SCHEMA_VERSION:
            raise EventDecodeError(
                f"event {self.event_id} declares schema version {self.version}; "
                f"this consumer understands up to {SCHEMA_VERSION}"
            )
        payload_cls = PAYLOAD_TYPES.get(self.event_type)
        if payload_cls is None:  # pragma: no cover - unreachable via the enum
            raise EventDecodeError(f"no payload type for {self.event_type}")
        try:
            return payload_cls(**self.payload)
        except Exception as exc:
            raise EventDecodeError(
                f"payload of event {self.event_id} ({self.event_type.value}) is "
                f"invalid: {exc}"
            ) from exc

    @property
    def partition_key(self) -> Optional[str]:
        """Kafka message key, read without fully decoding the payload."""
        key = self.payload.get("hotel_id")
        return str(key) if key is not None else None

    # -- serialisation ----------------------------------------------------- #

    def to_bytes(self) -> bytes:
        """UTF-8 JSON, ready for the producer."""
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, raw: bytes) -> "EventEnvelope":
        """Decode bytes from a topic.

        Raises:
            EventDecodeError: If the bytes are not UTF-8, not JSON, or not a
                well-formed envelope. The consumer treats this as poison rather
                than as a retryable error.
        """
        try:
            document = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EventDecodeError(f"message is not JSON: {exc}") from exc

        if not isinstance(document, dict):
            raise EventDecodeError(
                f"message is a JSON {type(document).__name__}, expected an object"
            )

        try:
            return cls(**document)
        except Exception as exc:
            raise EventDecodeError(f"message is not a valid envelope: {exc}") from exc


def serialize(payload: EventPayload, *, source: str = "synthetic") -> bytes:
    """Payload -> wire bytes, in one call."""
    return EventEnvelope.wrap(payload, source=source).to_bytes()


def deserialize(raw: bytes) -> tuple[EventEnvelope, EventPayload]:
    """Wire bytes -> (envelope, typed payload).

    Raises:
        EventDecodeError: On any structural problem.
    """
    envelope = EventEnvelope.from_bytes(raw)
    return envelope, envelope.decode_payload()


__all__ = [
    "MAX_HORIZON_DAYS",
    "PAYLOAD_TYPES",
    "SCHEMA_VERSION",
    "BookingPayload",
    "CompetitorPricePayload",
    "DemandSignalPayload",
    "EventDecodeError",
    "EventEnvelope",
    "EventPayload",
    "EventType",
    "PricePredictionPayload",
    "deserialize",
    "serialize",
]
