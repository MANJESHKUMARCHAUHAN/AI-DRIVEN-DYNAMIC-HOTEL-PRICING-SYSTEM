"""SQLAlchemy ORM models -- the nine tables described in docs/architecture.md §8.

Design rules applied throughout:

**Surrogate + natural keys.** Every table has an integer surrogate primary key,
and the tables that model real-world entities also carry a stable business key
(``hotels.hotel_id = "H001"``). Joins and foreign keys use the business key
where the business key is what the outside world sends us, because an API caller
knows ``H001`` and has never heard of row 47.

**Portable enums.** Enumerations are stored as ``VARCHAR`` with a ``CHECK``
constraint (``native_enum=False``) rather than as PostgreSQL ``ENUM`` types. Two
reasons: adding a value to a native enum requires DDL and a migration, and the
whole schema stays creatable on SQLite, which is what lets the model tests run in
milliseconds without a container.

**Money as NUMERIC, read as float.** ``Numeric(12, 2)`` stores prices exactly --
no binary floating point drift in a revenue figure. ``asdecimal=False`` converts
on read, so the ML and pricing layers work with plain floats and never have to
juggle ``Decimal`` against ``numpy.float64``.

**Composite foreign keys.** Fact tables key on ``(hotel_id, room_type)`` and that
pair is a real foreign key into ``rooms``. A booking for a room type the hotel
does not sell is rejected by the database, not by a code review.

**UTC everywhere.** Every timestamp column is ``TIMESTAMP(timezone=True)``.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Date,
    Enum,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TIMESTAMP

# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #

#: Deterministic constraint names. Without this, SQLAlchemy lets the database
#: invent names for CHECK/UNIQUE/FK constraints, and a migration tool can then
#: neither find nor drop them. Naming them here makes the schema diffable.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base carrying the shared :class:`MetaData`."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    def to_dict(self) -> Dict[str, Any]:
        """Column values as a plain dict. Handy for logging and CSV export."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


# --------------------------------------------------------------------------- #
# Reusable column types
# --------------------------------------------------------------------------- #

#: Currency amounts. Exact in the database, float in Python.
MONEY = Numeric(12, 2, asdecimal=False)

#: JSON payloads. ``JSONB`` on PostgreSQL (indexable, binary), plain ``JSON``
#: elsewhere so the same models work on SQLite.
JSON_DICT = JSON().with_variant(JSONB(), "postgresql")

#: Timezone-aware timestamp. Every temporal column in the schema uses this.
TS = TIMESTAMP(timezone=True)


def _enum(enum_cls: type, name: str) -> Enum:
    """Portable enum column: VARCHAR + CHECK, keyed on the *value* not the name."""
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        length=32,
        validate_strings=True,
        values_callable=lambda e: [member.value for member in e],
    )


# --------------------------------------------------------------------------- #
# Domain enumerations
# --------------------------------------------------------------------------- #
#
# Defined in `domain/enums.py` and re-exported here.
#
# They used to be declared in this module, which meant `pricing/` -- the one
# package that must stay free of persistence concerns -- imported the
# persistence module purely to say what a room type is. Nothing broke at
# runtime, but it made "pricing depends on the database layer" true in the
# import graph.
#
# Moving them one layer down fixed the direction rather than weakening the rule:
# `database/` and `pricing/` now both depend on `domain/`, which depends on
# nothing. Re-exported so `from database.models import RoomType` still works --
# one definition, two reachable paths.

from domain.enums import (  # noqa: E402
    Competitor,
    MarketSegment,
    ModelType,
    RoomType,
    RunStatus,
    Season,
)


# --------------------------------------------------------------------------- #
# Mixins
# --------------------------------------------------------------------------- #


class TimestampMixin:
    """``created_at`` / ``updated_at`` maintained by the database clock.

    ``server_default``/``onupdate`` rather than Python defaults: the database
    clock is the single source of truth, so rows written by a bulk COPY, a
    migration or a different service all get consistent timestamps.
    """

    created_at: Mapped[datetime] = mapped_column(
        TS, server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        TS, server_default=func.now(), onupdate=func.now(), nullable=False
    )


# --------------------------------------------------------------------------- #
# Core entities
# --------------------------------------------------------------------------- #


class Hotel(TimestampMixin, Base):
    """A property. The root of the reference-data tree."""

    __tablename__ = "hotels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    #: Business key used by the API, Kafka message keys and the dashboard.
    hotel_id: Mapped[str] = mapped_column(String(16), nullable=False, unique=True)

    hotel_name: Mapped[str] = mapped_column(String(128), nullable=False)
    city: Mapped[str] = mapped_column(String(64), nullable=False)
    country: Mapped[str] = mapped_column(String(64), nullable=False, default="India")
    star_rating: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    total_rooms: Mapped[int] = mapped_column(Integer, nullable=False)
    segment: Mapped[MarketSegment] = mapped_column(
        _enum(MarketSegment, "market_segment"),
        nullable=False,
        default=MarketSegment.MIXED,
    )
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")
    latitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    longitude: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    rooms: Mapped[List["Room"]] = relationship(
        back_populates="hotel", cascade="all, delete-orphan", lazy="selectin"
    )

    __table_args__ = (
        CheckConstraint("star_rating BETWEEN 1 AND 5", name="star_rating_range"),
        CheckConstraint("total_rooms > 0", name="total_rooms_positive"),
        Index("ix_hotels_city", "city"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Hotel {self.hotel_id} {self.hotel_name!r} {self.city}>"


class Room(TimestampMixin, Base):
    """One sellable room category within a hotel.

    ``(hotel_id, room_type)`` is unique, which is what allows every fact table to
    hold a real composite foreign key back to this row.
    """

    __tablename__ = "rooms"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    room_id: Mapped[str] = mapped_column(String(24), nullable=False, unique=True)

    hotel_id: Mapped[str] = mapped_column(
        String(16),
        ForeignKey("hotels.hotel_id", ondelete="CASCADE"),
        nullable=False,
    )
    room_type: Mapped[RoomType] = mapped_column(_enum(RoomType, "room_type"), nullable=False)

    capacity: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=2)
    room_count: Mapped[int] = mapped_column(Integer, nullable=False)
    base_price: Mapped[float] = mapped_column(MONEY, nullable=False)

    #: Optional per-room guardrail overrides. When NULL the global MIN_PRICE /
    #: MAX_PRICE from configuration applies.
    floor_price: Mapped[Optional[float]] = mapped_column(MONEY, nullable=True)
    ceiling_price: Mapped[Optional[float]] = mapped_column(MONEY, nullable=True)

    hotel: Mapped["Hotel"] = relationship(back_populates="rooms")

    __table_args__ = (
        UniqueConstraint("hotel_id", "room_type", name="uq_rooms_hotel_room_type"),
        CheckConstraint("room_count > 0", name="room_count_positive"),
        CheckConstraint("base_price > 0", name="base_price_positive"),
        CheckConstraint("capacity > 0", name="capacity_positive"),
        CheckConstraint(
            "floor_price IS NULL OR ceiling_price IS NULL "
            "OR floor_price < ceiling_price",
            name="floor_below_ceiling",
        ),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        rt = self.room_type.value if isinstance(self.room_type, RoomType) else self.room_type
        return f"<Room {self.hotel_id}/{rt} n={self.room_count} base={self.base_price}>"


# --------------------------------------------------------------------------- #
# Fact tables
# --------------------------------------------------------------------------- #


class Booking(Base):
    """Daily pickup: reservations taken on ``booking_date`` for ``check_in_date``.

    The grain is deliberately *not* "one row per reservation". Storing the
    (booking_date × stay_date) pickup grid preserves the booking curve -- how
    demand for a given night accumulates as that night approaches -- which is the
    single most important structure in hotel revenue management. Phase 4 rebuilds
    on-the-books occupancy at any lead time from exactly these rows.

    ``cancellation_count`` is attributed to the row the booking came from: "of
    the N rooms sold on this day for that night, M were later cancelled".
    """

    __tablename__ = "bookings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    #: Kafka event id, unique. NULL for rows loaded by the batch seeder. Same
    #: role as on ``competitor_prices``: at-least-once delivery means a
    #: redelivered booking must become a rejected insert, not a second booking.
    event_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, unique=True)

    hotel_id: Mapped[str] = mapped_column(String(16), nullable=False)
    room_type: Mapped[RoomType] = mapped_column(_enum(RoomType, "room_type"), nullable=False)

    booking_date: Mapped[date] = mapped_column(Date, nullable=False)
    check_in_date: Mapped[date] = mapped_column(Date, nullable=False)
    check_out_date: Mapped[date] = mapped_column(Date, nullable=False)

    booking_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cancellation_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: Nights sold × achieved rate, net of cancellations.
    revenue: Mapped[float] = mapped_column(MONEY, nullable=False, default=0.0)
    #: Average achieved rate for this pickup row. Denormalised on purpose: it is
    #: read constantly by the ADR charts and recomputing it is a division by a
    #: quantity that can be zero.
    adr: Mapped[float] = mapped_column(MONEY, nullable=False, default=0.0)

    #: ``check_in_date - booking_date`` in days, stored so the booking-curve
    #: queries do not need a date-arithmetic expression the planner cannot index.
    lead_time_days: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    channel: Mapped[str] = mapped_column(String(24), nullable=False, default="direct")
    created_at: Mapped[datetime] = mapped_column(
        TS, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["hotel_id", "room_type"],
            ["rooms.hotel_id", "rooms.room_type"],
            ondelete="CASCADE",
        ),
        CheckConstraint("booking_count >= 0", name="booking_count_non_negative"),
        CheckConstraint(
            "cancellation_count >= 0 AND cancellation_count <= booking_count",
            name="cancellations_within_bookings",
        ),
        CheckConstraint("revenue >= 0", name="revenue_non_negative"),
        CheckConstraint("adr >= 0", name="adr_non_negative"),
        CheckConstraint("check_out_date > check_in_date", name="stay_is_positive"),
        CheckConstraint("booking_date <= check_in_date", name="booked_before_stay"),
        CheckConstraint("lead_time_days >= 0", name="lead_time_non_negative"),
        Index("ix_bookings_hotel_checkin", "hotel_id", "check_in_date"),
        Index("ix_bookings_hotel_room_checkin", "hotel_id", "room_type", "check_in_date"),
        Index("ix_bookings_booking_date", "booking_date"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<Booking {self.hotel_id} {self.check_in_date} "
            f"booked={self.booking_count} lead={self.lead_time_days}d>"
        )


class CompetitorPrice(Base):
    """One observed competitor rate for one night, at one point in time.

    ``event_id`` carries the Kafka event's uuid and is unique. The streaming
    consumer is at-least-once, so the same event can legitimately arrive twice
    after a rebalance; the unique index turns that into a no-op insert instead of
    a duplicated data point that would drag the competitor average around.
    """

    __tablename__ = "competitor_prices"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    #: Kafka event id. NULL for rows loaded by the batch seeder.
    event_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True, unique=True)

    hotel_id: Mapped[str] = mapped_column(String(16), nullable=False)
    room_type: Mapped[RoomType] = mapped_column(_enum(RoomType, "room_type"), nullable=False)
    competitor: Mapped[Competitor] = mapped_column(
        _enum(Competitor, "competitor"), nullable=False
    )

    check_in_date: Mapped[date] = mapped_column(Date, nullable=False)
    price: Mapped[float] = mapped_column(MONEY, nullable=False)
    currency: Mapped[str] = mapped_column(String(3), nullable=False, default="INR")

    #: Whether the competitor showed availability at that rate. A sold-out
    #: competitor is a *stronger* demand signal than a high price.
    is_available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    #: Where the observation came from: ``synthetic``, ``booking``, ``expedia``.
    source: Mapped[str] = mapped_column(String(16), nullable=False, default="synthetic")

    #: When the rate was observed (the "timestamp" field of the event).
    collected_at: Mapped[datetime] = mapped_column(TS, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(
        TS, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["hotel_id", "room_type"],
            ["rooms.hotel_id", "rooms.room_type"],
            ondelete="CASCADE",
        ),
        CheckConstraint("price > 0", name="price_positive"),
        Index("ix_competitor_prices_hotel_checkin", "hotel_id", "check_in_date"),
        Index(
            "ix_competitor_prices_lookup",
            "hotel_id",
            "room_type",
            "check_in_date",
            "collected_at",
        ),
        Index("ix_competitor_prices_collected_at", "collected_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        comp = self.competitor.value if isinstance(self.competitor, Competitor) else self.competitor
        return f"<CompetitorPrice {self.hotel_id} {comp} {self.check_in_date} {self.price}>"


class DemandFeature(Base):
    """The feature store: one row per (hotel, room type, stay date).

    The table is filled in two passes by two different owners, which is why so
    many columns are nullable:

    1. **Exogenous signals** (``search_demand``, ``weather_score``,
       ``local_event_score``, calendar flags) are facts about the world. They
       arrive from the seeder and, in the running system, from the
       ``hotel.demand_events`` topic.
    2. **Derived features and the target** (occupancy, pickup, competitor
       aggregates, ``target_demand``) are computed by the Phase 4 feature
       pipeline and upserted onto the same row.

    ``feature_version`` records which revision of the pipeline wrote the derived
    half, so a model trained on v1 features is never served v2 features.
    """

    __tablename__ = "demand_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    hotel_id: Mapped[str] = mapped_column(String(16), nullable=False)
    room_type: Mapped[RoomType] = mapped_column(_enum(RoomType, "room_type"), nullable=False)
    stay_date: Mapped[date] = mapped_column(Date, nullable=False)

    # --- calendar (always known, never leaks) -------------------------------
    day_of_week: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    is_weekend: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    season: Mapped[Season] = mapped_column(_enum(Season, "season"), nullable=False)
    holiday_flag: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    holiday_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # --- exogenous signals ---------------------------------------------------
    local_event_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    weather_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    search_demand: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # --- inventory and pickup, as observed at ``days_to_checkin`` ------------
    days_to_checkin: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_rooms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    available_rooms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    occupancy_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    booking_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cancellation_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    lead_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    historical_demand: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # --- rates ---------------------------------------------------------------
    current_room_price: Mapped[Optional[float]] = mapped_column(MONEY, nullable=True)
    competitor_rate: Mapped[Optional[float]] = mapped_column(MONEY, nullable=True)
    competitor_min_rate: Mapped[Optional[float]] = mapped_column(MONEY, nullable=True)
    competitor_max_rate: Mapped[Optional[float]] = mapped_column(MONEY, nullable=True)
    competitor_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # --- supervised target ---------------------------------------------------
    #: Final realised demand for the night, as a fraction of inventory. Known
    #: only after the stay date has passed -- never available at serving time.
    target_demand: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    feature_version: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    computed_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TS, server_default=func.now(), nullable=False
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["hotel_id", "room_type"],
            ["rooms.hotel_id", "rooms.room_type"],
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "hotel_id", "room_type", "stay_date", name="uq_demand_features_grain"
        ),
        CheckConstraint("day_of_week BETWEEN 0 AND 6", name="day_of_week_range"),
        CheckConstraint(
            "occupancy_rate IS NULL OR (occupancy_rate >= 0 AND occupancy_rate <= 1)",
            name="occupancy_rate_fraction",
        ),
        CheckConstraint(
            "target_demand IS NULL OR target_demand >= 0", name="target_non_negative"
        ),
        CheckConstraint(
            "available_rooms IS NULL OR available_rooms >= 0",
            name="available_rooms_non_negative",
        ),
        Index("ix_demand_features_stay_date", "stay_date"),
        Index("ix_demand_features_hotel_stay", "hotel_id", "stay_date"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<DemandFeature {self.hotel_id} {self.stay_date} occ={self.occupancy_rate}>"


# --------------------------------------------------------------------------- #
# Serving artefacts
# --------------------------------------------------------------------------- #


class Prediction(Base):
    """One demand prediction served by the API.

    Stores the exact feature vector used, so a disputed price can be replayed
    against the same inputs months later.
    """

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)

    hotel_id: Mapped[str] = mapped_column(String(16), nullable=False)
    room_type: Mapped[RoomType] = mapped_column(_enum(RoomType, "room_type"), nullable=False)
    check_in_date: Mapped[date] = mapped_column(Date, nullable=False)

    forecasted_demand: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_demand: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    blended_demand: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    #: Human-readable version string, e.g. ``v1.0``. Denormalised from
    #: ``model_versions`` so a prediction stays readable after a model is purged.
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    prophet_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    gbr_version: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    features: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON_DICT, nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    correlation_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        TS, server_default=func.now(), nullable=False
    )

    decision: Mapped[Optional["PricingDecision"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan", uselist=False
    )

    __table_args__ = (
        ForeignKeyConstraint(
            ["hotel_id", "room_type"],
            ["rooms.hotel_id", "rooms.room_type"],
            ondelete="CASCADE",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1", name="confidence_fraction"
        ),
        Index("ix_predictions_hotel_created", "hotel_id", "created_at"),
        Index("ix_predictions_hotel_checkin", "hotel_id", "check_in_date"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<Prediction {self.prediction_id} {self.hotel_id} d={self.blended_demand:.3f}>"


class PricingDecision(Base):
    """The audit trail for one price: every input, every adjustment, every rule.

    Verbose by design. When someone asks "why was this night priced at ₹7,340",
    this row answers it without re-running a model.
    """

    __tablename__ = "pricing_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("predictions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    hotel_id: Mapped[str] = mapped_column(String(16), nullable=False)
    room_type: Mapped[RoomType] = mapped_column(_enum(RoomType, "room_type"), nullable=False)
    check_in_date: Mapped[date] = mapped_column(Date, nullable=False)

    # --- inputs --------------------------------------------------------------
    base_price: Mapped[float] = mapped_column(MONEY, nullable=False)
    current_price: Mapped[Optional[float]] = mapped_column(MONEY, nullable=True)
    occupancy_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    competitor_rate: Mapped[Optional[float]] = mapped_column(MONEY, nullable=True)

    # --- the five adjustment factors, as fractions (0.12 == +12%) ------------
    demand_adjustment: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    occupancy_adjustment: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    competitor_adjustment: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    season_adjustment: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    event_adjustment: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    total_adjustment: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # --- outputs -------------------------------------------------------------
    raw_recommended_price: Mapped[float] = mapped_column(MONEY, nullable=False)
    final_recommended_price: Mapped[float] = mapped_column(MONEY, nullable=False)
    price_change_percent: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    #: List of guardrail identifiers that modified the price, in order.
    guardrails_applied: Mapped[Optional[List[Any]]] = mapped_column(
        JSON_DICT, nullable=True
    )
    #: Full human-readable explanation, one entry per adjustment step.
    breakdown: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON_DICT, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        TS, server_default=func.now(), nullable=False
    )

    prediction: Mapped["Prediction"] = relationship(back_populates="decision")

    __table_args__ = (
        CheckConstraint("base_price > 0", name="base_price_positive"),
        CheckConstraint("raw_recommended_price > 0", name="raw_price_positive"),
        CheckConstraint("final_recommended_price > 0", name="final_price_positive"),
        Index("ix_pricing_decisions_hotel_checkin", "hotel_id", "check_in_date"),
        Index("ix_pricing_decisions_created_at", "created_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<PricingDecision {self.hotel_id} raw={self.raw_recommended_price} "
            f"final={self.final_recommended_price}>"
        )


# --------------------------------------------------------------------------- #
# MLOps
# --------------------------------------------------------------------------- #


class ModelVersion(TimestampMixin, Base):
    """A registered, loadable model artifact.

    One row per trained model that survived evaluation. ``is_active`` marks the
    version the API serves; a partial unique index guarantees at most one active
    version per model type, so "which model produced this price" always has
    exactly one answer.
    """

    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    model_type: Mapped[ModelType] = mapped_column(_enum(ModelType, "model_type"), nullable=False)

    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_sha256: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # --- evaluation ----------------------------------------------------------
    mae: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    rmse: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mape: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    r2: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON_DICT, nullable=True)

    # --- provenance ----------------------------------------------------------
    feature_list: Mapped[Optional[List[Any]]] = mapped_column(JSON_DICT, nullable=True)
    hyperparameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON_DICT, nullable=True
    )
    training_rows: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    #: Hash of the exact training frame. Two runs with the same hash and the same
    #: hyperparameters must produce the same model.
    dataset_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    trained_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    runs: Mapped[List["TrainingRun"]] = relationship(back_populates="model_version")

    __table_args__ = (
        UniqueConstraint("model_type", "version", name="uq_model_versions_type_version"),
        Index("ix_model_versions_trained_at", "trained_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        mt = self.model_type.value if isinstance(self.model_type, ModelType) else self.model_type
        return f"<ModelVersion {mt}:{self.version} active={self.is_active}>"


# A *partial* unique index -- "at most one active version per model type" -- needs
# a WHERE clause referencing a built column, which ``__table_args__`` cannot
# express with a string reference. Declaring it against the table afterwards is
# the supported idiom. PostgreSQL and SQLite both honour partial indexes; on any
# other backend the index is simply omitted and ``models.model_registry``
# enforces the same invariant in code.
Index(
    "ix_model_versions_one_active",
    ModelVersion.__table__.c.model_type,
    unique=True,
    postgresql_where=ModelVersion.__table__.c.is_active.is_(True),
    sqlite_where=ModelVersion.__table__.c.is_active.is_(True),
)


class TrainingRun(Base):
    """One training attempt -- including the ones that failed.

    Failures are recorded, not discarded: "the retrain has been failing for three
    days" is the kind of thing a monitoring page must be able to show.
    """

    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), nullable=False, unique=True)

    model_type: Mapped[ModelType] = mapped_column(_enum(ModelType, "model_type"), nullable=False)
    status: Mapped[RunStatus] = mapped_column(
        _enum(RunStatus, "run_status"), nullable=False, default=RunStatus.RUNNING
    )

    model_version_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True
    )

    started_at: Mapped[datetime] = mapped_column(
        TS, server_default=func.now(), nullable=False
    )
    finished_at: Mapped[Optional[datetime]] = mapped_column(TS, nullable=True)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    rows_trained: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rows_validated: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    dataset_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON_DICT, nullable=True)
    metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON_DICT, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    triggered_by: Mapped[str] = mapped_column(String(32), nullable=False, default="cli")

    model_version: Mapped[Optional["ModelVersion"]] = relationship(back_populates="runs")

    __table_args__ = (
        Index("ix_training_runs_started_at", "started_at"),
        Index("ix_training_runs_type_status", "model_type", "status"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        st = self.status.value if isinstance(self.status, RunStatus) else self.status
        return f"<TrainingRun {self.run_id} {st}>"


#: Creation order. ``Base.metadata.sorted_tables`` already resolves dependencies;
#: this list exists so scripts can iterate tables in a stable, readable order.
ALL_TABLES = [
    Hotel.__table__,
    Room.__table__,
    Booking.__table__,
    CompetitorPrice.__table__,
    DemandFeature.__table__,
    ModelVersion.__table__,
    TrainingRun.__table__,
    Prediction.__table__,
    PricingDecision.__table__,
]

TABLE_NAMES = [t.name for t in ALL_TABLES]


__all__ = [
    "ALL_TABLES",
    "TABLE_NAMES",
    "Base",
    "Booking",
    "Competitor",
    "CompetitorPrice",
    "DemandFeature",
    "Hotel",
    "JSON_DICT",
    "MONEY",
    "MarketSegment",
    "ModelType",
    "ModelVersion",
    "Prediction",
    "PricingDecision",
    "Room",
    "RoomType",
    "RunStatus",
    "Season",
    "TrainingRun",
]
