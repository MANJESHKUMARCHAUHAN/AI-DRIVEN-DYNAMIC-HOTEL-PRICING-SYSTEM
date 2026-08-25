"""Pydantic request and response schemas for the HTTP edge.

These types *are* the API contract. Every field carries a description and an
example because they become the Swagger page, and a Swagger page that says
``occupancy_rate: number`` and nothing else is documentation in name only.

Validation lives here rather than in the route handlers. A handler that has to
check its own inputs is a handler that will eventually forget to, and FastAPI
turns a schema violation into a 422 with the offending field named -- which is
more useful than anything a hand-written check would produce.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from database.models import Competitor, RoomType

#: How far ahead a caller may ask us to price. Beyond this the models are
#: extrapolating well past any useful signal, and saying so is more honest than
#: returning a confident number.
MAX_PRICING_HORIZON_DAYS = 365

#: Bound on ``GET`` list endpoints, so one request cannot pull a whole table.
MAX_PAGE_SIZE = 500


def _utc_now() -> datetime:
    """Timezone-aware UTC now. The only clock this project reads."""
    return datetime.now(tz=timezone.utc)


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


class ServiceStatus(str, Enum):
    """Aggregate health of the service."""

    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"


class DependencyState(str, Enum):
    """Reachability of a single downstream dependency."""

    UP = "up"
    DOWN = "down"
    DISABLED = "disabled"
    UNAVAILABLE = "unavailable"


class DependencyStatus(BaseModel):
    """Result of probing one dependency."""

    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(description="Dependency identifier, e.g. 'postgres'.")
    state: DependencyState
    target: str = Field(description="What was probed.")
    latency_ms: Optional[float] = Field(
        default=None, description="Probe round-trip in milliseconds."
    )
    detail: Optional[str] = Field(
        default=None, description="Human-readable reason when not 'up'."
    )


class HealthResponse(BaseModel):
    """``GET /health`` payload.

    Always returns HTTP 200 while the process is alive. A dependency being down
    is reported in ``dependencies`` and reflected in ``status``, but does not
    make the endpoint fail -- otherwise the container healthcheck would kill an
    API that is working correctly and merely waiting on Postgres.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: ServiceStatus
    app: str
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=_utc_now)
    dependencies: List[DependencyStatus] = Field(default_factory=list)
    models: Dict[str, Any] = Field(
        default_factory=dict, description="Which model artifacts are serving."
    )


class ErrorResponse(BaseModel):
    """Uniform error envelope for every non-2xx response."""

    model_config = ConfigDict(protected_namespaces=())

    error: str = Field(description="Short machine-readable error code.")
    detail: str = Field(description="Human-readable explanation.")
    correlation_id: Optional[str] = Field(
        default=None, description="Ties this error to the server logs."
    )
    timestamp: datetime = Field(default_factory=_utc_now)
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Extra, non-sensitive diagnostic fields."
    )


# --------------------------------------------------------------------------- #
# Hotels and rooms
# --------------------------------------------------------------------------- #


class RoomSummary(BaseModel):
    """One sellable room category."""

    model_config = ConfigDict(protected_namespaces=())

    room_id: str = Field(examples=["H001-DEL"])
    room_type: RoomType
    capacity: int = Field(examples=[2])
    room_count: int = Field(description="Inventory of this category.", examples=[72])
    base_price: float = Field(description="Rack rate.", examples=[7936.0])
    floor_price: Optional[float] = None
    ceiling_price: Optional[float] = None


class HotelSummary(BaseModel):
    """A property, without its rooms. Returned by the list endpoint."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    hotel_id: str = Field(examples=["H001"])
    hotel_name: str = Field(examples=["Sanchay Grand Mumbai"])
    city: str = Field(examples=["Mumbai"])
    country: str = Field(examples=["India"])
    star_rating: int = Field(ge=1, le=5, examples=[5])
    total_rooms: int = Field(examples=[240])
    segment: str = Field(examples=["business"])
    currency: str = Field(examples=["INR"])
    is_active: bool = True

    # Always present on HotelDetail; on HotelSummary only when the caller asks
    # for include_performance=true. Null otherwise rather than defaulted to
    # zero: "we did not compute this" and "this hotel sold nothing" are
    # different facts, and a dashboard that cannot tell them apart draws the
    # second one.
    occupancy_last_30_days: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Mean occupancy over the last 30 completed nights.",
    )
    adr_last_30_days: Optional[float] = Field(
        default=None, ge=0, description="Average daily rate over the same window."
    )
    revpar_last_30_days: Optional[float] = Field(
        default=None,
        ge=0,
        description="Revenue per available room -- occupancy x ADR, the number "
        "hotels actually manage against.",
    )


class HotelDetail(HotelSummary):
    """A property with its room inventory and recent trading.

    The three trading fields are inherited rather than redeclared. Redeclaring
    them shadowed the base definitions, so ``HotelDetail(**summary, occupancy=…)``
    raised "multiple values for keyword argument" the moment the base class
    gained them.
    """

    rooms: List[RoomSummary] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #


class PricingRequestSchema(BaseModel):
    """``POST /api/v1/pricing/predict`` body.

    Only the first three fields are required. Everything else refines the
    answer, and anything omitted is looked up from the database or contributes
    nothing -- so a caller who knows only "hotel, room, date" still gets a
    usable price rather than a validation error.
    """

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "hotel_id": "H001",
                "room_type": "deluxe",
                "check_in_date": "2026-09-15",
                "current_price": 6000,
                "occupancy_rate": 0.72,
                "competitor_rate": 6500,
                "available_rooms": 28,
            }
        },
    )

    hotel_id: str = Field(min_length=1, max_length=16, examples=["H001"])
    room_type: RoomType = Field(examples=["deluxe"])
    check_in_date: date = Field(examples=["2026-09-15"])

    current_price: Optional[float] = Field(
        default=None, gt=0, le=1_000_000,
        description="Today's rate for this night. Enables the daily-change cap.",
    )
    occupancy_rate: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Rooms already sold, as a fraction of inventory.",
    )
    available_rooms: Optional[int] = Field(
        default=None, ge=0,
        description="Unsold rooms. Used to derive occupancy when it is absent.",
    )
    competitor_rate: Optional[float] = Field(
        default=None, gt=0, le=1_000_000,
        description="Competitive-set average. Omit and the market is looked up.",
    )
    competitor_min_rate: Optional[float] = Field(default=None, gt=0, le=1_000_000)
    competitor_max_rate: Optional[float] = Field(default=None, gt=0, le=1_000_000)
    base_price: Optional[float] = Field(
        default=None, gt=0, le=1_000_000,
        description="Overrides the room's configured rack rate.",
    )
    as_of: Optional[date] = Field(
        default=None,
        description="Pricing date, for the lead-time features. Defaults to today.",
    )
    persist: bool = Field(
        default=True,
        description="Write the decision to the audit trail. Set false for what-if "
        "queries that should not appear in the history.",
    )

    @field_validator("check_in_date")
    @classmethod
    def _within_horizon(cls, value: date) -> date:
        horizon = (value - datetime.now(timezone.utc).date()).days
        if horizon > MAX_PRICING_HORIZON_DAYS:
            raise ValueError(
                f"check_in_date is {horizon} days ahead; the models cannot say "
                f"anything useful beyond {MAX_PRICING_HORIZON_DAYS} days"
            )
        if horizon < -MAX_PRICING_HORIZON_DAYS:
            raise ValueError(f"check_in_date is {abs(horizon)} days in the past")
        return value

    @model_validator(mode="after")
    def _competitor_band_is_ordered(self) -> "PricingRequestSchema":
        low, high = self.competitor_min_rate, self.competitor_max_rate
        if low is not None and high is not None and low > high:
            raise ValueError(
                f"competitor_min_rate ({low}) is above competitor_max_rate ({high})"
            )
        return self


class AdjustmentSchema(BaseModel):
    """One priced signal, with the reasoning that produced it."""

    name: str = Field(examples=["demand"])
    value: float = Field(description="Fraction applied, e.g. 0.12 for +12%.")
    percent: float = Field(examples=[12.0])
    clamped: bool = Field(description="True if the raw value hit its limit.")
    reason: str = Field(
        examples=["forecast demand 82% is above the 65% baseline, so there is room "
                  "to charge more"]
    )
    inputs: Dict[str, Any] = Field(default_factory=dict)


class GuardrailSchema(BaseModel):
    """A guardrail that changed the price."""

    rule: str = Field(examples=["MAX_DAILY_RISE"])
    before: float
    after: float
    delta: float
    reason: str


class DemandSchema(BaseModel):
    """The blended demand estimate behind a price."""

    model_config = ConfigDict(protected_namespaces=())

    blended_demand: float = Field(examples=[0.805])
    forecasted_demand: Optional[float] = Field(
        default=None, description="Prophet's view.", examples=[0.82]
    )
    predicted_demand: Optional[float] = Field(
        default=None, description="Gradient Boosting's view.", examples=[0.79]
    )
    prophet_weight: float
    confidence: float = Field(ge=0.0, le=1.0, examples=[0.87])
    lower: float
    upper: float
    disagreement: Optional[float] = None
    sources: List[str] = Field(default_factory=list)
    degraded: bool = Field(
        default=False, description="True when a model was unavailable."
    )
    notes: List[str] = Field(default_factory=list)


class PricingResponse(BaseModel):
    """``POST /api/v1/pricing/predict`` payload.

    Carries the full calculation, not just the number. A price a revenue manager
    cannot interrogate is a price they will override.
    """

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "hotel_id": "H001",
                "room_type": "deluxe",
                "check_in_date": "2026-09-15",
                "forecasted_demand": 0.82,
                "predicted_demand": 0.79,
                "base_price": 5000,
                "raw_recommended_price": 6900,
                "final_recommended_price": 6750,
                "price_change_percent": 12.5,
                "competitor_rate": 6500,
                "confidence": 0.87,
                "guardrails_applied": [],
                "model_version": "v1",
            }
        },
    )

    hotel_id: str
    room_type: RoomType
    check_in_date: date
    currency: str = "INR"

    forecasted_demand: Optional[float] = Field(description="Prophet's view.")
    predicted_demand: Optional[float] = Field(description="Gradient Boosting's view.")
    blended_demand: float

    base_price: float
    current_price: Optional[float] = None
    raw_recommended_price: float = Field(description="Before guardrails.")
    final_recommended_price: float = Field(description="After guardrails. Serve this.")
    price_change_percent: float

    competitor_rate: Optional[float] = None
    confidence: float = Field(ge=0.0, le=1.0)

    adjustments: List[AdjustmentSchema] = Field(default_factory=list)
    total_adjustment: float
    guardrails_applied: List[str] = Field(
        default_factory=list, description="Identifiers of the rules that fired."
    )
    guardrail_detail: List[GuardrailSchema] = Field(default_factory=list)
    demand: DemandSchema

    model_version: str = Field(examples=["v1"])
    feature_version: str = Field(examples=["v1"])
    prediction_id: str
    explanation: str = Field(
        description="The whole calculation as readable text, for tickets and audits."
    )
    latency_ms: float
    timestamp: datetime = Field(default_factory=_utc_now)


class PricingHistoryItem(BaseModel):
    """One past pricing decision."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    prediction_id: str
    hotel_id: str
    room_type: RoomType
    check_in_date: date
    base_price: float
    raw_recommended_price: float
    final_recommended_price: float
    price_change_percent: float
    guardrails_applied: List[str] = Field(default_factory=list)
    blended_demand: Optional[float] = None
    confidence: Optional[float] = None
    model_version: Optional[str] = None
    created_at: datetime


class PricingHistoryResponse(BaseModel):
    """``GET /api/v1/pricing/{hotel_id}`` payload."""

    hotel_id: str
    count: int
    items: List[PricingHistoryItem] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Forecasts
# --------------------------------------------------------------------------- #


class ForecastPoint(BaseModel):
    """One forecast day."""

    date: date
    forecast: float = Field(description="Demand as a fraction of inventory.")
    lower: float
    upper: float
    trend: float


class ForecastResponse(BaseModel):
    """``GET /api/v1/forecast/{hotel_id}`` payload."""

    model_config = ConfigDict(protected_namespaces=())

    hotel_id: str
    room_type: RoomType
    horizon_days: int
    model_version: Optional[str] = None
    generated_at: datetime = Field(default_factory=_utc_now)
    points: List[ForecastPoint] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Competitors
# --------------------------------------------------------------------------- #


class CompetitorPriceItem(BaseModel):
    """One observed competitor rate."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    competitor: Competitor
    room_type: RoomType
    check_in_date: date
    price: float
    currency: str
    is_available: bool
    source: str
    collected_at: datetime


class CompetitorSummary(BaseModel):
    """The competitive set for one night, aggregated."""

    check_in_date: date
    room_type: RoomType
    competitor_rate: float = Field(description="Mean across sources.")
    competitor_min_rate: float
    competitor_max_rate: float
    competitor_count: int
    spread_percent: float = Field(
        description="(max - min) / mean. A wide spread means weak price discipline."
    )


class CompetitorResponse(BaseModel):
    """``GET /api/v1/competitors/{hotel_id}`` payload."""

    hotel_id: str
    count: int
    summaries: List[CompetitorSummary] = Field(default_factory=list)
    observations: List[CompetitorPriceItem] = Field(default_factory=list)


class CompetitorEventRequest(BaseModel):
    """``POST /api/v1/competitors/events`` body: one observed rate."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "hotel_id": "H001",
                "competitor": "booking",
                "room_type": "deluxe",
                "check_in_date": "2026-09-15",
                "price": 6200,
                "currency": "INR",
            }
        },
    )

    hotel_id: str = Field(min_length=1, max_length=16)
    competitor: Competitor
    room_type: RoomType
    check_in_date: date
    price: float = Field(gt=0, le=1_000_000)
    currency: str = Field(default="INR", min_length=3, max_length=3)
    is_available: bool = True
    source: str = Field(default="api", max_length=16)


class CompetitorEventResponse(BaseModel):
    """Acknowledgement for an accepted competitor event."""

    accepted: bool
    event_id: str
    published_to_kafka: bool = Field(
        description="False when Kafka is disabled or unreachable; the event is "
        "still persisted."
    )
    persisted: bool
    detail: str


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class ModelVersionSchema(BaseModel):
    """One trained version on disk."""

    model_config = ConfigDict(protected_namespaces=())

    version: str
    is_active: bool
    gradient_boosting: Optional[str] = None
    prophet: Optional[str] = None
    trained_at: Optional[str] = None
    feature_version: Optional[str] = None
    dataset_hash: Optional[str] = None
    n_train: Optional[int] = None
    n_test: Optional[int] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ModelsResponse(BaseModel):
    """``GET /api/v1/models`` payload."""

    model_config = ConfigDict(protected_namespaces=())

    active_version: Optional[str] = None
    available: List[str] = Field(default_factory=list)
    loaded_at: Optional[str] = None
    errors: Dict[str, str] = Field(default_factory=dict)
    feature_version: str
    versions: List[ModelVersionSchema] = Field(default_factory=list)


class TrainRequest(BaseModel):
    """``POST /api/v1/models/train`` body."""

    model_config = ConfigDict(protected_namespaces=())

    test_days: int = Field(default=60, ge=7, le=365)
    train_prophet: bool = True
    train_gradient_boosting: bool = True
    backtest_folds: int = Field(default=0, ge=0, le=5)
    reload_after: bool = Field(
        default=True, description="Serve the new models as soon as they are saved."
    )


class TrainResponse(BaseModel):
    """``POST /api/v1/models/train`` payload."""

    model_config = ConfigDict(protected_namespaces=())

    version: str
    succeeded: bool
    duration_seconds: float
    n_train: int
    n_test: int
    steps: List[Dict[str, Any]] = Field(default_factory=list)
    artifacts: Dict[str, str] = Field(default_factory=dict)
    reloaded: bool
    summary: str


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


class IngestionSourceInfo(BaseModel):
    """What the competitor feed is currently configured to do."""

    source: str = Field(description="synthetic, demo_ota, booking or expedia.")
    is_scraper: bool = Field(description="Whether this source fetches over the network.")
    is_third_party: bool = Field(
        description="Whether it sends traffic to somebody else's site. Third-party "
        "sources additionally require INGESTION_ENABLE_REAL_SCRAPERS=true."
    )
    real_scrapers_enabled: bool
    base_url: Optional[str] = Field(
        default=None, description="Host being scraped. Null for the synthetic source."
    )
    layout: Optional[str] = Field(
        default=None,
        description="Demo OTA markup version. 'v2' simulates a redesign that "
        "breaks the scraper's selectors.",
    )
    rate_limit_seconds: float
    user_agent: str


class RobotsStatus(BaseModel):
    """Result of reading the target's ``robots.txt``.

    Surfaced because a scraper that quietly stopped honouring robots is a failure
    nobody notices. ``allowed=false`` is a correct, healthy outcome when the site
    says no -- it is not an error state to be cleared.
    """

    checked: bool
    allowed: Optional[bool] = None
    url: Optional[str] = None
    detail: str


class IngestionStatusResponse(BaseModel):
    """Configuration, crawl permission, and what has actually landed."""

    source: IngestionSourceInfo
    robots: RobotsStatus
    observations_total: int = Field(description="Rows in competitor_prices.")
    observations_last_hour: int
    latest_observed_at: Optional[datetime] = None
    by_source: Dict[str, int] = Field(
        default_factory=dict,
        description="Row counts keyed by the source that produced them.",
    )
    checked_at: datetime = Field(default_factory=_utc_now)


class IngestionRunRequest(BaseModel):
    """Ask the API to run one collection pass now."""

    hotel_ids: Optional[List[str]] = Field(
        default=None,
        description="Hotels to collect for. Defaults to every hotel in the catalogue.",
    )
    room_types: Optional[List[RoomType]] = Field(
        default=None, description="Categories to collect. Defaults to all four."
    )
    horizons: Optional[List[int]] = Field(
        default=None,
        description="Days ahead to collect for. Defaults to INGESTION_SCRAPE_HORIZONS.",
    )
    publish: bool = Field(
        default=True,
        description="Publish each rate to Kafka and persist it. Set false to "
        "preview what the scraper returns without changing any data.",
    )

    @field_validator("horizons")
    @classmethod
    def _bounded_horizons(cls, value: Optional[List[int]]) -> Optional[List[int]]:
        if value is None:
            return value
        if not value:
            raise ValueError("horizons must not be empty")
        # Every horizon multiplies the request count by hotels x room types, and
        # the rate limiter turns that into wall-clock time. A caller asking for
        # fifty horizons wants a background job, not an HTTP request.
        if len(value) > 12:
            raise ValueError("at most 12 horizons per run")
        for horizon in value:
            if not 0 <= horizon <= MAX_PRICING_HORIZON_DAYS:
                raise ValueError(f"horizon {horizon} is outside 0..{MAX_PRICING_HORIZON_DAYS}")
        return value


class ScrapedRate(BaseModel):
    """One rate as the scraper returned it, before anything downstream."""

    hotel_id: str
    competitor: Competitor
    room_type: RoomType
    check_in_date: date
    price: float
    currency: str
    is_available: bool
    source: str


class IngestionRunResponse(BaseModel):
    """Outcome of one collection pass.

    A run can be ``succeeded=true`` with a non-zero ``failed``: missing competitor
    data for some nights is normal, and the feature pipeline already handles gaps.
    What is not normal is every request failing, which ``blocked`` together with a
    zero ``rates_collected`` is what reports.
    """

    succeeded: bool
    source: str
    requests_made: int
    rates_collected: int
    failed: int
    published: int
    persisted: int
    duplicates: int
    blocked: bool = Field(
        description="True when the site refused us and collection stopped early."
    )
    duration_seconds: float
    rates: List[ScrapedRate] = Field(
        default_factory=list, description="The rates collected, capped for transport."
    )
    errors: List[str] = Field(default_factory=list)
    detail: str


__all__ = [
    "MAX_PAGE_SIZE",
    "MAX_PRICING_HORIZON_DAYS",
    "AdjustmentSchema",
    "CompetitorEventRequest",
    "CompetitorEventResponse",
    "CompetitorPriceItem",
    "CompetitorResponse",
    "CompetitorSummary",
    "DemandSchema",
    "DependencyState",
    "DependencyStatus",
    "ErrorResponse",
    "ForecastPoint",
    "ForecastResponse",
    "GuardrailSchema",
    "HealthResponse",
    "HotelDetail",
    "HotelSummary",
    "IngestionRunRequest",
    "IngestionRunResponse",
    "IngestionSourceInfo",
    "IngestionStatusResponse",
    "ModelVersionSchema",
    "ModelsResponse",
    "PricingHistoryItem",
    "PricingHistoryResponse",
    "PricingRequestSchema",
    "PricingResponse",
    "RobotsStatus",
    "RoomSummary",
    "ScrapedRate",
    "ServiceStatus",
    "TrainRequest",
    "TrainResponse",
]
