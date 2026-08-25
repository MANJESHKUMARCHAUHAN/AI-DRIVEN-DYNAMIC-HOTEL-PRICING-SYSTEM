"""Pricing endpoints: the reason this service exists.

``POST /api/v1/pricing/predict`` is the whole system in one request. It:

1. resolves the hotel, room and base rate from the database;
2. fills in whatever the caller did not send from the feature store;
3. builds the serving feature row through the *same* code the training matrix
   uses, which is what keeps train/serve parity real;
4. blends Prophet and the Gradient Boosting model into one demand estimate;
5. runs the pricing rules and the guardrails;
6. persists the prediction and the decision as an audit trail;
7. publishes the result to Kafka, best-effort.

Steps 6 and 7 are deliberately non-fatal. A price that was computed correctly
should be returned even if the audit write or the event publish fails -- those
are recorded as warnings, not turned into a 500 for the caller who just wanted
a rate.
"""

from __future__ import annotations

import time
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.dependencies import (
    demand_engine_dependency,
    get_registry_dependency,
    pricing_engine_dependency,
    require_hotel,
    require_room,
    session_dependency,
    settings_dependency,
)
from api.security import require_read
from api.schemas import (
    MAX_PAGE_SIZE,
    AdjustmentSchema,
    DemandSchema,
    GuardrailSchema,
    PricingHistoryItem,
    PricingHistoryResponse,
    PricingRequestSchema,
    PricingResponse,
)
from config import Settings
from database.models import (
    CompetitorPrice,
    DemandFeature,
    Hotel,
    Prediction,
    PricingDecision,
    Room,
    RoomType,
)
from features.calendars import event_score, is_holiday, is_weekend, season_of
from features.feature_engineering import FEATURE_VERSION, build_serving_row
from models.model_registry import ModelRegistry
from monitoring.logging_config import get_correlation_id, get_logger
from monitoring.metrics import observe_pricing_decision
from pricing.demand_engine import DemandEngine
from pricing.pricing_engine import PriceDecision, PricingEngine, PricingRequest
from streaming.events import PricePredictionPayload
from streaming.producer import get_producer
from streaming.topics import TopicName

logger = get_logger(__name__)

router = APIRouter(tags=["pricing"])

#: How far back to look for the most recent competitor observation when the
#: caller did not supply one. Older than this and the market has moved.
COMPETITOR_LOOKBACK_DAYS = 30


# --------------------------------------------------------------------------- #
# Context assembly
# --------------------------------------------------------------------------- #


def _competitor_context(
    session: Session, hotel_id: str, room_type: RoomType, check_in_date: date
) -> Dict[str, Any]:
    """Look up the competitive set for one night.

    Uses the freshest observation per competitor rather than an average over
    time: a rate published three weeks ago is not the market today.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=COMPETITOR_LOOKBACK_DAYS)

    freshest = (
        select(
            CompetitorPrice.competitor,
            func.max(CompetitorPrice.collected_at).label("latest"),
        )
        .where(
            CompetitorPrice.hotel_id == hotel_id,
            CompetitorPrice.room_type == room_type,
            CompetitorPrice.check_in_date == check_in_date,
            CompetitorPrice.collected_at >= cutoff,
        )
        .group_by(CompetitorPrice.competitor)
        .subquery()
    )

    rows = session.execute(
        select(CompetitorPrice.price).join(
            freshest,
            (CompetitorPrice.competitor == freshest.c.competitor)
            & (CompetitorPrice.collected_at == freshest.c.latest),
        ).where(
            CompetitorPrice.hotel_id == hotel_id,
            CompetitorPrice.room_type == room_type,
            CompetitorPrice.check_in_date == check_in_date,
        )
    ).scalars().all()

    prices = [float(p) for p in rows if p]
    if not prices:
        return {"competitor_rate": None, "competitor_min_rate": None,
                "competitor_max_rate": None, "competitor_count": 0,
                "competitor_missing": True}

    return {
        "competitor_rate": sum(prices) / len(prices),
        "competitor_min_rate": min(prices),
        "competitor_max_rate": max(prices),
        "competitor_count": len(prices),
        "competitor_missing": False,
    }


def _stored_features(
    session: Session, hotel_id: str, room_type: RoomType, check_in_date: date
) -> Dict[str, Any]:
    """Whatever the feature store already knows about this night."""
    row = session.execute(
        select(DemandFeature).where(
            DemandFeature.hotel_id == hotel_id,
            DemandFeature.room_type == room_type,
            DemandFeature.stay_date == check_in_date,
        )
    ).scalar_one_or_none()
    return row.to_dict() if row is not None else {}


def _historical_demand(
    session: Session, hotel_id: str, room_type: RoomType, check_in_date: date
) -> Optional[float]:
    """Mean realised demand for this hotel, room and weekday over the last year.

    The fallback when no model can answer. Matching on weekday matters: the
    average of a business hotel's Wednesdays and Sundays describes neither.
    """
    value = session.execute(
        select(func.avg(DemandFeature.target_demand)).where(
            DemandFeature.hotel_id == hotel_id,
            DemandFeature.room_type == room_type,
            DemandFeature.target_demand.is_not(None),
            DemandFeature.day_of_week == check_in_date.weekday(),
        )
    ).scalar_one_or_none()
    return float(value) if value is not None else None


def _build_request(
    body: PricingRequestSchema,
    hotel: Hotel,
    room: Room,
    session: Session,
) -> tuple[PricingRequest, Dict[str, Any]]:
    """Merge what the caller sent with what the database knows.

    Caller-supplied values always win: they are describing the situation as they
    see it right now, which is fresher than anything stored.
    """
    stored = _stored_features(session, hotel.hotel_id, body.room_type, body.check_in_date)
    market = _competitor_context(
        session, hotel.hotel_id, body.room_type, body.check_in_date
    )

    as_of = body.as_of or datetime.now(timezone.utc).date()
    days_to_checkin = max((body.check_in_date - as_of).days, 0)
    base_price = float(body.base_price or room.base_price)

    occupancy = body.occupancy_rate
    if occupancy is None and body.available_rooms is not None and room.room_count:
        occupancy = 1.0 - (body.available_rooms / room.room_count)
    if occupancy is None:
        occupancy = stored.get("occupancy_rate")

    competitor_rate = body.competitor_rate if body.competitor_rate is not None else market["competitor_rate"]
    competitor_min = body.competitor_min_rate or market["competitor_min_rate"]
    competitor_max = body.competitor_max_rate or market["competitor_max_rate"]
    competitor_missing = competitor_rate is None

    # One competitor rate with no band is still a band of one.
    if competitor_rate is not None:
        competitor_min = competitor_min or competitor_rate
        competitor_max = competitor_max or competitor_rate

    request = PricingRequest(
        hotel_id=hotel.hotel_id,
        room_type=body.room_type,
        check_in_date=body.check_in_date,
        base_price=base_price,
        current_price=body.current_price,
        occupancy_rate=occupancy,
        available_rooms=body.available_rooms,
        total_rooms=room.room_count,
        days_to_checkin=days_to_checkin,
        competitor_rate=competitor_rate,
        competitor_min_rate=competitor_min,
        competitor_max_rate=competitor_max,
        competitor_missing=competitor_missing,
        season=season_of(body.check_in_date),
        is_weekend=is_weekend(body.check_in_date),
        is_holiday=is_holiday(body.check_in_date),
        event_score=event_score(hotel.city, body.check_in_date),
        room_floor_price=float(room.floor_price) if room.floor_price else None,
        room_ceiling_price=float(room.ceiling_price) if room.ceiling_price else None,
    )

    context = {
        "stored": stored,
        "market": market,
        "as_of": as_of,
        "city": hotel.city,
    }
    return request, context


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def _persist(
    session: Session,
    decision: PriceDecision,
    prediction_id: str,
    model_version: str,
    features: Dict[str, Any],
    latency_ms: float,
) -> bool:
    """Write the prediction and its decision to the audit trail.

    Returns ``False`` on failure rather than raising: a correctly computed price
    should still reach the caller if the audit write fails.
    """
    try:
        prediction = Prediction(
            prediction_id=prediction_id,
            hotel_id=decision.hotel_id,
            room_type=decision.room_type,
            check_in_date=decision.check_in_date,
            forecasted_demand=decision.demand.prophet,
            predicted_demand=decision.demand.gbr,
            blended_demand=decision.demand.blended,
            confidence=decision.demand.confidence,
            model_version=model_version,
            features=features,
            latency_ms=latency_ms,
            correlation_id=get_correlation_id(),
        )
        session.add(prediction)
        session.flush()

        adjustments = decision.adjustment_map()
        session.add(
            PricingDecision(
                prediction_id=prediction.id,
                hotel_id=decision.hotel_id,
                room_type=decision.room_type,
                check_in_date=decision.check_in_date,
                base_price=decision.base_price,
                current_price=decision.current_price,
                occupancy_rate=None,
                competitor_rate=None,
                demand_adjustment=adjustments.get("demand", 0.0),
                occupancy_adjustment=adjustments.get("occupancy", 0.0),
                competitor_adjustment=adjustments.get("competitor", 0.0),
                season_adjustment=adjustments.get("season", 0.0),
                event_adjustment=adjustments.get("event", 0.0),
                total_adjustment=decision.total_adjustment,
                raw_recommended_price=decision.raw_price,
                final_recommended_price=decision.final_price,
                price_change_percent=decision.price_change_percent,
                guardrails_applied=decision.guardrails_applied,
                breakdown=decision.as_dict(),
            )
        )
        session.commit()
        return True
    except SQLAlchemyError as exc:
        session.rollback()
        logger.warning(
            "Could not persist prediction %s: %s: %s",
            prediction_id,
            type(exc).__name__,
            exc,
        )
        return False


def _publish(decision: PriceDecision, prediction_id: str, model_version: str) -> bool:
    """Publish the price to Kafka, best-effort."""
    payload = PricePredictionPayload(
        hotel_id=decision.hotel_id,
        room_type=decision.room_type,
        check_in_date=decision.check_in_date,
        prediction_id=prediction_id,
        base_price=decision.base_price,
        raw_recommended_price=max(decision.raw_price, 0.01),
        final_recommended_price=decision.final_price,
        price_change_percent=decision.price_change_percent,
        blended_demand=decision.demand.blended,
        confidence=decision.demand.confidence,
        model_version=model_version,
        guardrails_applied=decision.guardrails_applied,
    )
    return get_producer().send(payload, TopicName.PRICE_PREDICTIONS, source="api")


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.post(
    "/pricing/predict",
    response_model=PricingResponse,
    summary="Recommend a price for one room-night",
    description=(
        "Blends the Prophet forecast and the Gradient Boosting prediction, "
        "applies the five pricing adjustments, runs every business guardrail, "
        "and returns the recommended rate together with the full reasoning.\n\n"
        "Only `hotel_id`, `room_type` and `check_in_date` are required; anything "
        "else is looked up from the feature store when omitted."
    ),
    responses={404: {"description": "Unknown hotel or room type"}},
)
def predict_price(
    body: PricingRequestSchema,
    session: Session = Depends(session_dependency),
    demand_engine: DemandEngine = Depends(demand_engine_dependency),
    pricing_engine: PricingEngine = Depends(pricing_engine_dependency),
    registry: ModelRegistry = Depends(get_registry_dependency),
    settings: Settings = Depends(settings_dependency),
    scope: str = Depends(require_read),
) -> PricingResponse:
    """Price one room-night and record the decision.

    Read-scoped by default, because a simulation changes nothing. Persisting
    writes to the audit trail, so that path needs the write scope -- checked
    here rather than by the dependency, since which one applies depends on a
    field in the body.
    """
    if body.persist and scope != "write":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="persisting a pricing decision requires a write-scoped key; "
            "send persist=false to simulate",
        )

    started = time.perf_counter()

    hotel = require_hotel(session, body.hotel_id)
    room = require_room(session, body.hotel_id, body.room_type)

    request, context = _build_request(body, hotel, room, session)
    stored = context["stored"]

    # The train/serve parity seam: the same function that builds a training row
    # builds this one.
    features = build_serving_row(
        hotel_id=hotel.hotel_id,
        room_type=body.room_type,
        check_in_date=body.check_in_date,
        base_price=request.base_price,
        total_rooms=room.room_count,
        city=hotel.city,
        as_of=context["as_of"],
        occupancy_rate=request.occupancy_rate,
        available_rooms=body.available_rooms,
        current_price=body.current_price,
        competitor_rate=request.competitor_rate,
        competitor_min_rate=request.competitor_min_rate,
        competitor_max_rate=request.competitor_max_rate,
        competitor_count=context["market"]["competitor_count"],
        booking_count=stored.get("booking_count"),
        cancellation_count=stored.get("cancellation_count"),
        lead_time=stored.get("lead_time"),
        search_demand=stored.get("search_demand"),
        weather_score=stored.get("weather_score"),
        local_event_score=stored.get("local_event_score"),
        historical_demand=stored.get("historical_demand"),
    )

    demand = demand_engine.estimate(
        hotel.hotel_id,
        body.room_type,
        body.check_in_date,
        features,
        fallback_demand=_historical_demand(
            session, hotel.hotel_id, body.room_type, body.check_in_date
        ),
    )

    decision = pricing_engine.price(request, demand)

    prediction_id = str(uuid.uuid4())
    model_version = registry.loaded.version or "unversioned"
    latency_ms = (time.perf_counter() - started) * 1000.0

    if body.persist:
        _persist(
            session,
            decision,
            prediction_id,
            model_version,
            features.iloc[0].to_dict(),
            latency_ms,
        )
        _publish(decision, prediction_id, model_version)

    # Guardrail pressure is the metric worth alerting on. One firing now and then
    # is the system working; one firing on most decisions means the model wants
    # prices the business will not allow -- a retuning signal that is invisible
    # unless somebody counts.
    observe_pricing_decision(
        persisted=body.persist,
        guardrails=decision.guardrails_applied or (),
    )

    return PricingResponse(
        hotel_id=decision.hotel_id,
        room_type=decision.room_type,
        check_in_date=decision.check_in_date,
        currency=settings.pricing.currency,
        forecasted_demand=demand.prophet,
        predicted_demand=demand.gbr,
        blended_demand=demand.blended,
        base_price=round(decision.base_price, 2),
        current_price=decision.current_price,
        raw_recommended_price=round(decision.raw_price, 2),
        final_recommended_price=round(decision.final_price, 2),
        price_change_percent=round(decision.price_change_percent, 2),
        competitor_rate=(
            round(request.competitor_rate, 2) if request.competitor_rate else None
        ),
        confidence=demand.confidence,
        adjustments=[AdjustmentSchema(**a.as_dict()) for a in decision.adjustments],
        total_adjustment=round(decision.total_adjustment, 6),
        guardrails_applied=decision.guardrails_applied,
        guardrail_detail=[GuardrailSchema(**hit) for hit in decision.guardrails],
        demand=DemandSchema(**demand.as_dict()),
        model_version=model_version,
        feature_version=FEATURE_VERSION,
        prediction_id=prediction_id,
        explanation=decision.explain(settings.pricing.currency),
        latency_ms=round(latency_ms, 2),
    )


@router.get(
    "/pricing/{hotel_id}",
    response_model=PricingHistoryResponse,
    summary="Recent pricing decisions for a hotel",
    description="The audit trail, newest first. Every price this service has "
    "recommended, with the guardrails that fired.",
    responses={404: {"description": "Unknown hotel"}},
)
def pricing_history(
    hotel_id: str = Path(examples=["H001"]),
    room_type: Optional[RoomType] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=MAX_PAGE_SIZE),
    session: Session = Depends(session_dependency),
) -> PricingHistoryResponse:
    """Return recent pricing decisions for one hotel."""
    require_hotel(session, hotel_id)

    statement = (
        select(PricingDecision, Prediction)
        .join(Prediction, PricingDecision.prediction_id == Prediction.id)
        .where(PricingDecision.hotel_id == hotel_id)
        .order_by(PricingDecision.created_at.desc())
        .limit(limit)
    )
    if room_type is not None:
        statement = statement.where(PricingDecision.room_type == room_type)

    items: List[PricingHistoryItem] = []
    for decision, prediction in session.execute(statement).all():
        items.append(
            PricingHistoryItem(
                prediction_id=prediction.prediction_id,
                hotel_id=decision.hotel_id,
                room_type=decision.room_type,
                check_in_date=decision.check_in_date,
                base_price=decision.base_price,
                raw_recommended_price=decision.raw_recommended_price,
                final_recommended_price=decision.final_recommended_price,
                price_change_percent=decision.price_change_percent,
                guardrails_applied=list(decision.guardrails_applied or []),
                blended_demand=prediction.blended_demand,
                confidence=prediction.confidence,
                model_version=prediction.model_version,
                created_at=decision.created_at,
            )
        )

    return PricingHistoryResponse(hotel_id=hotel_id, count=len(items), items=items)


__all__ = ["router"]
