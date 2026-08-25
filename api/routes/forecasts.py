"""Demand forecast endpoint.

Serves Prophet's view directly, unblended. That is deliberate: the blended
number belongs to a *price* and only makes sense alongside the situation that
produced it, whereas a forecast is a statement about the calendar that stands on
its own. A dashboard plotting "expected demand for the next 30 nights" wants the
time series, not thirty pricing decisions.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from api.dependencies import (
    get_registry_dependency,
    require_room,
    session_dependency,
)
from api.schemas import ForecastPoint, ForecastResponse
from database.models import RoomType
from models.model_registry import ModelRegistry
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["forecasts"])

#: The horizons the specification names, plus the bounds a caller may ask for.
MIN_HORIZON_DAYS = 1
MAX_HORIZON_DAYS = 90


@router.get(
    "/forecast/{hotel_id}",
    response_model=ForecastResponse,
    summary="Demand forecast for a hotel and room type",
    description=(
        "Prophet's demand forecast, with an 80% uncertainty band and the "
        "underlying trend. Demand is expressed as a fraction of inventory, so "
        "0.82 means 82% of rooms are expected to sell.\n\n"
        "Returns 503 when no forecasting model has been trained yet."
    ),
    responses={
        404: {"description": "Unknown hotel or room type"},
        503: {"description": "No Prophet model is loaded"},
    },
)
def get_forecast(
    hotel_id: str = Path(examples=["H001"]),
    room_type: RoomType = Query(default=RoomType.DELUXE),
    horizon_days: int = Query(
        default=30,
        ge=MIN_HORIZON_DAYS,
        le=MAX_HORIZON_DAYS,
        description="Days ahead. The specification's headline horizons are 7, 14 and 30.",
    ),
    start_date: Optional[date] = Query(
        default=None,
        description="First night to forecast. Defaults to today.",
    ),
    session: Session = Depends(session_dependency),
    registry: ModelRegistry = Depends(get_registry_dependency),
) -> ForecastResponse:
    """Forecast demand for one hotel and room type."""
    require_room(session, hotel_id, room_type)

    bundle = registry.loaded.prophet
    if bundle is None:
        # 503 rather than 500: this is a missing capability, not a fault, and it
        # resolves by training rather than by debugging.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No Prophet model is loaded. Run scripts/train_models.py, or "
                "POST /api/v1/models/train."
            ),
        )

    if not bundle.has(hotel_id, room_type):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No forecast series was fitted for {hotel_id}/{room_type.value}. "
                f"The series may have had too little history at training time."
            ),
        )

    # forecast_range, not forecast: the latter continues from the end of the
    # *training* window, which the pipeline deliberately ends sixty days before
    # today. "The next 7 nights" must mean the next 7 nights.
    start = start_date or datetime.now(timezone.utc).date()
    result = bundle.forecast_range(
        hotel_id, room_type, start=start, horizon_days=horizon_days
    )

    points: List[ForecastPoint] = [
        ForecastPoint(
            date=row.ds.date(),
            forecast=round(float(row.yhat), 4),
            lower=round(float(row.yhat_lower), 4),
            upper=round(float(row.yhat_upper), 4),
            trend=round(float(row.trend), 4),
        )
        for row in result.frame.itertuples()
    ]

    return ForecastResponse(
        hotel_id=hotel_id,
        room_type=room_type,
        horizon_days=horizon_days,
        model_version=registry.loaded.version,
        generated_at=datetime.now(timezone.utc),
        points=points,
    )


__all__ = ["router"]
