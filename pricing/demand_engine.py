"""Two demand models, one number.

Prophet and the Gradient Boosting regressor answer different questions and
neither subsumes the other:

* Prophet knows *"mid-September Tuesdays in Goa trend like this"* -- calendar
  structure, trend, the weekly rhythm. It works for dates nobody has booked yet,
  because it needs only the date.
* The GBR knows *"given 72% on the books, a competitor at 6,500 and 14 days to
  go, demand looks like this"* -- the contextual response to the situation as it
  actually stands.

Prophet cannot see today's competitor rate. The GBR cannot see that next
Thursday is Diwali. Blending them is not hedging, it is combining complementary
signal::

    blended = w x prophet + (1 - w) x gbr

Two operational properties matter as much as the arithmetic:

**Degradation is graceful and visible.** If one model is missing, unfitted for
this series, or throws, the weight collapses onto the other and ``confidence``
drops. If both are gone the engine falls back to the stored historical demand
and says so. A pricing API that returns 500 because a model file is missing has
turned a degraded feature into an outage.

**Confidence is derived, not asserted.** It combines each model's own
uncertainty with how much the two disagree. Two models that agree closely, each
with a tight interval, is the only situation that earns a high number -- and the
pricing engine uses it to decide how far to move the price.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd

from config import Settings, get_settings
from domain.enums import RoomType
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

#: Confidence assigned when neither model could be consulted and the engine is
#: running on stored history alone. Low by design: it is a fallback, not a
#: prediction, and the pricing engine should barely move the price on it.
FALLBACK_CONFIDENCE = 0.25

#: Confidence ceiling when only one model contributed.
SINGLE_MODEL_CONFIDENCE_CAP = 0.70


@dataclass(frozen=True)
class DemandEstimate:
    """The blended demand prediction and everything behind it.

    Attributes:
        blended: The number the pricing engine uses.
        prophet: Prophet's forecast, or ``None`` if unavailable.
        gbr: The Gradient Boosting prediction, or ``None``.
        weight: Weight actually given to Prophet. Collapses to 0 or 1 when only
            one model contributed.
        confidence: 0-1. Combines model uncertainty with model disagreement.
        lower / upper: Uncertainty band around ``blended``.
        sources: Which models actually contributed.
        degraded: True when at least one model was unavailable.
    """

    blended: float
    prophet: Optional[float]
    gbr: Optional[float]
    weight: float
    confidence: float
    lower: float
    upper: float
    sources: List[str] = field(default_factory=list)
    degraded: bool = False
    notes: List[str] = field(default_factory=list)

    @property
    def disagreement(self) -> Optional[float]:
        """Absolute gap between the two models, when both are present."""
        if self.prophet is None or self.gbr is None:
            return None
        return abs(self.prophet - self.gbr)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "blended_demand": round(self.blended, 4),
            "forecasted_demand": round(self.prophet, 4) if self.prophet is not None else None,
            "predicted_demand": round(self.gbr, 4) if self.gbr is not None else None,
            "prophet_weight": round(self.weight, 3),
            "confidence": round(self.confidence, 3),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "disagreement": (
                round(self.disagreement, 4) if self.disagreement is not None else None
            ),
            "sources": list(self.sources),
            "degraded": self.degraded,
            "notes": list(self.notes),
        }


class DemandEngine:
    """Produces a single demand estimate from whichever models are loaded.

    Example::

        engine = DemandEngine(prophet_bundle=bundle, gbr_model=model)
        estimate = engine.estimate("H001", RoomType.DELUXE, check_in, features)
    """

    def __init__(
        self,
        *,
        prophet_bundle: Optional[Any] = None,
        gbr_model: Optional[Any] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.prophet_bundle = prophet_bundle
        self.gbr_model = gbr_model

    # -- availability -------------------------------------------------------- #

    @property
    def has_prophet(self) -> bool:
        return self.prophet_bundle is not None

    @property
    def has_gbr(self) -> bool:
        return self.gbr_model is not None and getattr(self.gbr_model, "model", None) is not None

    @property
    def available_models(self) -> List[str]:
        return [
            name
            for name, present in (("prophet", self.has_prophet), ("gradient_boosting", self.has_gbr))
            if present
        ]

    # -- the models ---------------------------------------------------------- #

    def _prophet_demand(
        self, hotel_id: str, room_type: RoomType, check_in_date: date
    ) -> Optional[Dict[str, float]]:
        """Prophet's view, or ``None``. Never raises into the caller."""
        if not self.has_prophet:
            return None
        try:
            return self.prophet_bundle.demand_on(hotel_id, room_type, check_in_date)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "Prophet failed for %s/%s on %s: %s",
                hotel_id,
                room_type,
                check_in_date,
                type(exc).__name__,
            )
            return None

    def _gbr_demand(self, features: Optional[pd.DataFrame]) -> Optional[Dict[str, float]]:
        """The Gradient Boosting view, or ``None``. Never raises into the caller."""
        if not self.has_gbr or features is None or features.empty:
            return None
        try:
            return self.gbr_model.predict_one(features)
        except Exception as exc:
            logger.warning(
                "Gradient Boosting prediction failed: %s: %s", type(exc).__name__, exc
            )
            return None

    # -- blending ------------------------------------------------------------ #

    def estimate(
        self,
        hotel_id: str,
        room_type: RoomType,
        check_in_date: date,
        features: Optional[pd.DataFrame] = None,
        *,
        fallback_demand: Optional[float] = None,
    ) -> DemandEstimate:
        """Blend whatever models are available into one demand number.

        Args:
            features: The serving feature row for the Gradient Boosting model.
            fallback_demand: Historical demand for this night, used when neither
                model can answer.
        """
        weight = self.settings.model.model_prophet_blend_weight

        prophet = self._prophet_demand(hotel_id, room_type, check_in_date)
        gbr = self._gbr_demand(features)

        notes: List[str] = []
        sources: List[str] = []

        # --- neither model -------------------------------------------------
        if prophet is None and gbr is None:
            value = fallback_demand if fallback_demand is not None else self.settings.pricing.baseline_demand
            reason = (
                "no demand model was available; using stored historical demand"
                if fallback_demand is not None
                else "no demand model and no history; using the configured baseline"
            )
            logger.warning("Demand estimate degraded: %s", reason)
            return DemandEstimate(
                blended=float(value),
                prophet=None,
                gbr=None,
                weight=0.0,
                confidence=FALLBACK_CONFIDENCE,
                lower=max(float(value) - 0.20, 0.0),
                upper=float(value) + 0.20,
                sources=[],
                degraded=True,
                notes=[reason],
            )

        # --- one model -----------------------------------------------------
        if prophet is None or gbr is None:
            only = gbr if prophet is None else prophet
            name = "gradient_boosting" if prophet is None else "prophet"
            missing = "prophet" if prophet is None else "gradient_boosting"
            notes.append(f"{missing} was unavailable, so the weight collapsed onto {name}")
            logger.info("Demand estimate using %s only", name)

            spread = max(only["upper"] - only["lower"], 0.0)
            return DemandEstimate(
                blended=float(only["demand"] if "demand" in only else only["forecast"]),
                prophet=None if prophet is None else float(prophet["forecast"]),
                gbr=None if gbr is None else float(gbr["demand"]),
                weight=0.0 if prophet is None else 1.0,
                confidence=min(
                    self._confidence_from_spread(spread), SINGLE_MODEL_CONFIDENCE_CAP
                ),
                lower=float(only["lower"]),
                upper=float(only["upper"]),
                sources=[name],
                degraded=True,
                notes=notes,
            )

        # --- both models ---------------------------------------------------
        prophet_value = float(prophet["forecast"])
        gbr_value = float(gbr["demand"])
        blended = weight * prophet_value + (1.0 - weight) * gbr_value

        lower = weight * float(prophet["lower"]) + (1.0 - weight) * float(gbr["lower"])
        upper = weight * float(prophet["upper"]) + (1.0 - weight) * float(gbr["upper"])

        sources = ["prophet", "gradient_boosting"]
        disagreement = abs(prophet_value - gbr_value)
        if disagreement > 0.20:
            notes.append(
                f"the two models disagree by {disagreement:.0%}, which lowers confidence"
            )
            logger.info(
                "Model disagreement %.3f for %s/%s on %s (prophet=%.3f gbr=%.3f)",
                disagreement,
                hotel_id,
                room_type,
                check_in_date,
                prophet_value,
                gbr_value,
            )

        return DemandEstimate(
            blended=max(blended, 0.0),
            prophet=prophet_value,
            gbr=gbr_value,
            weight=weight,
            confidence=self._blended_confidence(upper - lower, disagreement),
            lower=max(lower, 0.0),
            upper=max(upper, lower),
            sources=sources,
            degraded=False,
            notes=notes,
        )

    # -- confidence ----------------------------------------------------------- #

    @staticmethod
    def _confidence_from_spread(spread: float) -> float:
        """Map an uncertainty band onto 0-1.

        A band of zero is total confidence; a band 0.5 wide (half the target's
        whole range) is none. Linear and monotone -- an elaborate calibration
        here would imply precision the residuals do not support.
        """
        if not math.isfinite(spread):
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - spread / 0.5)))

    def _blended_confidence(self, spread: float, disagreement: float) -> float:
        """Combine band width with model disagreement.

        Both must be small to earn a high number. Two confident models that
        disagree is precisely the situation where the blend is least
        trustworthy, and averaging them hides that -- so disagreement is
        penalised separately rather than being allowed to cancel out.
        """
        from_spread = self._confidence_from_spread(spread)
        from_agreement = float(max(0.0, min(1.0, 1.0 - disagreement / 0.35)))
        return round(0.6 * from_spread + 0.4 * from_agreement, 4)


__all__ = [
    "FALLBACK_CONFIDENCE",
    "SINGLE_MODEL_CONFIDENCE_CAP",
    "DemandEngine",
    "DemandEstimate",
]
