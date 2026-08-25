"""The pricing engine: demand and market conditions in, a defensible price out.

The whole calculation, start to finish::

    base_price x (1 + demand + occupancy + competitor + season + event)
      = raw_price
    guardrails(raw_price, context)
      = final_price

Five adjustments, each computed by a pure function in :mod:`pricing.rules`, each
clamped, summed rather than chained. Then :func:`pricing.guardrails.apply`, which
is the only thing that can produce a servable :class:`~pricing.guardrails.FinalPrice`.

The engine's real product is not the number, it is the **explanation**. Every
decision carries the inputs it saw, the five adjustments with a sentence each,
the raw price, every guardrail that fired with before and after values, and the
final price. That record is what lets someone answer "why was room 204 priced at
7,340 last Tuesday" three months later without re-running anything -- and it is
what makes an automated pricing system something a revenue manager will actually
leave switched on.

No framework, database or model imports here: the engine takes numbers and
returns numbers (ADR-003). The API layer assembles the inputs; this module does
the arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from config import Settings, get_settings
from domain.enums import RoomType, Season
from monitoring.logging_config import get_logger
from pricing.demand_engine import DemandEstimate
from pricing.guardrails import (
    FinalPrice,
    GuardrailContext,
    RawPrice,
    apply as apply_guardrails,
)
from pricing.rules import (
    Adjustment,
    competitor_adjustment,
    demand_adjustment,
    event_adjustment,
    occupancy_adjustment,
    season_adjustment,
)

logger = get_logger(__name__)

#: Demand above this is "exceptional" and lets the price out of the competitor
#: band. A hotel in a city that is genuinely selling out should be allowed above
#: the market -- that is the whole point of dynamic pricing.
EXCEPTIONAL_DEMAND = 0.92


@dataclass
class PricingRequest:
    """Everything the engine needs to price one room-night.

    Only ``hotel_id``, ``room_type``, ``check_in_date`` and ``base_price`` are
    required. Every other field narrows the estimate; anything absent simply
    means that signal contributes nothing, which is why a caller with partial
    information still gets a usable, honest price.
    """

    hotel_id: str
    room_type: RoomType
    check_in_date: date
    base_price: float

    current_price: Optional[float] = None
    occupancy_rate: Optional[float] = None
    available_rooms: Optional[int] = None
    total_rooms: Optional[int] = None
    days_to_checkin: Optional[int] = None

    competitor_rate: Optional[float] = None
    competitor_min_rate: Optional[float] = None
    competitor_max_rate: Optional[float] = None
    competitor_missing: bool = False

    season: Optional[Season] = None
    is_weekend: bool = False
    is_holiday: bool = False
    event_score: float = 0.0

    room_floor_price: Optional[float] = None
    room_ceiling_price: Optional[float] = None

    def resolved_days_to_checkin(self, as_of: Optional[date] = None) -> int:
        """Lead time, from the field if given or from the calendar otherwise."""
        if self.days_to_checkin is not None:
            return max(int(self.days_to_checkin), 0)
        reference = as_of or datetime.now(timezone.utc).date()
        return max((self.check_in_date - reference).days, 0)


@dataclass
class PriceDecision:
    """A priced room-night and the complete reasoning behind it."""

    hotel_id: str
    room_type: RoomType
    check_in_date: date
    base_price: float
    current_price: Optional[float]
    adjustments: List[Adjustment]
    total_adjustment: float
    raw_price: float
    final_price: float
    price_change_percent: float
    demand: DemandEstimate
    guardrails: List[Dict[str, Any]] = field(default_factory=list)
    priced_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def adjustment(self, name: str) -> Optional[Adjustment]:
        return next((a for a in self.adjustments if a.name == name), None)

    def adjustment_map(self) -> Dict[str, float]:
        """Adjustment name -> fraction. What the audit row stores."""
        return {a.name: round(a.value, 6) for a in self.adjustments}

    @property
    def guardrails_applied(self) -> List[str]:
        return [hit["rule"] for hit in self.guardrails]

    def as_dict(self) -> Dict[str, Any]:
        """The full record, ready for JSON or the ``pricing_decisions`` table."""
        return {
            "hotel_id": self.hotel_id,
            "room_type": self.room_type.value,
            "check_in_date": self.check_in_date.isoformat(),
            "base_price": round(self.base_price, 2),
            "current_price": (
                round(self.current_price, 2) if self.current_price is not None else None
            ),
            "adjustments": [a.as_dict() for a in self.adjustments],
            "total_adjustment": round(self.total_adjustment, 6),
            "raw_recommended_price": round(self.raw_price, 2),
            "final_recommended_price": round(self.final_price, 2),
            "price_change_percent": round(self.price_change_percent, 2),
            "guardrails_applied": self.guardrails_applied,
            "guardrail_detail": self.guardrails,
            "demand": self.demand.as_dict(),
            "priced_at": self.priced_at.isoformat(),
        }

    def explain(self, currency: str = "INR") -> str:
        """The decision as a revenue manager would want to read it.

        This is the output requirement 14 asks for, and the thing that gets
        pasted into a ticket when someone queries a rate.
        """
        lines = [
            f"{self.hotel_id} / {self.room_type.value} / {self.check_in_date}",
            "=" * 62,
            f"Base price:            {currency} {self.base_price:>10,.0f}"
            f"   (the rate adjustments apply to)",
        ]
        # Both anchors, always. The adjustments are a percentage of *base*
        # while the headline change is against *current*, and those two numbers
        # can point in opposite directions when a hotel is discounting off its
        # rack rate. Showing only one of them reads as a contradiction.
        if self.current_price:
            lines.append(
                f"Current price:         {currency} {self.current_price:>10,.0f}"
                f"   (what we charge today)"
            )
        lines += ["", "Adjustments"]
        for adjustment in self.adjustments:
            # ``+6.1f`` rather than a hand-prepended sign: the sign counts
            # towards the field width, so positives and negatives line up.
            flag = "  [clamped]" if adjustment.clamped else ""
            lines.append(
                f"  {adjustment.name.title():<12} {adjustment.percent:+7.1f}%"
                f"{flag}   {adjustment.reason}"
            )

        lines += [
            "",
            f"  {'Total':<12} {self.total_adjustment * 100:+7.1f}%",
            "",
            f"Raw price:             {currency} {self.raw_price:>10,.0f}"
            f"   (base {self.total_adjustment * 100:+.1f}%)",
        ]

        if self.guardrails:
            lines.append("")
            lines.append("Guardrails")
            for hit in self.guardrails:
                lines.append(
                    f"  {hit['rule']:<24} {currency} {hit['before']:>9,.0f} -> "
                    f"{currency} {hit['after']:>9,.0f}"
                )
                lines.append(f"  {'':<24} {hit['reason']}")
        else:
            lines += ["", "Guardrails: none triggered"]

        change = f"{self.price_change_percent:+.1f}% vs current" if self.current_price else ""
        lines += [
            "",
            f"Final price:           {currency} {self.final_price:>10,.0f}   {change}",
            "-" * 62,
            f"Demand {self.demand.blended:.0%} "
            f"(prophet {self._fmt(self.demand.prophet)}, "
            f"gbr {self._fmt(self.demand.gbr)}), "
            f"confidence {self.demand.confidence:.0%}",
        ]
        return "\n".join(lines)

    @staticmethod
    def _fmt(value: Optional[float]) -> str:
        return f"{value:.0%}" if value is not None else "n/a"


class PricingEngine:
    """Turns a demand estimate and market conditions into a defensible price.

    Example::

        engine = PricingEngine()
        decision = engine.price(request, demand_estimate)
        print(decision.explain())
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    # -- the calculation ------------------------------------------------------ #

    def compute_adjustments(
        self, request: PricingRequest, demand: DemandEstimate
    ) -> List[Adjustment]:
        """The five priced signals, in the order they are reported."""
        limits = self.settings.pricing
        lead = request.resolved_days_to_checkin()

        return [
            demand_adjustment(
                demand.blended,
                baseline_demand=limits.baseline_demand,
                limit=limits.max_demand_adjustment,
            ),
            occupancy_adjustment(
                request.occupancy_rate,
                lead,
                limit=limits.max_occupancy_adjustment,
            ),
            competitor_adjustment(
                request.base_price,
                request.competitor_rate,
                competitor_missing=request.competitor_missing,
                limit=limits.max_competitor_adjustment,
            ),
            season_adjustment(request.season, limit=limits.max_season_adjustment),
            event_adjustment(
                request.event_score,
                is_weekend=request.is_weekend,
                is_holiday=request.is_holiday,
                limit=limits.max_event_adjustment,
            ),
        ]

    def price(
        self, request: PricingRequest, demand: DemandEstimate
    ) -> PriceDecision:
        """Price one room-night, end to end.

        Args:
            request: The situation.
            demand: The blended demand estimate from :class:`~pricing.demand_engine.DemandEngine`.

        Returns:
            The decision, with the full explanation attached.
        """
        if request.base_price <= 0:
            raise ValueError(
                f"base_price must be positive, got {request.base_price!r} for "
                f"{request.hotel_id}/{request.room_type}"
            )

        adjustments = self.compute_adjustments(request, demand)
        total = sum(a.value for a in adjustments)

        # A low-confidence estimate should move the price less. Scaling the
        # whole multiplier is simpler and safer than scaling each term, and it
        # means "the models are unsure" degrades towards the base rate rather
        # than towards an arbitrary number.
        confidence_scale = self._confidence_scale(demand.confidence)
        scaled_total = total * confidence_scale

        raw_amount = request.base_price * (1.0 + scaled_total)

        raw = RawPrice(
            amount=raw_amount,
            base_price=request.base_price,
            total_adjustment=scaled_total,
            breakdown={
                "adjustments": [a.as_dict() for a in adjustments],
                "unscaled_total": round(total, 6),
                "confidence_scale": round(confidence_scale, 4),
            },
        )

        context = GuardrailContext(
            base_price=request.base_price,
            current_price=request.current_price,
            occupancy_rate=request.occupancy_rate,
            competitor_min_rate=request.competitor_min_rate,
            competitor_max_rate=request.competitor_max_rate,
            room_floor_price=request.room_floor_price,
            room_ceiling_price=request.room_ceiling_price,
            # Exceptional demand earns the right to price above the market.
            allow_increase_override=demand.blended >= EXCEPTIONAL_DEMAND,
        )
        final: FinalPrice = apply_guardrails(raw, context, self.settings)

        reference = request.current_price or request.base_price
        change_percent = (
            (final.amount - reference) / reference * 100.0 if reference else 0.0
        )

        decision = PriceDecision(
            hotel_id=request.hotel_id,
            room_type=request.room_type,
            check_in_date=request.check_in_date,
            base_price=request.base_price,
            current_price=request.current_price,
            adjustments=adjustments,
            total_adjustment=scaled_total,
            raw_price=raw.amount,
            final_price=final.amount,
            price_change_percent=change_percent,
            demand=demand,
            guardrails=[hit.as_dict() for hit in final.applied],
        )

        logger.info(
            "Priced %s/%s %s: base=%.0f raw=%.0f final=%.0f (%+.1f%%) "
            "demand=%.3f confidence=%.2f guardrails=%s",
            request.hotel_id,
            request.room_type.value,
            request.check_in_date,
            request.base_price,
            raw.amount,
            final.amount,
            change_percent,
            demand.blended,
            demand.confidence,
            decision.guardrails_applied or "none",
        )
        return decision

    # -- helpers --------------------------------------------------------------- #

    @staticmethod
    def _confidence_scale(confidence: float) -> float:
        """How much of the computed adjustment to actually apply.

        Never zero and never above one: even a low-confidence estimate carries
        some signal, and a high-confidence one should not be amplified beyond
        what the rules already allow. The floor of 0.5 means an unusable model
        halves the move rather than disabling pricing entirely.
        """
        return float(max(0.5, min(1.0, 0.5 + 0.5 * confidence)))


__all__ = [
    "EXCEPTIONAL_DEMAND",
    "PriceDecision",
    "PricingEngine",
    "PricingRequest",
]
