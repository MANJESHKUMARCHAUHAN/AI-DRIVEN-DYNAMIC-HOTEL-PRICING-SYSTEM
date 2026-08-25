"""The five pricing adjustments, as pure functions.

Each rule takes the situation and returns a fraction: ``+0.12`` means "twelve
percent above base". The engine sums them and multiplies once::

    raw_price = base_price x (1 + demand + occupancy + competitor + season + event)

**Additive, not chained.** ``1.12 x 1.08 x 1.05`` is 27%, not the 25% a reader
adds up in their head, and the gap widens as factors are added. Additive terms
are readable ("+12 demand, +8 occupancy, so +20"), bounded, and a revenue
manager can check them without a calculator. Surprising a revenue manager is how
a pricing system gets switched off.

**Each term is clamped before summing.** No single signal can run away with the
price, so a broken competitor feed or a wild model output moves the number by at
most its own cap. The caps live in configuration, so tuning them is an
environment change rather than a deploy.

Nothing in this module imports a framework, a database or a model. It is
arithmetic over numbers, which is what makes every rule testable in one line --
ADR-003.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from domain.enums import Season

#: Occupancy below which the engine treats inventory as distressed.
LOW_OCCUPANCY = 0.40

#: Occupancy above which a night is considered to be selling out.
HIGH_OCCUPANCY = 0.75

#: Lead time beyond which "there is still plenty of time to sell" holds.
LONG_LEAD_DAYS = 21

#: Lead time inside which the sale is nearly over.
SHORT_LEAD_DAYS = 5

#: Residual seasonal correction, on top of whatever is already in the base rate.
#: Monsoon is the only strongly negative season in the Indian market this models.
SEASON_FACTORS: Dict[Season, float] = {
    Season.WINTER: 0.06,
    Season.AUTUMN: 0.05,
    Season.SUMMER: -0.02,
    Season.MONSOON: -0.08,
}


@dataclass(frozen=True)
class Adjustment:
    """One priced signal, with the reasoning attached.

    Attributes:
        name: Stable identifier, used as a dictionary key and in the audit row.
        value: The fraction actually applied, after clamping.
        raw_value: What the rule computed before clamping. Equal to ``value``
            when nothing was clamped.
        reason: One human-readable sentence. This is what ends up in the
            explanation a revenue manager reads.
        inputs: The numbers the rule saw, so a decision can be replayed.
    """

    name: str
    value: float
    raw_value: float
    reason: str
    inputs: Dict[str, Any]

    @property
    def clamped(self) -> bool:
        return not math.isclose(self.value, self.raw_value, rel_tol=1e-9, abs_tol=1e-12)

    @property
    def percent(self) -> float:
        return self.value * 100.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": round(self.value, 6),
            "raw_value": round(self.raw_value, 6),
            "percent": round(self.percent, 2),
            "clamped": self.clamped,
            "reason": self.reason,
            "inputs": self.inputs,
        }

    def describe(self) -> str:
        """One line, as it appears in the price explanation."""
        sign = "+" if self.value >= 0 else ""
        suffix = "  (clamped)" if self.clamped else ""
        return f"{self.name.replace('_', ' ').title()}: {sign}{self.percent:.1f}%{suffix}"


def _clamp(value: float, limit: float) -> float:
    """Restrict a fraction to ``[-limit, +limit]``."""
    return max(-limit, min(limit, value))


def _finite(value: Optional[float], fallback: float = 0.0) -> float:
    """Coerce ``None``/NaN/inf to a safe number.

    Model outputs and upstream feeds both produce these, and a NaN that reaches
    the multiplication turns the whole price into NaN with no error anywhere.
    """
    if value is None:
        return fallback
    number = float(value)
    return number if math.isfinite(number) else fallback


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #


def demand_adjustment(
    blended_demand: Optional[float],
    *,
    baseline_demand: float = 0.65,
    limit: float = 0.25,
    sensitivity: float = 0.60,
) -> Adjustment:
    """Price on predicted demand relative to a normal night.

    The ratio, not the level, is what matters: 70% demand is strong for a hotel
    that usually runs at 55% and weak for one that usually runs at 85%. Using
    the ratio means one rule works for the whole estate without per-property
    tuning.

    ``sensitivity`` below 1.0 is deliberate. Demand is forecast with a real
    error bar (the Gradient Boosting model's holdout MAE is about 0.06), so
    passing the full deviation through to price would amplify forecast noise
    into rate volatility that guests notice and competitors exploit.
    """
    demand = _finite(blended_demand, baseline_demand)
    ratio = demand / max(baseline_demand, 1e-6)
    raw = (ratio - 1.0) * sensitivity
    value = _clamp(raw, limit)

    if demand > baseline_demand * 1.05:
        reason = (
            f"forecast demand {demand:.0%} is above the {baseline_demand:.0%} "
            f"baseline, so there is room to charge more"
        )
    elif demand < baseline_demand * 0.95:
        reason = (
            f"forecast demand {demand:.0%} is below the {baseline_demand:.0%} "
            f"baseline, so price has to work harder"
        )
    else:
        reason = f"forecast demand {demand:.0%} is about normal"

    return Adjustment(
        name="demand",
        value=value,
        raw_value=raw,
        reason=reason,
        inputs={"blended_demand": round(demand, 4), "baseline_demand": baseline_demand},
    )


def occupancy_adjustment(
    occupancy_rate: Optional[float],
    days_to_checkin: Optional[int],
    *,
    limit: float = 0.20,
) -> Adjustment:
    """Price on how full we are *and* how long is left to sell.

    The one genuinely non-obvious rule. Occupancy alone does not justify a
    price move -- it depends entirely on the time remaining:

    =====================  ==============================  =========================
    \\                      far out (>21 days)              near (<5 days)
    =====================  ==============================  =========================
    **high occupancy**     raise hard: demand is real       raise a little: nearly
                           and it arrived early             sold out anyway
    **low occupancy**      hold: there is time to sell      discount: the room
                                                            perishes tonight
    =====================  ==============================  =========================

    An unsold room-night is worth exactly zero at midnight, which is why the
    low-occupancy/short-lead corner discounts rather than holds. The
    high-occupancy/short-lead corner raises only modestly because there is
    little inventory left for the increase to earn anything on.
    """
    occupancy = _finite(occupancy_rate, 0.0)
    lead = int(_finite(days_to_checkin, 0.0))

    # 1.0 far out, 0.0 on the day. This is the "how much time is left" dial.
    time_left = min(max(lead, 0), LONG_LEAD_DAYS) / LONG_LEAD_DAYS

    if occupancy >= HIGH_OCCUPANCY:
        # Selling fast. Reward early pace much more than late pace.
        strength = (occupancy - HIGH_OCCUPANCY) / max(1.0 - HIGH_OCCUPANCY, 1e-6)
        raw = limit * strength * (0.35 + 0.65 * time_left)
        reason = (
            f"{occupancy:.0%} sold with {lead} day(s) to go"
            + (
                " -- selling early, push the rate"
                if lead >= LONG_LEAD_DAYS
                else " -- nearly full, a modest rise"
            )
        )
    elif occupancy <= LOW_OCCUPANCY:
        # Behind pace. Only discount once time is genuinely short.
        shortfall = (LOW_OCCUPANCY - occupancy) / max(LOW_OCCUPANCY, 1e-6)
        urgency = 1.0 - time_left
        raw = -limit * shortfall * urgency
        if lead <= SHORT_LEAD_DAYS:
            reason = (
                f"only {occupancy:.0%} sold with {lead} day(s) left -- discount, "
                f"an empty room earns nothing tonight"
            )
        else:
            reason = (
                f"{occupancy:.0%} sold but {lead} day(s) remain -- hold, there is "
                f"still time to sell"
            )
    else:
        raw = 0.0
        reason = f"{occupancy:.0%} sold at {lead} day(s) out is on pace"

    return Adjustment(
        name="occupancy",
        value=_clamp(raw, limit),
        raw_value=raw,
        reason=reason,
        inputs={
            "occupancy_rate": round(occupancy, 4),
            "days_to_checkin": lead,
            "time_left": round(time_left, 3),
        },
    )


def competitor_adjustment(
    base_price: float,
    competitor_rate: Optional[float],
    *,
    competitor_missing: bool = False,
    limit: float = 0.15,
    sensitivity: float = 0.50,
) -> Adjustment:
    """Move part of the way towards the competitive set's rate.

    Only *part* of the way, and clamped. Following the market one-for-one is how
    two automated pricing systems talk each other into a race to the bottom;
    moving half the gap converges without either side chasing.

    When no competitor rate was visible the adjustment is exactly zero rather
    than a guess. Pricing against an imagined market is worse than pricing
    against none.
    """
    if competitor_missing or competitor_rate is None:
        return Adjustment(
            name="competitor",
            value=0.0,
            raw_value=0.0,
            reason="no competitor rate was visible, so the market is not priced in",
            inputs={"competitor_rate": None, "competitor_missing": True},
        )

    rate = _finite(competitor_rate, base_price)
    if rate <= 0 or base_price <= 0:
        return Adjustment(
            name="competitor",
            value=0.0,
            raw_value=0.0,
            reason="competitor rate was not usable, so the market is not priced in",
            inputs={"competitor_rate": rate},
        )

    gap = (rate - base_price) / base_price
    raw = gap * sensitivity
    value = _clamp(raw, limit)

    if gap > 0.02:
        reason = (
            f"the market is {gap:+.0%} above our base rate, so there is headroom"
        )
    elif gap < -0.02:
        reason = f"the market is {gap:+.0%} below our base rate, so we are exposed"
    else:
        reason = "we are priced level with the market"

    return Adjustment(
        name="competitor",
        value=value,
        raw_value=raw,
        reason=reason,
        inputs={
            "competitor_rate": round(rate, 2),
            "base_price": round(base_price, 2),
            "gap": round(gap, 4),
        },
    )


def season_adjustment(
    season: Optional[Season], *, limit: float = 0.10
) -> Adjustment:
    """A small residual correction for the time of year.

    Small on purpose. Whoever set the room's base rate already had the season in
    mind; this only nudges the residual, and a large seasonal term here would
    double-count.
    """
    if season is None:
        return Adjustment(
            name="season",
            value=0.0,
            raw_value=0.0,
            reason="no season was supplied",
            inputs={"season": None},
        )

    raw = SEASON_FACTORS.get(season, 0.0)
    value = _clamp(raw, limit)
    direction = "supports" if raw > 0 else ("weakens" if raw < 0 else "is neutral for")
    return Adjustment(
        name="season",
        value=value,
        raw_value=raw,
        reason=f"{season.value} {direction} rates in this market",
        inputs={"season": season.value},
    )


def event_adjustment(
    event_score: Optional[float],
    *,
    is_weekend: bool = False,
    is_holiday: bool = False,
    limit: float = 0.15,
) -> Adjustment:
    """Price city events, holidays and the weekend.

    Combined with diminishing returns rather than added: a festival on a bank
    holiday weekend is busier than any one of them, but the city cannot be more
    than full. The formula is ``1 - product(1 - effect)`` -- the standard "at
    least one of these happened" combination.
    """
    score = max(0.0, min(1.0, _finite(event_score, 0.0)))

    effects = []
    reasons = []
    if score > 0.01:
        effects.append(score * 0.9)
        reasons.append(f"a local event scoring {score:.2f}")
    if is_holiday:
        effects.append(0.5)
        reasons.append("a public holiday")
    if is_weekend:
        effects.append(0.3)
        reasons.append("a weekend")

    remaining = 1.0
    for effect in effects:
        remaining *= 1.0 - effect
    raw = limit * (1.0 - remaining)

    reason = (
        f"demand pressure from {', '.join(reasons)}"
        if reasons
        else "no event, holiday or weekend pressure"
    )
    return Adjustment(
        name="event",
        value=_clamp(raw, limit),
        raw_value=raw,
        reason=reason,
        inputs={
            "event_score": round(score, 4),
            "is_holiday": is_holiday,
            "is_weekend": is_weekend,
        },
    )


__all__ = [
    "HIGH_OCCUPANCY",
    "LONG_LEAD_DAYS",
    "LOW_OCCUPANCY",
    "SEASON_FACTORS",
    "SHORT_LEAD_DAYS",
    "Adjustment",
    "competitor_adjustment",
    "demand_adjustment",
    "event_adjustment",
    "occupancy_adjustment",
    "season_adjustment",
]
