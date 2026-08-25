"""Business guardrails: the last gate before a price is allowed out.

Guardrails exist because a model is a statistical statement and a price is a
commercial commitment. The model can be right on average and still produce a
number that would embarrass the hotel on a particular night -- a rate below cost
because a competitor feed glitched, or a 60% overnight rise because a festival
was double-counted. These rules make that impossible rather than unlikely.

**Structurally unbypassable.** :func:`apply` is the only function that can
construct a :class:`FinalPrice`; constructing one anywhere else raises. The
pricing engine returns a :class:`RawPrice`, the API can only serialise a
``FinalPrice``, so there is no code path that reaches a caller without passing
through here. A guardrail that a future refactor can route around is a comment,
not a control.

**Order is fixed: relative rules first, absolute rules last.** A floor that a
relative rule can undercut is not a floor. So the daily-change cap and the
competitor bands are applied while the number is still free to move, and
``MIN_PRICE``/``MAX_PRICE`` clamp whatever survives.

**Every rule that fires is recorded and logged at WARNING.** A guardrail firing
occasionally is the system working; a guardrail firing constantly means the
model needs retuning, and that only gets noticed if it is visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from config import Settings, get_settings
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

#: Private construction token. Held by this module alone, which is what makes
#: `apply` the only way to obtain a FinalPrice.
_CONSTRUCTION_TOKEN = object()

#: A model output beyond this multiple of base price is treated as broken rather
#: than as an extreme but valid recommendation.
SANITY_MULTIPLE = 5.0


class Rule(str, Enum):
    """Identifiers for every guardrail, stable enough to alert on."""

    SANITY = "SANITY_CHECK"
    LOW_OCCUPANCY_BLOCK = "LOW_OCCUPANCY_RISE_BLOCK"
    MAX_DAILY_RISE = "MAX_DAILY_RISE"
    MAX_DAILY_FALL = "MAX_DAILY_FALL"
    COMPETITOR_UPPER = "COMPETITOR_UPPER_BAND"
    COMPETITOR_LOWER = "COMPETITOR_LOWER_BAND"
    ROOM_CEILING = "ROOM_CEILING"
    ROOM_FLOOR = "ROOM_FLOOR"
    MIN_PRICE = "MIN_PRICE_FLOOR"
    MAX_PRICE = "MAX_PRICE_CEILING"


@dataclass(frozen=True)
class GuardrailHit:
    """A rule that changed the price, with the before and after."""

    rule: Rule
    before: float
    after: float
    reason: str

    @property
    def delta(self) -> float:
        return self.after - self.before

    def as_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.rule.value,
            "before": round(self.before, 2),
            "after": round(self.after, 2),
            "delta": round(self.delta, 2),
            "reason": self.reason,
        }

    def describe(self) -> str:
        return f"{self.rule.value}: {self.before:,.0f} -> {self.after:,.0f} ({self.reason})"


@dataclass(frozen=True)
class GuardrailContext:
    """Everything the rules need to judge a price.

    Attributes:
        base_price: The room's configured rate, and the fallback if the model
            output turns out to be unusable.
        current_price: Yesterday's price, if there was one. Without it the
            daily-change cap cannot apply and is skipped rather than guessed.
        occupancy_rate: On-the-books occupancy, for the low-occupancy block.
        competitor_min_rate / competitor_max_rate: The competitive band. ``None``
            disables the band rules -- pricing against an imagined market is
            worse than pricing against none.
        room_floor_price / room_ceiling_price: Optional per-room overrides that
            sit inside the global limits.
        allow_increase_override: Set when demand is exceptionally strong, which
            lets the engine price above the competitive band. Never lets it
            through the absolute limits.
    """

    base_price: float
    current_price: Optional[float] = None
    occupancy_rate: Optional[float] = None
    competitor_min_rate: Optional[float] = None
    competitor_max_rate: Optional[float] = None
    room_floor_price: Optional[float] = None
    room_ceiling_price: Optional[float] = None
    allow_increase_override: bool = False


@dataclass(frozen=True)
class RawPrice:
    """The engine's recommendation, before any guardrail has seen it.

    Deliberately a distinct type from :class:`FinalPrice`. Anything that wants
    to serve a price must hold a ``FinalPrice``, and the only way to get one is
    :func:`apply`.
    """

    amount: float
    base_price: float
    total_adjustment: float
    breakdown: Dict[str, Any] = field(default_factory=dict)


class FinalPrice:
    """A price that has passed every guardrail.

    Cannot be constructed directly::

        >>> FinalPrice(amount=1.0, raw=..., applied=[])
        Traceback (most recent call last):
        TypeError: FinalPrice can only be created by guardrails.apply()
    """

    __slots__ = ("amount", "raw", "applied")

    def __init__(
        self,
        *,
        amount: float,
        raw: RawPrice,
        applied: List[GuardrailHit],
        _token: Any = None,
    ) -> None:
        if _token is not _CONSTRUCTION_TOKEN:
            raise TypeError(
                "FinalPrice can only be created by guardrails.apply(). "
                "A price that has not passed the guardrails must never reach a "
                "caller -- see docs/architecture.md §11."
            )
        self.amount = amount
        self.raw = raw
        self.applied = applied

    @property
    def was_clamped(self) -> bool:
        return bool(self.applied)

    @property
    def rules_applied(self) -> List[str]:
        """Rule identifiers, in the order they fired. Goes into the API response."""
        return [hit.rule.value for hit in self.applied]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "final_price": round(self.amount, 2),
            "raw_price": round(self.raw.amount, 2),
            "was_clamped": self.was_clamped,
            "guardrails_applied": [hit.as_dict() for hit in self.applied],
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<FinalPrice {self.amount:,.2f} "
            f"(raw {self.raw.amount:,.2f}, {len(self.applied)} rule(s))>"
        )


def _round_to(amount: float, places: int) -> float:
    """Round a price to the configured precision."""
    return round(amount, places) if places else float(round(amount))


def apply(
    raw: RawPrice,
    context: GuardrailContext,
    settings: Optional[Settings] = None,
) -> FinalPrice:
    """Run every guardrail in the fixed order and return a servable price.

    Args:
        raw: The engine's recommendation.
        context: What the rules need to judge it.
        settings: Configuration holding the thresholds.

    Returns:
        A :class:`FinalPrice`, with every rule that fired recorded on it.
    """
    settings = settings or get_settings()
    limits = settings.pricing
    applied: List[GuardrailHit] = []
    price = float(raw.amount)

    def record(rule: Rule, after: float, reason: str) -> float:
        nonlocal price
        if math.isclose(after, price, rel_tol=1e-9, abs_tol=0.005):
            return price
        hit = GuardrailHit(rule=rule, before=price, after=after, reason=reason)
        applied.append(hit)
        logger.warning("Guardrail %s", hit.describe())
        price = after
        return price

    # --- 0. sanity ---------------------------------------------------------
    # A non-finite or absurd number is a broken model, not an aggressive one.
    # Fall back to the base rate: knowably safe beats confidently wrong.
    if not math.isfinite(price) or price <= 0:
        price = record(
            Rule.SANITY,
            context.base_price,
            f"recommended price was {raw.amount!r}; fell back to the base rate",
        )
    elif price > context.base_price * SANITY_MULTIPLE:
        price = record(
            Rule.SANITY,
            context.base_price * SANITY_MULTIPLE,
            f"recommended price exceeded {SANITY_MULTIPLE:g}x the base rate, "
            f"which indicates a model fault rather than an opportunity",
        )

    # --- 1. relative rules -------------------------------------------------
    # Applied while the number can still move. Absolutes come last and win.

    # 1a. Never raise the price of a hotel that is not selling. Enforced here
    # and again in the occupancy rule -- belt and braces, deliberately.
    if (
        context.occupancy_rate is not None
        and context.occupancy_rate < limits.low_occupancy_threshold
        and context.current_price is not None
        and price > context.current_price
    ):
        price = record(
            Rule.LOW_OCCUPANCY_BLOCK,
            context.current_price,
            f"occupancy {context.occupancy_rate:.0%} is below the "
            f"{limits.low_occupancy_threshold:.0%} threshold, so no increase is allowed",
        )

    # 1b. Day-over-day movement cap. Guests notice volatility, and a rate that
    # jumps 40% overnight reads as a pricing error whether or not it is one.
    if context.current_price is not None and context.current_price > 0:
        cap = limits.max_daily_change_percent
        ceiling = context.current_price * (1.0 + cap)
        floor = context.current_price * (1.0 - cap)
        if price > ceiling:
            price = record(
                Rule.MAX_DAILY_RISE,
                ceiling,
                f"rise capped at {cap:.0%} per day from {context.current_price:,.0f}",
            )
        elif price < floor:
            price = record(
                Rule.MAX_DAILY_FALL,
                floor,
                f"fall capped at {cap:.0%} per day from {context.current_price:,.0f}",
            )

    # 1c. Stay within touching distance of the competitive set. Skipped when
    # demand is exceptional -- a sold-out city is exactly when a hotel should be
    # allowed above the market.
    if context.competitor_max_rate and not context.allow_increase_override:
        ceiling = context.competitor_max_rate * (
            1.0 + limits.competitor_upper_bound_percent
        )
        if price > ceiling:
            price = record(
                Rule.COMPETITOR_UPPER,
                ceiling,
                f"capped at {limits.competitor_upper_bound_percent:.0%} above the "
                f"highest competitor rate ({context.competitor_max_rate:,.0f})",
            )

    if context.competitor_min_rate:
        floor = context.competitor_min_rate * (
            1.0 - limits.competitor_lower_bound_percent
        )
        if price < floor:
            price = record(
                Rule.COMPETITOR_LOWER,
                floor,
                f"held at {limits.competitor_lower_bound_percent:.0%} below the "
                f"lowest competitor rate ({context.competitor_min_rate:,.0f}); "
                f"undercutting further buys volume we cannot service",
            )

    # --- 2. absolute rules --------------------------------------------------
    # These must win. Anything above can only move the price within them.

    if context.room_ceiling_price and price > context.room_ceiling_price:
        price = record(
            Rule.ROOM_CEILING,
            context.room_ceiling_price,
            "capped at this room type's configured ceiling",
        )
    if context.room_floor_price and price < context.room_floor_price:
        price = record(
            Rule.ROOM_FLOOR,
            context.room_floor_price,
            "raised to this room type's configured floor",
        )

    if price > limits.max_price:
        price = record(
            Rule.MAX_PRICE,
            limits.max_price,
            f"capped at the absolute ceiling of {limits.currency} {limits.max_price:,.0f}",
        )
    if price < limits.min_price:
        price = record(
            Rule.MIN_PRICE,
            limits.min_price,
            f"raised to the absolute floor of {limits.currency} {limits.min_price:,.0f}",
        )

    return FinalPrice(
        amount=_round_to(price, limits.price_rounding),
        raw=raw,
        applied=applied,
        _token=_CONSTRUCTION_TOKEN,
    )


def describe(final: FinalPrice, currency: str = "INR") -> List[str]:
    """The guardrail section of a human-readable price explanation."""
    if not final.applied:
        return ["No guardrails were triggered."]
    return [
        f"{hit.rule.value}: {currency} {hit.before:,.0f} -> {currency} {hit.after:,.0f}"
        f"  ({hit.reason})"
        for hit in final.applied
    ]


__all__ = [
    "SANITY_MULTIPLE",
    "FinalPrice",
    "GuardrailContext",
    "GuardrailHit",
    "RawPrice",
    "Rule",
    "apply",
    "describe",
]
