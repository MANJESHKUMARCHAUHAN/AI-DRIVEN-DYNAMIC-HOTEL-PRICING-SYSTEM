"""The business core: demand blending, price computation, guardrails.

This package imports **no framework**. No FastAPI, no SQLAlchemy, no Kafka --
only the standard library, NumPy and configuration types. That constraint
(ADR-003) is what lets the pricing logic be tested in milliseconds without a
database or an HTTP server, and it is the rule most worth being pedantic about.

Guardrails are a hard gate, not a convention: :mod:`pricing.pricing_engine`
yields a ``RawPrice``, and the only function that turns a ``RawPrice`` into a
``FinalPrice`` is ``guardrails.apply``. The API can only serialise a
``FinalPrice``, so no code path can emit an unchecked number.

Module map::

    rules.py           the five adjustments, as pure functions
    demand_engine.py   blends Prophet and the GBR into one number + confidence
    pricing_engine.py  orchestrates: base -> adjustments -> raw -> final
    guardrails.py      the last gate; the only source of a FinalPrice
"""

from pricing.demand_engine import DemandEngine, DemandEstimate
from pricing.guardrails import (
    FinalPrice,
    GuardrailContext,
    GuardrailHit,
    RawPrice,
    Rule,
    apply,
)
from pricing.pricing_engine import PriceDecision, PricingEngine, PricingRequest
from pricing.rules import (
    Adjustment,
    competitor_adjustment,
    demand_adjustment,
    event_adjustment,
    occupancy_adjustment,
    season_adjustment,
)

__all__ = [
    "Adjustment",
    "DemandEngine",
    "DemandEstimate",
    "FinalPrice",
    "GuardrailContext",
    "GuardrailHit",
    "PriceDecision",
    "PricingEngine",
    "PricingRequest",
    "RawPrice",
    "Rule",
    "apply",
    "competitor_adjustment",
    "demand_adjustment",
    "event_adjustment",
    "occupancy_adjustment",
    "season_adjustment",
]
