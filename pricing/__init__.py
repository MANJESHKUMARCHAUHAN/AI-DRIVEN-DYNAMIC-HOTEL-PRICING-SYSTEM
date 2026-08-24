"""The business core: demand blending, price computation, guardrails.

This package imports **no framework**. No FastAPI, no SQLAlchemy, no Kafka --
only the standard library, NumPy and configuration types. That constraint
(ADR-003) is what lets the pricing logic be tested in milliseconds without a
database or an HTTP server, and it is the rule most worth being pedantic about.

Guardrails are a hard gate, not a convention: :mod:`pricing.pricing_engine`
yields a ``RawPrice``, and the only function that turns a ``RawPrice`` into a
``FinalPrice`` is ``guardrails.apply``. The API can only serialise a
``FinalPrice``, so no code path can emit an unchecked number.

Phase 7 implements ``demand_engine``, ``pricing_engine``, ``guardrails``, ``rules``.
"""

__all__: list = []
