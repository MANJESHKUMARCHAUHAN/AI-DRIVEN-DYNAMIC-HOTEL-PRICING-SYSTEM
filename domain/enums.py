"""The domain's closed vocabularies.

Every one of these is a ``str`` subclass, so a member compares equal to its own
value. That is what lets the same enum be a SQLAlchemy column type, a Pydantic
field, a JSON value and a dictionary key without conversion at any boundary.

This module imports nothing -- not SQLAlchemy, not Pydantic, not ``config``.
It is the bottom of the dependency graph, and ``tests/test_architecture.py``
asserts that it stays there.
"""

from __future__ import annotations

from enum import Enum


class RoomType(str, Enum):
    """Sellable room categories, cheapest to most expensive."""

    STANDARD = "standard"
    DELUXE = "deluxe"
    PREMIUM = "premium"
    SUITE = "suite"


class Competitor(str, Enum):
    """Rate sources we observe.

    Four brands rather than two: ``competitor_min_rate`` and
    ``competitor_max_rate`` only carry information if there is a spread to
    measure. Live scrapers exist for ``booking`` and ``expedia`` only; the
    synthetic generator and the demo OTA produce all four.
    """

    BOOKING = "booking"
    EXPEDIA = "expedia"
    AGODA = "agoda"
    MAKEMYTRIP = "makemytrip"


class Season(str, Enum):
    """Indian seasonal calendar. Drives both demand and rate levels."""

    WINTER = "winter"  # Dec-Feb  -- peak for leisure/coastal
    SUMMER = "summer"  # Mar-May  -- hill stations peak, plains slump
    MONSOON = "monsoon"  # Jun-Sep  -- trough almost everywhere
    AUTUMN = "autumn"  # Oct-Nov  -- festival and wedding season


class MarketSegment(str, Enum):
    """What a property mainly sells. Decides weekday vs weekend demand shape."""

    BUSINESS = "business"
    LEISURE = "leisure"
    MIXED = "mixed"


class ModelType(str, Enum):
    PROPHET = "prophet"
    GRADIENT_BOOSTING = "gradient_boosting"


class RunStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


__all__ = [
    "Competitor",
    "MarketSegment",
    "ModelType",
    "RoomType",
    "RunStatus",
    "Season",
]
