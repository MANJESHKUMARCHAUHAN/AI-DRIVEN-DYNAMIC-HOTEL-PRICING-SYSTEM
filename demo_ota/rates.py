"""Rate generation for the demo OTA.

The numbers here are shaped like a real market: city tier sets the base, weekends
and holiday windows lift it, long lead times discount it, and each property has a
persistent position relative to its local set. Given the same query it returns
the same rates, so a test can assert on them.

WHY THIS DOES NOT REUSE ``features.calendars``
----------------------------------------------
It would be easy -- and wrong -- to price this "competitor" using the same
``event_score`` the feature pipeline uses. That would make ``competitor_avg_price``
a deterministic function of ``local_event_score``, the gradient booster would
discover a relationship that exists only because we created it, and the
competitor feature would look far more predictive than any real feed ever is.

This project has already been bitten by exactly that: ``search_demand`` was
0.83-correlated with the target until the generator was changed to compress it
and add multiplicative noise. The lesson generalises. So the calendar below is
*deliberately different* from ours -- different windows, different weights, and
per-observation noise on top. The two overlap where reality overlaps (everyone is
busy at New Year) and diverge everywhere else, which is what a real competitive
set looks like.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

#: Room categories this OTA lists, matching our own four.
ROOM_TYPES: Tuple[str, ...] = ("standard", "deluxe", "premium", "suite")

#: Multiplier applied to a city's base rate for each category.
_ROOM_MULTIPLIER: Dict[str, float] = {
    "standard": 1.00,
    "deluxe": 1.38,
    "premium": 1.72,
    "suite": 2.45,
}

#: Nightly base rate by city, in INR, for a standard room in a normal week.
#: Loosely calibrated against mid-market Indian hotel pricing.
_CITY_BASE: Dict[str, float] = {
    "Goa": 6200.0,
    "Jaipur": 4800.0,
    "Udaipur": 7100.0,
    "Bengaluru": 5400.0,
    "Mumbai": 8200.0,
    "New Delhi": 6600.0,
}

#: Fallback for a city we do not have a base for. Requests for unknown cities
#: still return rates rather than an error: an OTA does not 404 because it has
#: thin inventory somewhere, and the scraper should handle a sparse result.
_DEFAULT_BASE = 5500.0

#: The properties this OTA lists per city. Each carries a persistent position
#: relative to the local base -- a fixed hierarchy, as real competitive sets have.
_PROPERTIES: Dict[str, Tuple[Tuple[str, float], ...]] = {
    "Goa": (
        ("Sea Breeze Resort & Spa", 1.14),
        ("Palm Court Goa", 0.92),
        ("The Anjuna House", 1.31),
        ("Calangute Bay Hotel", 0.86),
    ),
    "Jaipur": (
        ("Amber Fort View Hotel", 1.08),
        ("Pink City Residency", 0.89),
        ("Rajputana Heritage Palace", 1.42),
        ("Johari Bazaar Inn", 0.81),
    ),
    "Udaipur": (
        ("Lake Pichola Grand", 1.36),
        ("City Palace Retreat", 1.12),
        ("Fateh Sagar Suites", 0.94),
        ("Aravalli View Lodge", 0.83),
    ),
    "Bengaluru": (
        ("Whitefield Business Tower", 1.05),
        ("MG Road Executive", 1.18),
        ("Koramangala Stay", 0.87),
        ("Electronic City Inn", 0.79),
    ),
    "Mumbai": (
        ("Marine Drive Plaza", 1.28),
        ("Bandra Kurla Suites", 1.16),
        ("Andheri Airport Hotel", 0.91),
        ("Colaba Causeway Rooms", 0.98),
    ),
    "New Delhi": (
        ("Connaught Circle Hotel", 1.15),
        ("Aerocity Grand", 1.09),
        ("Karol Bagh Comfort", 0.82),
        ("Saket Metro Residency", 0.88),
    ),
}

_DEFAULT_PROPERTIES: Tuple[Tuple[str, float], ...] = (
    ("City Central Hotel", 1.06),
    ("Station Road Residency", 0.88),
    ("Grand Metropolitan", 1.24),
)

#: This OTA's own view of when a city is busy. Deliberately NOT our calendar:
#: different windows, different weights. ``(city, start (m, d), end (m, d), lift)``.
_PEAK_WINDOWS: Tuple[Tuple[str, Tuple[int, int], Tuple[int, int], float], ...] = (
    ("Goa", (12, 20), (1, 5), 0.62),
    ("Goa", (2, 10), (2, 20), 0.24),
    ("Jaipur", (1, 18), (1, 30), 0.38),
    ("Jaipur", (11, 10), (12, 20), 0.30),
    ("Udaipur", (10, 25), (2, 25), 0.34),
    ("Bengaluru", (11, 14), (11, 24), 0.29),
    ("Mumbai", (12, 24), (1, 3), 0.33),
    ("Mumbai", (3, 8), (3, 18), 0.19),
    ("New Delhi", (11, 10), (11, 30), 0.31),
    ("New Delhi", (1, 20), (1, 28), 0.22),
)

#: Rough monsoon slump by city, applied June to September.
_MONSOON_DISCOUNT: Dict[str, float] = {
    "Goa": -0.31,
    "Mumbai": -0.22,
    "Udaipur": -0.14,
    "Jaipur": -0.11,
    "Bengaluru": -0.04,
    "New Delhi": -0.08,
}


def _covers(day: date, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
    """Whether ``day`` falls in a ``(month, day)`` window that may wrap the year."""
    md = (day.month, day.day)
    if start <= end:
        return start <= md <= end
    return md >= start or md <= end


def _jitter(*parts: object) -> float:
    """Deterministic pseudo-noise in ``[-0.5, 0.5)`` from the given key parts.

    A hash rather than ``random``: the same query must return the same rate on
    every call and in every process, or the scraper's output would change between
    a dashboard preview and the background producer run that follows it.
    """
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:4], "big") / 0xFFFFFFFF - 0.5


def _peak_lift(city: str, stay_date: date) -> float:
    """Combined peak-window lift for a city on a date."""
    return sum(
        lift
        for window_city, start, end, lift in _PEAK_WINDOWS
        if window_city == city and _covers(stay_date, start, end)
    )


def _seasonal(city: str, stay_date: date) -> float:
    """Season adjustment: monsoon slump, mild winter lift."""
    if 6 <= stay_date.month <= 9:
        return _MONSOON_DISCOUNT.get(city, -0.10)
    if stay_date.month in (12, 1, 2):
        return 0.12
    return 0.0


def _lead_time_curve(lead_days: int) -> float:
    """Advance-purchase shape: cheap far out, firm inside a week, spiking last-minute.

    Real OTA rates are not monotonic in lead time. Distressed inventory at 30+
    days is discounted, the rate firms as the date approaches, and the last two
    days go either way depending on how the property sold. The bump at the very
    short end is what makes ``days_to_checkin`` interesting to the model rather
    than a proxy for "cheaper".
    """
    if lead_days <= 1:
        return 0.18
    if lead_days <= 3:
        return 0.09
    if lead_days <= 7:
        return 0.02
    if lead_days <= 21:
        return -0.05
    if lead_days <= 60:
        return -0.12
    return -0.17


@dataclass(frozen=True)
class RateQuote:
    """One property's rate for one room category on one night.

    ``price`` is ``None`` when the property has no availability. The scraper must
    cope with that: a sold-out card is normal, and treating it as a parse failure
    would make every busy weekend look like a broken parser.
    """

    property_name: str
    room_type: str
    price: Optional[float]
    currency: str = "INR"

    @property
    def is_available(self) -> bool:
        return self.price is not None


def quote_rates(
    city: str,
    stay_date: date,
    room_type: str,
    *,
    today: Optional[date] = None,
) -> List[RateQuote]:
    """Every listed property's rate for one city, night and room category.

    Args:
        city: City name, e.g. ``"Goa"``. Unknown cities return a small default
            set rather than an error.
        stay_date: The night being priced.
        room_type: One of :data:`ROOM_TYPES`.
        today: Reference date for the lead-time curve. Injectable so tests are
            not clock-dependent.

    Returns:
        One quote per listed property, in listing order. Some may be sold out.

    Raises:
        ValueError: If ``room_type`` is not a known category.
    """
    if room_type not in _ROOM_MULTIPLIER:
        raise ValueError(
            f"unknown room type {room_type!r}; expected one of {list(ROOM_TYPES)}"
        )

    today = today or date.today()
    base = _CITY_BASE.get(city, _DEFAULT_BASE)
    properties = _PROPERTIES.get(city, _DEFAULT_PROPERTIES)

    lead_days = (stay_date - today).days
    weekend = 0.16 if stay_date.weekday() >= 4 else 0.0  # Fri/Sat/Sun carry a premium

    market = (
        1.0
        + weekend
        + _peak_lift(city, stay_date)
        + _seasonal(city, stay_date)
        + _lead_time_curve(lead_days)
    )

    quotes: List[RateQuote] = []
    for property_name, position in properties:
        key = (city, property_name, room_type, stay_date.isoformat())

        # Sold out roughly one time in nine, and more often when the market is
        # hot -- which is what makes competitor coverage vary by night rather
        # than being a constant the monitoring checks can never move.
        sellout_pressure = 0.09 + max(0.0, market - 1.2) * 0.22
        if _jitter("avail", *key) + 0.5 < sellout_pressure:
            quotes.append(RateQuote(property_name, room_type, None))
            continue

        # +/- 7% per observation. Without this the rate is a pure function of the
        # inputs and any model would learn it exactly; see the module docstring.
        noise = 1.0 + _jitter("price", *key) * 0.14

        price = base * _ROOM_MULTIPLIER[room_type] * market * position * noise
        quotes.append(RateQuote(property_name, room_type, round(price, 2)))

    return quotes


def known_cities() -> Tuple[str, ...]:
    """Cities this OTA has inventory for."""
    return tuple(_CITY_BASE)


__all__ = ["ROOM_TYPES", "RateQuote", "known_cities", "quote_rates"]
