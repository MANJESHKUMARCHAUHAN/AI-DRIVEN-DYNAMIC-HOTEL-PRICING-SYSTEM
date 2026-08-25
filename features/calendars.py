"""Calendar facts: seasons, Indian holidays, and city event windows.

This module answers questions about *dates*, never about demand. "Is 2026-11-08
a holiday, and which one?" belongs here; "how much does Diwali lift occupancy in
Udaipur?" belongs to the demand model that consumes these facts.

That split matters for two reasons. The synthetic data generator and the Phase 4
feature pipeline must agree exactly on what ``season``, ``holiday_flag`` and
``local_event_score`` mean, or every model trained on generated data is being
served differently-defined features. And these are the only features in the
system that are knowable arbitrarily far into the future, which is what makes a
30-day price forecast possible at all.

Named ``calendars`` rather than ``calendar`` on purpose: a module named
``calendar.py`` shadows the standard library module of that name for any process
that puts this directory on ``sys.path``.

Holiday dates for lunar festivals (Diwali, Holi, Eid) are the commonly published
observance dates. They can differ by a day between regional calendars; for
demand modelling that is immaterial, and swapping in the ``holidays`` package is
a change to :func:`_load_holidays` alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Tuple

from database.models import Season

# --------------------------------------------------------------------------- #
# Seasons
# --------------------------------------------------------------------------- #

#: Month -> season, on the Indian meteorological calendar rather than the
#: temperate four-season one. Monsoon is a real and large demand trough here,
#: and a model trained on "summer = June" would learn the wrong sign.
_MONTH_TO_SEASON: Dict[int, Season] = {
    12: Season.WINTER, 1: Season.WINTER, 2: Season.WINTER,
    3: Season.SUMMER, 4: Season.SUMMER, 5: Season.SUMMER,
    6: Season.MONSOON, 7: Season.MONSOON, 8: Season.MONSOON, 9: Season.MONSOON,
    10: Season.AUTUMN, 11: Season.AUTUMN,
}


def season_of(day: date) -> Season:
    """Return the season a date falls in."""
    return _MONTH_TO_SEASON[day.month]


def is_weekend(day: date) -> bool:
    """Saturday or Sunday.

    Kept as the plain calendar definition. Hotels whose weekend peak is Friday
    night express that through their segment's day-of-week profile, not by
    redefining what "weekend" means -- a feature whose meaning varies per row is
    a feature a model cannot learn.
    """
    return day.weekday() >= 5


# --------------------------------------------------------------------------- #
# Holidays
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Holiday:
    """A public holiday or major festival.

    Attributes:
        name: Display name.
        significance: How strongly the date pulls travel demand, 0-1. Diwali and
            New Year move the whole country; Mahavir Jayanti moves almost nobody.
        leisure_bias: Where the demand goes. ``1.0`` is purely leisure (people
            travel to resorts), ``0.0`` purely business (offices shut, corporate
            travel stops and business hotels *lose* demand).
    """

    name: str
    significance: float
    leisure_bias: float


#: ``(month, day) -> Holiday`` for holidays that fall on the same Gregorian date
#: every year.
_FIXED_HOLIDAYS: Dict[Tuple[int, int], Holiday] = {
    (1, 1): Holiday("New Year's Day", 0.95, 0.9),
    (1, 14): Holiday("Makar Sankranti / Pongal", 0.55, 0.7),
    (1, 26): Holiday("Republic Day", 0.70, 0.6),
    (4, 14): Holiday("Ambedkar Jayanti", 0.35, 0.5),
    (5, 1): Holiday("Labour Day", 0.30, 0.6),
    (8, 15): Holiday("Independence Day", 0.75, 0.7),
    (10, 2): Holiday("Gandhi Jayanti", 0.60, 0.6),
    (12, 25): Holiday("Christmas", 0.90, 0.95),
    (12, 31): Holiday("New Year's Eve", 1.00, 1.0),
}

#: Lunar and computed festivals, which move every year. Covers the range the
#: synthetic dataset and the forecast horizon can reach.
_MOVABLE_HOLIDAYS: Dict[date, Holiday] = {
    # --- 2024 ---------------------------------------------------------------
    date(2024, 3, 25): Holiday("Holi", 0.80, 0.75),
    date(2024, 4, 11): Holiday("Eid al-Fitr", 0.65, 0.6),
    date(2024, 6, 17): Holiday("Bakrid", 0.55, 0.6),
    date(2024, 9, 7): Holiday("Ganesh Chaturthi", 0.70, 0.65),
    date(2024, 10, 12): Holiday("Dussehra", 0.75, 0.7),
    date(2024, 11, 1): Holiday("Diwali", 1.00, 0.8),
    # --- 2025 ---------------------------------------------------------------
    date(2025, 2, 26): Holiday("Maha Shivaratri", 0.45, 0.6),
    date(2025, 3, 14): Holiday("Holi", 0.80, 0.75),
    date(2025, 3, 31): Holiday("Eid al-Fitr", 0.65, 0.6),
    date(2025, 4, 6): Holiday("Ram Navami", 0.40, 0.6),
    date(2025, 4, 18): Holiday("Good Friday", 0.40, 0.7),
    date(2025, 6, 7): Holiday("Bakrid", 0.55, 0.6),
    date(2025, 8, 16): Holiday("Janmashtami", 0.50, 0.6),
    date(2025, 8, 27): Holiday("Ganesh Chaturthi", 0.70, 0.65),
    date(2025, 10, 2): Holiday("Dussehra", 0.80, 0.7),
    date(2025, 10, 20): Holiday("Diwali", 1.00, 0.8),
    date(2025, 11, 5): Holiday("Guru Nanak Jayanti", 0.45, 0.6),
    # --- 2026 ---------------------------------------------------------------
    date(2026, 2, 15): Holiday("Maha Shivaratri", 0.45, 0.6),
    date(2026, 3, 4): Holiday("Holi", 0.80, 0.75),
    date(2026, 3, 21): Holiday("Eid al-Fitr", 0.65, 0.6),
    date(2026, 3, 26): Holiday("Ram Navami", 0.40, 0.6),
    date(2026, 4, 3): Holiday("Good Friday", 0.40, 0.7),
    date(2026, 5, 1): Holiday("Buddha Purnima", 0.35, 0.6),
    date(2026, 5, 27): Holiday("Bakrid", 0.55, 0.6),
    date(2026, 9, 4): Holiday("Janmashtami", 0.50, 0.6),
    date(2026, 9, 14): Holiday("Ganesh Chaturthi", 0.70, 0.65),
    date(2026, 10, 20): Holiday("Dussehra", 0.80, 0.7),
    date(2026, 11, 8): Holiday("Diwali", 1.00, 0.8),
    date(2026, 11, 24): Holiday("Guru Nanak Jayanti", 0.45, 0.6),
    # --- 2027 ---------------------------------------------------------------
    date(2027, 3, 22): Holiday("Holi", 0.80, 0.75),
    date(2027, 3, 11): Holiday("Eid al-Fitr", 0.65, 0.6),
    date(2027, 9, 3): Holiday("Ganesh Chaturthi", 0.70, 0.65),
    date(2027, 10, 9): Holiday("Dussehra", 0.80, 0.7),
    date(2027, 10, 29): Holiday("Diwali", 1.00, 0.8),
}


@lru_cache(maxsize=8)
def _load_holidays(year: int) -> Dict[date, Holiday]:
    """All holidays in a year, fixed and movable, as a date-keyed map.

    Collisions are real: 1 May 2026 is both Labour Day and Buddha Purnima. The
    map holds one holiday per date, so the **more significant** one wins rather
    than whichever happened to be inserted last. Insertion order is not a
    modelling decision and should not silently become one.
    """
    result: Dict[date, Holiday] = {
        date(year, month, day): holiday
        for (month, day), holiday in _FIXED_HOLIDAYS.items()
    }
    for day, holiday in _MOVABLE_HOLIDAYS.items():
        if day.year != year:
            continue
        incumbent = result.get(day)
        if incumbent is None or holiday.significance > incumbent.significance:
            result[day] = holiday
    return result


def holiday_on(day: date) -> Optional[Holiday]:
    """The holiday falling on ``day``, or ``None``."""
    return _load_holidays(day.year).get(day)


def is_holiday(day: date) -> bool:
    """Whether ``day`` is a public holiday or major festival."""
    return holiday_on(day) is not None


def holiday_name(day: date) -> Optional[str]:
    """Name of the holiday on ``day``, or ``None``."""
    holiday = holiday_on(day)
    return holiday.name if holiday else None


def holiday_proximity(day: date, window: int = 2) -> float:
    """Holiday pressure on ``day``, including the days around a holiday.

    A hotel does not fill up *on* Diwali and empty out the next morning: the
    long weekend either side carries most of the volume. This returns the
    strongest nearby holiday's significance, decayed linearly by distance, so a
    Friday next to a Monday holiday still scores well above an ordinary Friday.

    Args:
        day: Date to score.
        window: How many days either side of a holiday still feel it.

    Returns:
        A score in ``[0, 1]``.
    """
    best = 0.0
    for offset in range(-window, window + 1):
        neighbour = holiday_on(day + timedelta(days=offset))
        if neighbour is None:
            continue
        decay = 1.0 - (abs(offset) / (window + 1))
        best = max(best, neighbour.significance * decay)
    return round(best, 4)


# --------------------------------------------------------------------------- #
# City events
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CityEvent:
    """A recurring annual event that moves a city's hotel demand.

    Windows are expressed as ``(month, day)`` pairs and may wrap across the year
    end -- Goa's New Year week runs 26 December to 2 January.

    Attributes:
        name: Display name.
        city: City the event affects.
        start: ``(month, day)`` the demand window opens.
        end: ``(month, day)`` the demand window closes, inclusive.
        intensity: Peak demand pressure, 0-1.
        every_other_year: Biennial events (Aero India) run in even years only.
    """

    name: str
    city: str
    start: Tuple[int, int]
    end: Tuple[int, int]
    intensity: float
    every_other_year: bool = False

    def covers(self, day: date) -> bool:
        """Whether ``day`` falls inside this event's window."""
        if self.every_other_year and day.year % 2 != 0:
            return False
        md = (day.month, day.day)
        if self.start <= self.end:
            return self.start <= md <= self.end
        # Window wraps the year boundary.
        return md >= self.start or md <= self.end


#: The event calendar. Real, recurring, publicly known events -- which is the
#: point: an event score is only useful to a pricing engine if the events are
#: knowable in advance.
_CITY_EVENTS: Tuple[CityEvent, ...] = (
    # --- Goa: leisure, extremely peaky --------------------------------------
    CityEvent("New Year Week", "Goa", (12, 26), (1, 2), 1.00),
    CityEvent("Sunburn Festival", "Goa", (12, 27), (12, 30), 0.90),
    CityEvent("Goa Carnival", "Goa", (2, 14), (2, 18), 0.60),
    # --- Jaipur: heritage tourism plus a very large wedding market ----------
    CityEvent("Jaipur Literature Festival", "Jaipur", (1, 22), (1, 26), 0.85),
    CityEvent("Wedding Season (winter)", "Jaipur", (11, 15), (12, 15), 0.65),
    CityEvent("Wedding Season (spring)", "Jaipur", (1, 20), (2, 28), 0.55),
    CityEvent("Teej Festival", "Jaipur", (8, 5), (8, 12), 0.45),
    # --- Udaipur: destination weddings dominate the calendar ----------------
    CityEvent("Destination Wedding Season", "Udaipur", (11, 1), (2, 20), 0.90),
    CityEvent("Mewar Festival", "Udaipur", (3, 20), (3, 24), 0.50),
    # --- Bengaluru: corporate and conference driven -------------------------
    CityEvent("Bengaluru Tech Summit", "Bengaluru", (11, 18), (11, 21), 0.80),
    CityEvent("Aero India", "Bengaluru", (2, 10), (2, 14), 0.70, every_other_year=True),
    CityEvent("IPL Home Fixtures", "Bengaluru", (4, 1), (5, 20), 0.45),
    # --- Mumbai: commercial capital, year-round base plus set pieces --------
    CityEvent("Mumbai Marathon", "Mumbai", (1, 17), (1, 19), 0.50),
    CityEvent("Lakme Fashion Week (spring)", "Mumbai", (3, 11), (3, 15), 0.60),
    CityEvent("Lakme Fashion Week (autumn)", "Mumbai", (10, 8), (10, 12), 0.60),
    CityEvent("IPL Home Fixtures", "Mumbai", (4, 1), (5, 20), 0.45),
    # --- New Delhi: government, trade and exhibitions ------------------------
    CityEvent("India International Trade Fair", "New Delhi", (11, 14), (11, 27), 0.80),
    CityEvent("Auto Expo", "New Delhi", (1, 12), (1, 18), 0.70, every_other_year=True),
    CityEvent("Republic Day Week", "New Delhi", (1, 23), (1, 27), 0.60),
)


def events_on(city: str, day: date) -> List[CityEvent]:
    """Every event active in ``city`` on ``day``, strongest first."""
    active = [e for e in _CITY_EVENTS if e.city == city and e.covers(day)]
    return sorted(active, key=lambda e: e.intensity, reverse=True)


def event_score(city: str, day: date) -> float:
    """Combined event pressure for a city on a date, in ``[0, 1]``.

    Overlapping events combine with diminishing returns rather than adding:
    Sunburn during New Year week is busier than either alone, but a city cannot
    be more than full. The formula is ``1 - Π(1 - intensity)``, the standard
    "probability that at least one thing happened" combination.
    """
    remaining = 1.0
    for event in events_on(city, day):
        remaining *= 1.0 - event.intensity
    return round(1.0 - remaining, 4)


def event_names(city: str, day: date) -> List[str]:
    """Names of the events active in ``city`` on ``day``."""
    return [e.name for e in events_on(city, day)]


def known_event_cities() -> FrozenSet[str]:
    """Cities the event calendar knows about."""
    return frozenset(e.city for e in _CITY_EVENTS)


__all__ = [
    "CityEvent",
    "Holiday",
    "Season",
    "event_names",
    "event_score",
    "events_on",
    "holiday_name",
    "holiday_on",
    "holiday_proximity",
    "is_holiday",
    "is_weekend",
    "known_event_cities",
    "season_of",
]
