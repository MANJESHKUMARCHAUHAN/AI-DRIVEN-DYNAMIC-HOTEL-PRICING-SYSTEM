"""Generation of the historical hotel dataset the models are trained on.

This is a *simulator*, not a random number dump. Requirement 7 is explicit that
the data must contain real structure, because a model trained on noise learns
nothing and a portfolio project whose feature importances are uniform is worse
than no project at all. Every relationship the pricing engine later claims to
exploit is put here deliberately, and asserted in ``tests/test_synthetic_data.py``:

===========================  ==================================================
Relationship                 How it is produced
===========================  ==================================================
weekend -> demand            Per-segment day-of-week profile. Business hotels
                             peak Tue-Thu and empty at the weekend; leisure
                             hotels do the opposite. Both are real, and a model
                             that learns "weekend = busy" globally is wrong.
season -> demand             Per-city seasonal multipliers on the Indian
                             calendar. Monsoon is a genuine trough.
holiday -> demand            Holiday significance, scaled by how well the
                             holiday's leisure bias matches the hotel segment.
event -> demand spike        City event calendar (:mod:`features.calendars`).
demand -> price              Achieved rate rises with occupancy, and rises
                             *more* for last-minute bookings into a full hotel.
demand -> competitor price   Competitors see the same market, so their rates
                             move with the same demand index plus independent
                             noise -- correlated, not identical.
lead time -> cancellation    Cancellation probability grows with lead time.
demand -> lead time          Peak dates get booked earlier: the lead-time
                             distribution stretches when demand is high.
===========================  ==================================================

Grain: every booking row is a **room-night**. A three-night stay appears as
three rows with ``check_out_date = check_in_date + 1``. This keeps occupancy
arithmetic exact -- rooms sold for a night never has to be inferred from
overlapping intervals -- and the schema still permits genuine multi-night rows
when real data arrives.

Determinism: the generator takes a seed and derives an independent child seed
per hotel, so adding a ninth hotel does not change the data of the first eight.
Two runs with the same seed produce byte-identical CSVs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import Settings, get_settings
from database.models import Competitor, MarketSegment, RoomType, Season
from features.calendars import (
    event_names,
    event_score,
    holiday_on,
    holiday_proximity,
    is_weekend,
    season_of,
)
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

#: Longest lead time the simulator books at. Beyond ~90 days the booking curve
#: is a thin tail that adds rows without adding signal.
MAX_LEAD_DAYS = 90

#: Lead times at which competitor rates are observed, in days before check-in.
#: Two snapshots per night: one while the market is still soft, one in the
#: pricing window where the decision actually gets made.
COMPETITOR_OBSERVATION_LEADS: Tuple[int, ...] = (21, 7)

#: Probability that a given competitor has a published, available rate for a
#: given night. Below 1.0 on purpose -- "competitor data is missing" is a case
#: the feature pipeline and the monitoring layer must both handle.
COMPETITOR_COVERAGE = 0.85


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class RoomSpec:
    """A room category's share of inventory, price and demand behaviour.

    Attributes:
        room_type: The category.
        capacity: Guests the room sleeps.
        inventory_share: Fraction of the hotel's rooms in this category.
        price_multiplier: Base price relative to a standard room.
        demand_multiplier: Relative sell-through. Suites are priced high and
            sell slowly; standard rooms clear first.
    """

    room_type: RoomType
    capacity: int
    inventory_share: float
    price_multiplier: float
    demand_multiplier: float


#: The room mix applied to every hotel. Shares sum to 1.0.
ROOM_MIX: Tuple[RoomSpec, ...] = (
    RoomSpec(RoomType.STANDARD, capacity=2, inventory_share=0.45,
             price_multiplier=1.00, demand_multiplier=1.06),
    RoomSpec(RoomType.DELUXE, capacity=2, inventory_share=0.30,
             price_multiplier=1.28, demand_multiplier=1.00),
    RoomSpec(RoomType.PREMIUM, capacity=3, inventory_share=0.17,
             price_multiplier=1.65, demand_multiplier=0.90),
    RoomSpec(RoomType.SUITE, capacity=4, inventory_share=0.08,
             price_multiplier=2.60, demand_multiplier=0.72),
)


@dataclass(frozen=True)
class CityProfile:
    """How a city behaves across the year.

    Attributes:
        season_demand: Multiplier on occupancy, by season.
        season_price: Multiplier on market rate, by season. Distinct from
            demand: Goa in monsoon is empty *and* cheap, but Bengaluru in
            monsoon is merely normal.
        weather: Pleasantness 0-1, by season. Exposed as the ``weather_score``
            feature.
        market_price_level: Structural rate level of the city's hotel market.
    """

    season_demand: Dict[Season, float]
    season_price: Dict[Season, float]
    weather: Dict[Season, float]
    market_price_level: float


def _seasons(winter: float, summer: float, monsoon: float, autumn: float) -> Dict[Season, float]:
    return {
        Season.WINTER: winter,
        Season.SUMMER: summer,
        Season.MONSOON: monsoon,
        Season.AUTUMN: autumn,
    }


CITY_PROFILES: Dict[str, CityProfile] = {
    "Mumbai": CityProfile(
        season_demand=_seasons(1.10, 1.00, 0.80, 1.06),
        season_price=_seasons(1.08, 1.00, 0.88, 1.05),
        weather=_seasons(0.90, 0.60, 0.35, 0.80),
        market_price_level=1.10,
    ),
    "Bengaluru": CityProfile(
        season_demand=_seasons(1.05, 1.00, 0.94, 1.06),
        season_price=_seasons(1.04, 1.00, 0.95, 1.05),
        weather=_seasons(0.90, 0.75, 0.70, 0.88),
        market_price_level=1.00,
    ),
    "New Delhi": CityProfile(
        season_demand=_seasons(1.14, 0.80, 0.86, 1.16),
        season_price=_seasons(1.12, 0.85, 0.90, 1.14),
        weather=_seasons(0.80, 0.35, 0.55, 0.85),
        market_price_level=1.05,
    ),
    "Goa": CityProfile(
        season_demand=_seasons(1.38, 0.86, 0.54, 1.06),
        season_price=_seasons(1.45, 0.90, 0.62, 1.08),
        weather=_seasons(0.95, 0.60, 0.30, 0.82),
        market_price_level=1.15,
    ),
    "Jaipur": CityProfile(
        season_demand=_seasons(1.26, 0.70, 0.76, 1.20),
        season_price=_seasons(1.30, 0.75, 0.80, 1.22),
        weather=_seasons(0.85, 0.30, 0.60, 0.86),
        market_price_level=0.95,
    ),
    "Udaipur": CityProfile(
        season_demand=_seasons(1.30, 0.70, 0.80, 1.24),
        season_price=_seasons(1.34, 0.74, 0.82, 1.26),
        weather=_seasons(0.90, 0.35, 0.65, 0.90),
        market_price_level=1.05,
    ),
}


#: Day-of-week demand profile per market segment, Monday first.
#: The business and leisure rows are near mirror images -- that opposition is
#: what stops the models from collapsing "is_weekend" into a single global sign.
DOW_PROFILES: Dict[MarketSegment, Tuple[float, ...]] = {
    MarketSegment.BUSINESS: (1.06, 1.16, 1.18, 1.10, 0.84, 0.60, 0.72),
    MarketSegment.LEISURE: (0.78, 0.76, 0.82, 0.92, 1.28, 1.44, 1.08),
    MarketSegment.MIXED: (0.95, 1.00, 1.03, 1.01, 1.09, 1.12, 0.90),
}


@dataclass(frozen=True)
class HotelProfile:
    """A property's structural parameters.

    Attributes:
        base_occupancy: Long-run average occupancy before any calendar effect.
        base_price: Rack rate of a standard room, in INR.
        event_sensitivity: How hard city events hit this property. A 5-star
            near a convention centre fills on a trade fair; a budget inn on the
            ring road barely notices.
        price_position: Rate position against the city market. >1 means the
            hotel habitually sits above the competitive set.
    """

    hotel_id: str
    hotel_name: str
    city: str
    star_rating: int
    total_rooms: int
    segment: MarketSegment
    base_occupancy: float
    base_price: float
    event_sensitivity: float
    price_position: float
    latitude: float
    longitude: float


#: Eight properties across six cities. Two cities carry two hotels each, which
#: is what makes the competitor set genuinely local rather than a global average.
HOTEL_CATALOG: Tuple[HotelProfile, ...] = (
    HotelProfile("H001", "Sanchay Grand Mumbai", "Mumbai", 5, 240,
                 MarketSegment.BUSINESS, 0.72, 6200.0, 0.55, 1.06, 19.0760, 72.8777),
    HotelProfile("H002", "Marine Bay Suites", "Mumbai", 4, 160,
                 MarketSegment.MIXED, 0.68, 4800.0, 0.45, 0.97, 18.9220, 72.8347),
    HotelProfile("H003", "Whitefield Tech Residency", "Bengaluru", 4, 180,
                 MarketSegment.BUSINESS, 0.70, 4200.0, 0.65, 1.00, 12.9698, 77.7500),
    HotelProfile("H004", "Azure Sands Resort", "Goa", 5, 140,
                 MarketSegment.LEISURE, 0.64, 7000.0, 0.80, 1.08, 15.2993, 74.1240),
    HotelProfile("H005", "Amber Fort Palace", "Jaipur", 5, 120,
                 MarketSegment.LEISURE, 0.60, 6500.0, 0.75, 1.10, 26.9124, 75.7873),
    HotelProfile("H006", "Capital Business Tower", "New Delhi", 4, 210,
                 MarketSegment.BUSINESS, 0.71, 5200.0, 0.70, 1.02, 28.6139, 77.2090),
    HotelProfile("H007", "Lake Pichola Heritage", "Udaipur", 5, 90,
                 MarketSegment.LEISURE, 0.58, 8200.0, 0.85, 1.12, 24.5760, 73.6800),
    HotelProfile("H008", "Koramangala Express Inn", "Bengaluru", 3, 130,
                 MarketSegment.MIXED, 0.74, 2900.0, 0.40, 0.92, 12.9352, 77.6245),
)


#: Systematic rate bias per rate source. Agoda discounts, MakeMyTrip loads a
#: convenience premium -- so ``competitor_min_rate`` and ``competitor_max_rate``
#: differ by something structural, not only by noise.
COMPETITOR_BIAS: Dict[Competitor, float] = {
    Competitor.BOOKING: 1.02,
    Competitor.EXPEDIA: 0.99,
    Competitor.AGODA: 0.955,
    Competitor.MAKEMYTRIP: 1.045,
}


# --------------------------------------------------------------------------- #
# The demand model
# --------------------------------------------------------------------------- #
# Module-level rather than a method, because two callers need it: the historical
# generator below, and the live competitor generator in
# ``ingestion.synthetic_generator``. Sharing one function is what keeps the
# streamed events consistent with the history the models were trained on.


def demand_index_for(
    profile: HotelProfile,
    stay_date: date,
    *,
    trend: float = 1.0,
    shock: float = 1.0,
) -> float:
    """Expected occupancy for a hotel on a night, before room-type effects.

    Multiplicative composition, which is how demand actually behaves: a Saturday
    in peak season during a festival is not "weekend plus season plus festival",
    it is the product, and it saturates near capacity.

    Args:
        profile: The property.
        stay_date: The night being priced.
        trend: Slow multiplicative drift, 1.0 for no trend.
        shock: Day-level random shock, 1.0 for the expected value.

    Returns:
        Occupancy in ``[0.05, 0.99]``.
    """
    city = CITY_PROFILES[profile.city]
    season = season_of(stay_date)
    holiday = holiday_on(stay_date)
    holiday_pressure = holiday_proximity(stay_date)

    value = profile.base_occupancy
    value *= city.season_demand[season]
    value *= DOW_PROFILES[profile.segment][stay_date.weekday()]

    # A holiday helps a leisure property and can *hurt* a business one: offices
    # are closed, so corporate travel stops.
    segment_leisure = {
        MarketSegment.BUSINESS: 0.15,
        MarketSegment.MIXED: 0.55,
        MarketSegment.LEISURE: 0.95,
    }[profile.segment]
    # Calibrated so a well-aligned holiday (leisure hotel, leisure festival)
    # lifts demand ~19%, while a badly aligned one (business hotel, national
    # holiday) is mildly negative. The sign flip is the point.
    leisure_bias = holiday.leisure_bias if holiday else 0.5
    alignment = 1.0 - abs(leisure_bias - segment_leisure)
    value *= 1.0 + holiday_pressure * (0.55 * alignment - 0.28)

    value *= 1.0 + event_score(profile.city, stay_date) * profile.event_sensitivity * 0.45
    value *= trend
    value *= shock

    # Saturating transform: demand above capacity turns into denied bookings,
    # not occupancy above 100%.
    return float(np.clip(value, 0.05, 0.99))


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #

_CSV_FILES = {
    "hotels": "hotels.csv",
    "rooms": "rooms.csv",
    "bookings": "bookings.csv",
    "competitor_prices": "competitor_prices.csv",
    "demand_signals": "demand_signals.csv",
}


@dataclass
class SyntheticDataset:
    """The five frames a full generation run produces."""

    hotels: pd.DataFrame
    rooms: pd.DataFrame
    bookings: pd.DataFrame
    competitor_prices: pd.DataFrame
    demand_signals: pd.DataFrame
    metadata: Dict[str, object] = field(default_factory=dict)

    def frames(self) -> Dict[str, pd.DataFrame]:
        """Frame name -> frame, in load order (parents before children)."""
        return {
            "hotels": self.hotels,
            "rooms": self.rooms,
            "bookings": self.bookings,
            "competitor_prices": self.competitor_prices,
            "demand_signals": self.demand_signals,
        }

    def row_counts(self) -> Dict[str, int]:
        return {name: len(frame) for name, frame in self.frames().items()}

    def to_csv(self, directory: Path) -> Dict[str, Path]:
        """Write every frame to ``directory``. Returns the paths written."""
        directory.mkdir(parents=True, exist_ok=True)
        written: Dict[str, Path] = {}
        for name, frame in self.frames().items():
            path = directory / _CSV_FILES[name]
            frame.to_csv(path, index=False)
            written[name] = path
            logger.info("Wrote %-18s %7d rows -> %s", name, len(frame), path.name)
        return written

    @staticmethod
    def _normalise_text_nulls(frame: pd.DataFrame) -> pd.DataFrame:
        """Turn the ``NaN`` pandas reads for empty text cells back into ``None``.

        A missing ``holiday_name`` is a null, not a float. Left as ``NaN`` it
        would reach psycopg2 as the float ``nan`` and be rejected by a VARCHAR
        column, and it would make a CSV round trip non-idempotent.
        """
        text_columns = [c for c in frame.columns if frame[c].dtype == object]
        for column in text_columns:
            frame[column] = frame[column].where(frame[column].notna(), None)
        return frame

    @classmethod
    def from_csv(cls, directory: Path) -> "SyntheticDataset":
        """Read back a previously written dataset, restoring date dtypes."""
        date_columns = {
            "bookings": ["booking_date", "check_in_date", "check_out_date"],
            "competitor_prices": ["check_in_date"],
            "demand_signals": ["stay_date"],
        }
        frames: Dict[str, pd.DataFrame] = {}
        for name, filename in _CSV_FILES.items():
            path = directory / filename
            if not path.is_file():
                raise FileNotFoundError(
                    f"{path} is missing. Run scripts/generate_data.py first."
                )
            frame = pd.read_csv(path)
            for column in date_columns.get(name, []):
                frame[column] = pd.to_datetime(frame[column]).dt.date
            if name == "competitor_prices":
                frame["collected_at"] = pd.to_datetime(frame["collected_at"], utc=True)
            frames[name] = cls._normalise_text_nulls(frame)
        return cls(**frames)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #


class SyntheticDatasetGenerator:
    """Builds a full historical dataset for a set of hotels.

    Example::

        generator = SyntheticDatasetGenerator(seed=42, history_days=365)
        dataset = generator.generate()
        dataset.to_csv(Path("data/synthetic"))
    """

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        n_hotels: Optional[int] = None,
        history_days: Optional[int] = None,
        end_date: Optional[date] = None,
        settings: Optional[Settings] = None,
    ) -> None:
        """
        Args:
            seed: Master seed. Defaults to ``INGESTION_SYNTHETIC_SEED``.
            n_hotels: How many properties from :data:`HOTEL_CATALOG` to include.
            history_days: Length of the generated history, ending at ``end_date``.
            end_date: Last stay date generated. Defaults to today (UTC), so the
                dataset always runs up to "now" and the forecast horizon starts
                where the data stops.
        """
        self.settings = settings or get_settings()
        ingestion = self.settings.ingestion

        self.seed = int(seed if seed is not None else ingestion.synthetic_seed)
        self.history_days = int(
            history_days if history_days is not None else ingestion.synthetic_history_days
        )
        requested = int(n_hotels if n_hotels is not None else ingestion.synthetic_hotels)
        if not 1 <= requested <= len(HOTEL_CATALOG):
            raise ValueError(
                f"n_hotels must be between 1 and {len(HOTEL_CATALOG)}; got {requested}"
            )
        self.profiles: Tuple[HotelProfile, ...] = HOTEL_CATALOG[:requested]

        self.end_date = end_date or datetime.now(timezone.utc).date()
        self.start_date = self.end_date - timedelta(days=self.history_days - 1)

        logger.info(
            "Synthetic dataset: %d hotels, %s..%s (%d days), seed=%d",
            len(self.profiles),
            self.start_date,
            self.end_date,
            self.history_days,
            self.seed,
        )

    # -- public ------------------------------------------------------------ #

    def generate(self) -> SyntheticDataset:
        """Produce the complete dataset."""
        hotels = self._hotel_frame()
        rooms = self._room_frame()

        booking_rows: List[Dict[str, object]] = []
        competitor_rows: List[Dict[str, object]] = []
        signal_rows: List[Dict[str, object]] = []

        for index, profile in enumerate(self.profiles):
            # Child seed per hotel: adding a hotel cannot perturb the others.
            rng = np.random.default_rng([self.seed, index])
            self._generate_hotel(profile, rng, booking_rows, competitor_rows, signal_rows)

        dataset = SyntheticDataset(
            hotels=hotels,
            rooms=rooms,
            bookings=pd.DataFrame(booking_rows),
            competitor_prices=pd.DataFrame(competitor_rows),
            demand_signals=pd.DataFrame(signal_rows),
            metadata={
                "seed": self.seed,
                "n_hotels": len(self.profiles),
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "history_days": self.history_days,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        logger.info("Generated %s", dataset.row_counts())
        return dataset

    # -- reference data ---------------------------------------------------- #

    def _hotel_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "hotel_id": p.hotel_id,
                    "hotel_name": p.hotel_name,
                    "city": p.city,
                    "country": "India",
                    "star_rating": p.star_rating,
                    "total_rooms": p.total_rooms,
                    "segment": p.segment.value,
                    "currency": self.settings.pricing.currency,
                    "latitude": p.latitude,
                    "longitude": p.longitude,
                    "is_active": True,
                }
                for p in self.profiles
            ]
        )

    def _room_frame(self) -> pd.DataFrame:
        rows: List[Dict[str, object]] = []
        for profile in self.profiles:
            for spec, count in self._room_counts(profile).items():
                base = round(profile.base_price * spec.price_multiplier, 2)
                rows.append(
                    {
                        "room_id": f"{profile.hotel_id}-{spec.room_type.value.upper()[:3]}",
                        "hotel_id": profile.hotel_id,
                        "room_type": spec.room_type.value,
                        "capacity": spec.capacity,
                        "room_count": count,
                        "base_price": base,
                        # Per-room guardrails sit inside the global ones and
                        # simply keep a room type from being sold at another
                        # room type's price.
                        "floor_price": round(base * 0.65, 2),
                        "ceiling_price": round(base * 2.20, 2),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _room_counts(profile: HotelProfile) -> Dict[RoomSpec, int]:
        """Split a hotel's inventory across room types, conserving the total.

        The largest category absorbs the rounding remainder so the room counts
        sum exactly to ``total_rooms`` -- occupancy would otherwise be computed
        against a denominator that disagrees with the hotel record.
        """
        counts: Dict[RoomSpec, int] = {}
        allocated = 0
        for spec in ROOM_MIX[1:]:
            n = max(1, int(round(profile.total_rooms * spec.inventory_share)))
            counts[spec] = n
            allocated += n
        counts[ROOM_MIX[0]] = max(1, profile.total_rooms - allocated)
        return {spec: counts[spec] for spec in ROOM_MIX}

    # -- per-hotel simulation ---------------------------------------------- #

    def _generate_hotel(
        self,
        profile: HotelProfile,
        rng: np.random.Generator,
        booking_rows: List[Dict[str, object]],
        competitor_rows: List[Dict[str, object]],
        signal_rows: List[Dict[str, object]],
    ) -> None:
        city = CITY_PROFILES[profile.city]
        room_counts = self._room_counts(profile)

        #: Slow multiplicative trend across the whole window: the property is
        #: gaining (or losing) share. Prophet should find this.
        annual_trend = float(rng.normal(0.06, 0.04))

        for offset in range(self.history_days):
            stay_date = self.start_date + timedelta(days=offset)
            season = season_of(stay_date)
            weekend = is_weekend(stay_date)
            holiday = holiday_on(stay_date)
            holiday_pressure = holiday_proximity(stay_date)
            events = event_score(profile.city, stay_date)

            trend = 1.0 + annual_trend * (offset / max(self.history_days - 1, 1))

            # A day-level shock shared by every room type in the hotel: a
            # conference block, a flight cancellation, a competitor closing.
            day_shock = float(rng.lognormal(mean=0.0, sigma=0.11))

            demand_index = demand_index_for(
                profile, stay_date, trend=trend, shock=day_shock
            )

            weather = self._weather(city, season, rng)
            search = self._search_demand(demand_index, events, rng)

            for spec, room_count in room_counts.items():
                occupancy = float(
                    np.clip(demand_index * spec.demand_multiplier, 0.04, 0.985)
                )
                base_price = round(profile.base_price * spec.price_multiplier, 2)

                self._emit_bookings(
                    profile=profile,
                    spec=spec,
                    room_count=room_count,
                    base_price=base_price,
                    stay_date=stay_date,
                    occupancy=occupancy,
                    city=city,
                    season=season,
                    event=events,
                    holiday_pressure=holiday_pressure,
                    rng=rng,
                    out=booking_rows,
                )

                self._emit_competitor_prices(
                    profile=profile,
                    spec=spec,
                    base_price=base_price,
                    stay_date=stay_date,
                    occupancy=occupancy,
                    city=city,
                    season=season,
                    event=events,
                    rng=rng,
                    out=competitor_rows,
                )

                signal_rows.append(
                    {
                        "hotel_id": profile.hotel_id,
                        "room_type": spec.room_type.value,
                        "stay_date": stay_date,
                        "day_of_week": stay_date.weekday(),
                        "is_weekend": weekend,
                        "season": season.value,
                        "holiday_flag": holiday is not None,
                        "holiday_name": holiday.name if holiday else None,
                        "local_event_score": events,
                        # ``None`` rather than "" so that "no events" survives a
                        # CSV round trip as a null instead of an empty string.
                        "event_names": "; ".join(event_names(profile.city, stay_date))
                        or None,
                        "weather_score": weather,
                        # Search interest is measured at the hotel level and
                        # varies slightly by room type, the way a booking
                        # funnel actually behaves.
                        "search_demand": round(
                            float(
                                np.clip(
                                    search * (0.85 + 0.3 * spec.demand_multiplier), 0.0, 1.0
                                )
                            ),
                            4,
                        ),
                    }
                )

    # -- demand ------------------------------------------------------------ #

    @staticmethod
    def _weather(city: CityProfile, season: Season, rng: np.random.Generator) -> float:
        return round(
            float(np.clip(city.weather[season] + rng.normal(0.0, 0.07), 0.0, 1.0)), 4
        )

    @staticmethod
    def _search_demand(
        demand_index: float, event: float, rng: np.random.Generator
    ) -> float:
        """Search interest: a noisy leading indicator of the same demand.

        Legitimate as a serving-time feature -- searches happen days before the
        stay -- but deliberately *weak*. An earlier calibration correlated 0.83
        with final occupancy, which made every other feature decorative and the
        model's job trivial. Real look-to-book ratios are far noisier than that:
        interest and conversion move for different reasons.

        Three sources of decoupling, not one:

        * a compressive exponent, so the extremes of search interest do not map
          one-for-one onto the extremes of demand;
        * multiplicative noise, which scales with the level the way traffic
          actually varies;
        * additive noise, which dominates at the quiet end.
        """
        compressed = float(np.clip(demand_index, 0.02, 1.0)) ** 0.75
        raw = compressed * (1.0 + 0.30 * event) * float(rng.lognormal(0.0, 0.22))
        raw += rng.normal(0.0, 0.10)
        return round(float(np.clip(raw, 0.0, 1.0)), 4)

    # -- booking curve ------------------------------------------------------ #

    @staticmethod
    def _lead_time_weights(
        segment: MarketSegment, peak_pressure: float
    ) -> np.ndarray:
        """Probability of a booking arriving ``L`` days before check-in.

        Business travel books late and tightly (a spike inside two weeks);
        leisure books on a broad hump around a fortnight out. ``peak_pressure``
        stretches the whole curve earlier, which is the real behaviour on
        high-demand dates: people who know a date will sell out book sooner.
        """
        leads = np.arange(MAX_LEAD_DAYS + 1, dtype=float)

        business = 0.72 * np.exp(-leads / 5.0) + 0.28 * np.exp(-leads / 26.0)
        leisure = np.power(leads + 1.0, 1.6) * np.exp(-(leads + 1.0) / 9.0)

        if segment is MarketSegment.BUSINESS:
            weights = business
        elif segment is MarketSegment.LEISURE:
            weights = leisure
        else:
            weights = 0.5 * business / business.sum() + 0.5 * leisure / leisure.sum()

        # Push mass towards longer lead times as demand pressure rises.
        weights = weights * np.exp(1.6 * peak_pressure * leads / MAX_LEAD_DAYS)
        return weights / weights.sum()

    @staticmethod
    def _cancellation_probability(lead: int) -> float:
        """Cancellation risk grows with lead time: 6% same-day, 21% at 90 days."""
        return 0.06 + 0.15 * (lead / MAX_LEAD_DAYS)

    def _emit_bookings(
        self,
        *,
        profile: HotelProfile,
        spec: RoomSpec,
        room_count: int,
        base_price: float,
        stay_date: date,
        occupancy: float,
        city: CityProfile,
        season: Season,
        event: float,
        holiday_pressure: float,
        rng: np.random.Generator,
        out: List[Dict[str, object]],
    ) -> None:
        """Spread one night's demand across the booking dates that produced it."""
        net_target = int(round(room_count * occupancy))
        if net_target <= 0:
            return

        peak_pressure = float(np.clip(max(event, holiday_pressure, occupancy - 0.5), 0.0, 1.0))
        weights = self._lead_time_weights(profile.segment, peak_pressure)

        # Book gross of expected cancellations so that *net* rooms sold lands on
        # the occupancy target -- which is exactly why hotels overbook.
        leads = np.arange(MAX_LEAD_DAYS + 1)
        expected_cancel_rate = float(
            sum(w * self._cancellation_probability(int(lead)) for lead, w in zip(leads, weights))
        )
        gross = int(math.ceil(net_target / max(1.0 - expected_cancel_rate, 0.5)))

        counts = rng.multinomial(gross, weights)

        achieved_rate = self._achieved_rate(
            base_price=base_price,
            occupancy=occupancy,
            city=city,
            season=season,
            event=event,
            profile=profile,
        )

        for lead, count in enumerate(counts):
            if count == 0:
                continue
            booking_date = stay_date - timedelta(days=int(lead))
            cancellations = int(
                rng.binomial(int(count), self._cancellation_probability(int(lead)))
            )
            sold = int(count) - cancellations

            # Last-minute pricing: into a full hotel it is a premium, into an
            # empty one it is a distress discount. Sign flips at 50% occupancy.
            late_factor = 1.0 - lead / MAX_LEAD_DAYS
            rate = achieved_rate * (
                1.0 + 0.18 * late_factor * (2.0 * occupancy - 1.0)
            )
            rate *= float(rng.normal(1.0, 0.025))
            rate = max(rate, base_price * 0.5)

            out.append(
                {
                    "hotel_id": profile.hotel_id,
                    "room_type": spec.room_type.value,
                    "booking_date": booking_date,
                    "check_in_date": stay_date,
                    # Room-night grain: one row is one night of one room type.
                    "check_out_date": stay_date + timedelta(days=1),
                    "booking_count": int(count),
                    "cancellation_count": cancellations,
                    "revenue": round(sold * rate, 2),
                    "adr": round(rate, 2),
                    "lead_time_days": int(lead),
                    "channel": self._channel(profile.segment, int(lead), rng),
                }
            )

    @staticmethod
    def _achieved_rate(
        *,
        base_price: float,
        occupancy: float,
        city: CityProfile,
        season: Season,
        event: float,
        profile: HotelProfile,
    ) -> float:
        """Average rate the hotel actually achieves for a night.

        Rises with occupancy and with seasonal and event pressure -- the
        "demand -> price" correlation requirement, applied to our own rates.
        """
        rate = base_price * profile.price_position
        rate *= city.season_price[season]
        rate *= 1.0 + 0.38 * (occupancy - 0.60)
        rate *= 1.0 + 0.24 * event * profile.event_sensitivity
        return rate

    @staticmethod
    def _channel(segment: MarketSegment, lead: int, rng: np.random.Generator) -> str:
        """Booking channel. Late business bookings skew corporate/direct."""
        if segment is MarketSegment.BUSINESS:
            options, weights = ("corporate", "direct", "ota"), (0.45, 0.30, 0.25)
        elif segment is MarketSegment.LEISURE:
            options, weights = ("ota", "direct", "travel_agent"), (0.55, 0.30, 0.15)
        else:
            options, weights = ("ota", "direct", "corporate"), (0.45, 0.35, 0.20)
        if lead <= 2:
            # Walk-ins and same-day bookings come through direct channels.
            weights = tuple(
                w * (1.6 if option == "direct" else 1.0)
                for option, w in zip(options, weights)
            )
        total = sum(weights)
        return str(rng.choice(options, p=[w / total for w in weights]))

    # -- competitor rates --------------------------------------------------- #

    def _emit_competitor_prices(
        self,
        *,
        profile: HotelProfile,
        spec: RoomSpec,
        base_price: float,
        stay_date: date,
        occupancy: float,
        city: CityProfile,
        season: Season,
        event: float,
        rng: np.random.Generator,
        out: List[Dict[str, object]],
    ) -> None:
        """Competitor rates observed at two lead times before the stay.

        Competitors are pricing into the same market, so their rates track the
        same demand index -- but through their own noisy read of it, which is
        why ``competitor_rate`` is informative without being a copy of the label.
        """
        for lead in COMPETITOR_OBSERVATION_LEADS:
            observed_on = stay_date - timedelta(days=lead)
            if observed_on < self.start_date:
                continue

            # The competitor set's own read of demand: correlated with ours,
            # not identical.
            market_occupancy = float(
                np.clip(occupancy + rng.normal(0.0, 0.06), 0.05, 0.99)
            )

            market_rate = base_price * city.market_price_level
            market_rate *= city.season_price[season]
            market_rate *= 1.0 + 0.32 * (market_occupancy - 0.60)
            market_rate *= 1.0 + 0.26 * event
            # Rates firm up close to check-in when the market is tight, and get
            # dumped when it is not.
            market_rate *= 1.0 + 0.09 * (1.0 - lead / 30.0) * (2.0 * market_occupancy - 1.0)

            for competitor, bias in COMPETITOR_BIAS.items():
                if rng.random() > COMPETITOR_COVERAGE:
                    continue  # No published rate from this source for this night.
                price = market_rate * bias * float(rng.normal(1.0, 0.035))
                out.append(
                    {
                        "hotel_id": profile.hotel_id,
                        "room_type": spec.room_type.value,
                        "competitor": competitor.value,
                        "check_in_date": stay_date,
                        "price": round(max(price, 500.0), 2),
                        "currency": self.settings.pricing.currency,
                        # A tight market sells out: availability drops away.
                        "is_available": bool(rng.random() > market_occupancy * 0.35),
                        "source": "synthetic",
                        "collected_at": datetime.combine(
                            observed_on, time(9, 0), tzinfo=timezone.utc
                        )
                        + timedelta(minutes=int(rng.integers(0, 240))),
                    }
                )


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #


def generate_dataset(
    *,
    seed: Optional[int] = None,
    n_hotels: Optional[int] = None,
    history_days: Optional[int] = None,
    end_date: Optional[date] = None,
    settings: Optional[Settings] = None,
) -> SyntheticDataset:
    """Build a dataset in one call. See :class:`SyntheticDatasetGenerator`."""
    return SyntheticDatasetGenerator(
        seed=seed,
        n_hotels=n_hotels,
        history_days=history_days,
        end_date=end_date,
        settings=settings,
    ).generate()


def summarise(dataset: SyntheticDataset) -> pd.DataFrame:
    """Per-hotel sanity summary: nights, rooms sold, cancellation rate, ADR.

    Printed by ``scripts/generate_data.py`` so an obviously broken generation
    (zero cancellations, flat ADR, one hotel missing) is visible immediately
    rather than three phases later when a model refuses to learn.
    """
    bookings = dataset.bookings
    net = bookings["booking_count"] - bookings["cancellation_count"]
    frame = bookings.assign(rooms_sold=net)

    grouped = frame.groupby("hotel_id").agg(
        nights=("check_in_date", "nunique"),
        rows=("booking_count", "size"),
        rooms_booked=("booking_count", "sum"),
        rooms_sold=("rooms_sold", "sum"),
        cancellations=("cancellation_count", "sum"),
        revenue=("revenue", "sum"),
        mean_lead_time=("lead_time_days", "mean"),
    )
    grouped["cancellation_rate"] = grouped["cancellations"] / grouped["rooms_booked"]
    grouped["adr"] = grouped["revenue"] / grouped["rooms_sold"]
    return grouped.round(3)


__all__ = [
    "CITY_PROFILES",
    "COMPETITOR_BIAS",
    "COMPETITOR_COVERAGE",
    "COMPETITOR_OBSERVATION_LEADS",
    "DOW_PROFILES",
    "HOTEL_CATALOG",
    "MAX_LEAD_DAYS",
    "ROOM_MIX",
    "CityProfile",
    "HotelProfile",
    "RoomSpec",
    "SyntheticDataset",
    "SyntheticDatasetGenerator",
    "generate_dataset",
    "summarise",
]
