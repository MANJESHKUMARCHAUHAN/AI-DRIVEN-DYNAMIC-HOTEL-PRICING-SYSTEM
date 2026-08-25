"""The default competitor data source: a live synthetic rate feed.

Where :mod:`ingestion.synthetic_dataset` builds *history* in one batch, this
builds *now*, one event at a time, forever. It is what the Kafka producer reads
in the demo pipeline, and it is deliberately the default (ADR-004).

The two share one demand model --
:func:`~ingestion.synthetic_dataset.demand_index_for` -- and one set of city,
room and competitor-bias parameters. That is not tidiness for its own sake: if
the streamed rates followed a different market model from the historical ones,
every model trained on history would be served features drawn from a different
distribution, and the drift monitor would fire on data that was correct.

**Reproducibility.** :meth:`SyntheticCompetitorGenerator.fetch` is a pure
function of ``(seed, hotel, room type, stay date, observation date)``. Polling
the same night twice on the same day returns identical rates; polling it again
tomorrow returns rates that have moved, the way a real market does. That makes
the feed both testable and non-trivial.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from config import Settings
from database.models import Competitor, RoomType
from ingestion.scraper_base import CompetitorScraper, ScrapeRequest, ScraperError
from ingestion.synthetic_dataset import (
    CITY_PROFILES,
    COMPETITOR_BIAS,
    COMPETITOR_COVERAGE,
    HOTEL_CATALOG,
    ROOM_MIX,
    HotelProfile,
    demand_index_for,
)
from features.calendars import event_score, season_of
from monitoring.logging_config import get_logger
from streaming.events import CompetitorPricePayload

logger = get_logger(__name__)

#: Days ahead the live feed looks by default. Short, medium and long lead --
#: enough to see the booking curve without flooding the topic.
DEFAULT_HORIZONS: Tuple[int, ...] = (1, 3, 7, 14, 30, 60)

#: Index of the shipped catalogue by hotel id.
_PROFILES: Dict[str, HotelProfile] = {p.hotel_id: p for p in HOTEL_CATALOG}

#: Base price multiplier by room category, from the shared room mix.
_PRICE_MULTIPLIERS: Dict[RoomType, float] = {
    spec.room_type: spec.price_multiplier for spec in ROOM_MIX
}


def default_catalog() -> Dict[Tuple[str, RoomType], float]:
    """``(hotel_id, room_type) -> base price``, derived from the hotel catalogue."""
    return {
        (profile.hotel_id, spec.room_type): round(
            profile.base_price * spec.price_multiplier, 2
        )
        for profile in HOTEL_CATALOG
        for spec in ROOM_MIX
    }


class SyntheticCompetitorGenerator(CompetitorScraper):
    """Produces realistic competitor rate events without touching a network.

    Example::

        generator = SyntheticCompetitorGenerator()
        for payload in generator.stream(max_events=10):
            print(payload.competitor, payload.price)
    """

    name = "synthetic"
    requires_network = False

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        catalog: Optional[Dict[Tuple[str, RoomType], float]] = None,
        seed: Optional[int] = None,
        competitors: Optional[Sequence[Competitor]] = None,
        coverage: float = COMPETITOR_COVERAGE,
    ) -> None:
        """
        Args:
            catalog: Base price per hotel and room type. Defaults to the shipped
                catalogue; ``scripts/run_producer.py`` passes the real one read
                from the ``rooms`` table so the feed prices against actual
                inventory.
            seed: Master seed. Defaults to ``INGESTION_SYNTHETIC_SEED``.
            competitors: Which rate sources to emit. Defaults to all four.
            coverage: Probability a given competitor publishes a rate for a
                given night. Below 1.0 on purpose -- missing competitor data is
                a case the feature pipeline and the monitor must both handle.
        """
        super().__init__(settings)
        self.catalog = catalog or default_catalog()
        self.seed = int(seed if seed is not None else self.settings.ingestion.synthetic_seed)
        self.competitors = list(competitors or Competitor)
        self.coverage = coverage

    # -- CompetitorScraper -------------------------------------------------- #

    def fetch(
        self, request: ScrapeRequest, *, as_of: Optional[date] = None
    ) -> List[CompetitorPricePayload]:
        """Return the competitor rates visible for one night.

        Args:
            request: Which hotel, room type and night to price.
            as_of: Observation date. Defaults to today; injectable so tests do
                not depend on the wall clock.
        """
        as_of = as_of or datetime.now(timezone.utc).date()
        profile = _PROFILES.get(request.hotel_id)
        if profile is None:
            raise ScraperError(
                f"no market profile for hotel {request.hotel_id!r}; the synthetic "
                f"feed only knows {sorted(_PROFILES)}"
            )

        base_price = self.catalog.get(
            (request.hotel_id, request.room_type)
        ) or round(profile.base_price * _PRICE_MULTIPLIERS[request.room_type], 2)

        rng = self._rng(request, as_of)
        lead_days = max((request.check_in_date - as_of).days, 0)

        occupancy = demand_index_for(profile, request.check_in_date)
        market_rate = self._market_rate(
            profile=profile,
            base_price=base_price,
            stay_date=request.check_in_date,
            occupancy=occupancy,
            lead_days=lead_days,
            rng=rng,
        )

        payloads: List[CompetitorPricePayload] = []
        for competitor in self.competitors:
            if rng.random() > self.coverage:
                continue  # This source has no published rate for this night.
            price = market_rate * COMPETITOR_BIAS[competitor] * float(
                rng.normal(1.0, 0.035)
            )
            payloads.append(
                CompetitorPricePayload(
                    hotel_id=request.hotel_id,
                    competitor=competitor,
                    room_type=request.room_type,
                    check_in_date=request.check_in_date,
                    price=round(max(price, 500.0), 2),
                    currency=self.settings.pricing.currency,
                    # A tight market sells out, and a sold-out competitor is a
                    # stronger demand signal than a high price.
                    is_available=bool(rng.random() > occupancy * 0.35),
                    source=self.name,
                )
            )
        return payloads

    # -- internals ---------------------------------------------------------- #

    def _rng(self, request: ScrapeRequest, as_of: date) -> np.random.Generator:
        """A generator keyed on the request *and* the observation date.

        Same night polled twice today: identical rates. Polled again tomorrow:
        the market has moved. Both properties are what a real feed has, and both
        are testable because neither is wall-clock dependent.
        """
        return np.random.default_rng(
            [
                self.seed,
                abs(hash(request.hotel_id)) % 100_000,
                list(RoomType).index(request.room_type),
                request.check_in_date.toordinal(),
                as_of.toordinal(),
            ]
        )

    @staticmethod
    def _market_rate(
        *,
        profile: HotelProfile,
        base_price: float,
        stay_date: date,
        occupancy: float,
        lead_days: int,
        rng: np.random.Generator,
    ) -> float:
        """The competitive set's average rate for a night.

        Identical in structure to the historical generator's competitor model,
        so streamed rates and stored rates come from one distribution.
        """
        city = CITY_PROFILES[profile.city]
        season = season_of(stay_date)

        # The competitors' own read of demand: correlated with ours, not equal.
        market_occupancy = float(np.clip(occupancy + rng.normal(0.0, 0.06), 0.05, 0.99))

        rate = base_price * city.market_price_level
        rate *= city.season_price[season]
        rate *= 1.0 + 0.32 * (market_occupancy - 0.60)
        rate *= 1.0 + 0.26 * event_score(profile.city, stay_date)
        # Rates firm up close to check-in when the market is tight, and get
        # dumped when it is not.
        rate *= 1.0 + 0.09 * (1.0 - min(lead_days, 30) / 30.0) * (
            2.0 * market_occupancy - 1.0
        )
        return rate

    # -- streaming ---------------------------------------------------------- #

    def requests(
        self,
        *,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        room_types: Optional[Sequence[RoomType]] = None,
        hotel_ids: Optional[Sequence[str]] = None,
        as_of: Optional[date] = None,
    ) -> List[ScrapeRequest]:
        """Every lookup one collection pass should perform."""
        as_of = as_of or datetime.now(timezone.utc).date()
        room_types = list(room_types or RoomType)
        hotels = [
            _PROFILES[h] for h in (hotel_ids or list(_PROFILES)) if h in _PROFILES
        ]
        return [
            ScrapeRequest(
                hotel_id=profile.hotel_id,
                city=profile.city,
                room_type=room_type,
                check_in_date=as_of + timedelta(days=horizon),
            )
            for profile in hotels
            for horizon in horizons
            for room_type in room_types
        ]

    def stream(
        self,
        *,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        room_types: Optional[Sequence[RoomType]] = None,
        hotel_ids: Optional[Sequence[str]] = None,
        max_events: Optional[int] = None,
        interval_seconds: Optional[float] = None,
        passes: Optional[int] = None,
    ) -> Iterator[CompetitorPricePayload]:
        """Yield competitor events continuously.

        Args:
            max_events: Stop after this many events. ``None`` is unbounded.
            interval_seconds: Pause between events, so a demo produces a
                readable trickle rather than a wall of text. Defaults to
                ``INGESTION_SYNTHETIC_INTERVAL_SECONDS``.
            passes: Number of full sweeps over the request set. ``None`` loops
                forever.

        Yields:
            Validated :class:`~streaming.events.CompetitorPricePayload` objects.
        """
        interval = (
            interval_seconds
            if interval_seconds is not None
            else self.settings.ingestion.synthetic_interval_seconds
        )
        emitted = 0
        completed = 0

        while passes is None or completed < passes:
            for request in self.requests(
                horizons=horizons, room_types=room_types, hotel_ids=hotel_ids
            ):
                for payload in self.fetch(request):
                    yield payload
                    emitted += 1
                    if max_events is not None and emitted >= max_events:
                        return
                    if interval > 0:
                        time.sleep(interval)
            completed += 1
            logger.info("Completed collection pass %d (%d event(s))", completed, emitted)


__all__ = [
    "DEFAULT_HORIZONS",
    "SyntheticCompetitorGenerator",
    "default_catalog",
]
