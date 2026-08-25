"""The competitor data source interface.

One abstraction, three implementations:

One abstraction, four implementations:

======================================  ====================================
:class:`SyntheticCompetitorGenerator`   The default. Deterministic, offline,
                                        always available. No network.
:class:`~ingestion.demo_ota_scraper.DemoOTAScraper`   A **real** scraper against
                                        the bundled :mod:`demo_ota` site. Real
                                        HTTP, robots.txt, CSS selectors and
                                        parse errors. Enabled by one setting.
:class:`~ingestion.booking_scraper.BookingScraper`    Third-party, opt-in,
                                        disabled.
:class:`~ingestion.expedia_scraper.ExpediaScraper`    Third-party, opt-in,
                                        disabled.
======================================  ====================================

**ADR-004, restated because it matters.** The third-party scrapers exist so the
architecture demonstrably supports an external feed, and they are switched off by
default. Turning one on requires *two* independent settings --
``INGESTION_SOURCE`` naming it and ``INGESTION_ENABLE_REAL_SCRAPERS=true`` --
because a single flag is too easy to flip by accident, and automated access to
Booking.com or Expedia may breach their terms of service. Whether you are
permitted to run them is your decision and your legal exposure, not a default
this project ships in the on position.

In practice both of those sites disallow the search paths their scrapers need,
and :class:`HttpCompetitorScraper` honours ``robots.txt`` -- so enabling them
produces :class:`ScraperBlocked`, which is the correct outcome. That is exactly
why ``demo_ota`` exists: it is the same scraping machinery pointed at a site
that genuinely grants permission, so the pipeline can be demonstrated end to end
without anyone deleting a robots check to make a screenshot work.

Everything downstream -- producer, Kafka, consumer, feature pipeline -- depends
only on this interface. Swapping the synthetic source for a licensed rate feed
is one new subclass and one environment variable.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, ClassVar, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from config import CompetitorSource, Settings, get_settings
from database.models import Competitor, RoomType
from monitoring.logging_config import get_logger
from streaming.events import CompetitorPricePayload

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ScraperError(RuntimeError):
    """Base class for every data-source failure."""


class ScraperDisabled(ScraperError):
    """A live scraper was requested while real scraping is switched off."""


class ScraperBlocked(ScraperError):
    """The remote site refused us: HTTP 403, 429, a CAPTCHA, or robots.txt.

    Distinct from :class:`ScraperParseError` because the response is correct
    behaviour by the site and must stop the run, not trigger a retry storm.
    """


class ScraperParseError(ScraperError):
    """The page loaded but does not have the structure we expect.

    Raised rather than returning partial data. A scraper that silently emits
    zero rows when a site redesigns looks exactly like a quiet market, and the
    pricing engine would price into a market it can no longer see.
    """


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScrapeRequest:
    """One rate lookup: a property, a room category and a night.

    Attributes:
        hotel_id: Our hotel whose competitive set is being priced.
        city: Used to build search URLs and to pick the local market.
        room_type: Category to look up.
        check_in_date: The night in question.
        nights: Length of stay to query. One, matching the room-night grain the
            rest of the system uses.
    """

    hotel_id: str
    city: str
    room_type: RoomType
    check_in_date: date
    nights: int = 1

    @property
    def check_out_date(self) -> date:
        return self.check_in_date + timedelta(days=self.nights)

    @property
    def lead_days(self) -> int:
        return (self.check_in_date - date.today()).days


def build_requests(
    hotels: Sequence[tuple],
    *,
    horizons: Sequence[int],
    room_types: Optional[Sequence[RoomType]] = None,
    today: Optional[date] = None,
) -> List[ScrapeRequest]:
    """Expand ``(hotel_id, city)`` pairs into one request per room and horizon.

    Args:
        hotels: ``(hotel_id, city)`` pairs.
        horizons: Days ahead to look up, e.g. ``[1, 7, 30]``.
        room_types: Categories to query. Defaults to all of them.
        today: Reference date, injectable so tests are not clock-dependent.
    """
    today = today or date.today()
    room_types = list(room_types or RoomType)
    return [
        ScrapeRequest(
            hotel_id=hotel_id,
            city=city,
            room_type=room_type,
            check_in_date=today + timedelta(days=horizon),
        )
        for hotel_id, city in hotels
        for horizon in horizons
        for room_type in room_types
    ]


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #


class CompetitorScraper(ABC):
    """A source of competitor rates.

    Subclasses implement :meth:`fetch` for a single request. Rate limiting,
    retries and error accounting are handled here so that every implementation
    behaves the same way under load and under failure.
    """

    #: Human-readable identifier, recorded as ``competitor_prices.source``.
    name: ClassVar[str] = "base"

    #: Whether this implementation reaches out to a third-party website.
    requires_network: ClassVar[bool] = False

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()
        self._last_request_at: float = 0.0
        self.fetch_count = 0
        self.error_count = 0

    # -- contract ---------------------------------------------------------- #

    @abstractmethod
    def fetch(self, request: ScrapeRequest) -> List[CompetitorPricePayload]:
        """Return every competitor rate found for one request.

        Raises:
            ScraperBlocked: The source refused the request.
            ScraperParseError: The response did not have the expected structure.
        """

    def close(self) -> None:
        """Release any held resources. Default implementation does nothing."""

    # -- shared behaviour --------------------------------------------------- #

    def _throttle(self) -> None:
        """Enforce ``INGESTION_RATE_LIMIT_SECONDS`` between outbound requests.

        Politeness for a live site, and a realistic pacing knob for the
        synthetic one. Applied by :meth:`collect`, never inside :meth:`fetch`,
        so a subclass cannot forget it.
        """
        interval = self.settings.ingestion.rate_limit_seconds
        if interval <= 0 or not self.requires_network:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < interval:
            time.sleep(interval - elapsed)
        self._last_request_at = time.monotonic()

    def collect(self, request: ScrapeRequest) -> List[CompetitorPricePayload]:
        """Fetch one request with throttling and retries.

        Transient failures are retried with exponential backoff.
        :class:`ScraperBlocked` is *not* retried -- hammering a site that just
        said no is how a soft block becomes a permanent one.

        Returns:
            The rates found, or an empty list once retries are exhausted.
            Missing competitor data is a normal condition the feature pipeline
            already handles; it must not take the collector down.
        """
        retries = self.settings.ingestion.max_retries
        delay = 1.0

        for attempt in range(1, retries + 2):
            self._throttle()
            try:
                payloads = self.fetch(request)
                self.fetch_count += 1
                return payloads
            except ScraperBlocked:
                self.error_count += 1
                raise
            except ScraperError as exc:
                self.error_count += 1
                if attempt > retries:
                    logger.warning(
                        "Giving up on %s %s %s after %d attempt(s): %s",
                        request.hotel_id,
                        request.room_type.value,
                        request.check_in_date,
                        attempt,
                        type(exc).__name__,
                    )
                    return []
                logger.warning(
                    "Fetch failed (%s), attempt %d/%d; retrying in %.1fs",
                    type(exc).__name__,
                    attempt,
                    retries + 1,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
        return []

    def collect_many(
        self, requests: Iterable[ScrapeRequest]
    ) -> Iterator[CompetitorPricePayload]:
        """Stream rates for many requests, yielding as they arrive.

        A generator rather than a list: the producer publishes each event as it
        is produced, so a long collection run does not sit in memory and a
        crash halfway through has still delivered half the data.
        """
        for request in requests:
            try:
                yield from self.collect(request)
            except ScraperBlocked as exc:
                logger.error("Source blocked us; stopping collection: %s", exc)
                return

    def __enter__(self) -> "CompetitorScraper":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# --------------------------------------------------------------------------- #
# HTTP base
# --------------------------------------------------------------------------- #


class HttpCompetitorScraper(CompetitorScraper):
    """Shared machinery for scrapers that fetch a real web page.

    Subclasses supply :meth:`build_url` and :meth:`parse`; everything that is
    easy to get wrong -- robots.txt, timeouts, connection reuse, status-code
    mapping, honest error types -- is handled once, here.

    ``robots.txt`` is fetched and honoured before the first request. A scraper
    that ignores robots is not a scraper, it is an incident.
    """

    requires_network: ClassVar[bool] = True

    #: Scheme and host, e.g. ``https://www.example.com``.
    base_url: ClassVar[str] = ""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        super().__init__(settings)
        self._client: Optional[Any] = None
        self._robots: Optional[Any] = None

    # -- contract for subclasses ------------------------------------------- #

    def build_url(self, request: ScrapeRequest) -> Tuple[str, Dict[str, str]]:
        """Return ``(url, query parameters)`` for one lookup."""
        raise NotImplementedError

    def parse(self, html: str, request: ScrapeRequest) -> List[CompetitorPricePayload]:
        """Extract rates from a fetched page.

        Raises:
            ScraperParseError: If the page does not contain the expected
                structure. Never return an empty list to paper over a redesign:
                "no rates found" and "the site changed" must not look the same
                to the pricing engine.
        """
        raise NotImplementedError

    # -- HTTP -------------------------------------------------------------- #

    @property
    def client(self) -> Any:
        """Lazily created ``httpx.Client`` with connection reuse."""
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                timeout=self.settings.ingestion.request_timeout_seconds,
                follow_redirects=True,
                headers={
                    "User-Agent": self.settings.ingestion.user_agent,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "en-IN,en;q=0.9",
                },
            )
        return self._client

    def _check_robots(self, url: str) -> None:
        """Refuse to fetch a URL that ``robots.txt`` disallows.

        Raises:
            ScraperBlocked: If the site's robots policy forbids this path.
        """
        from urllib.robotparser import RobotFileParser

        if self._robots is None:
            parser = RobotFileParser()
            parser.set_url(f"{self.base_url}/robots.txt")
            try:
                parser.read()
            except Exception as exc:
                # An unreadable robots.txt is treated as "disallow". The
                # permissive alternative is how projects end up sending traffic
                # they were never allowed to send.
                raise ScraperBlocked(
                    f"could not read {self.base_url}/robots.txt ({type(exc).__name__}); "
                    "refusing to fetch"
                ) from exc
            self._robots = parser

        agent = self.settings.ingestion.user_agent
        if not self._robots.can_fetch(agent, url):
            raise ScraperBlocked(f"robots.txt disallows {url} for {agent!r}")

    def get_html(self, request: ScrapeRequest) -> str:
        """Fetch the page for a request, mapping HTTP failures to scraper errors."""
        import httpx

        url, params = self.build_url(request)
        self._check_robots(url)

        try:
            response = self.client.get(url, params=params)
        except httpx.TimeoutException as exc:
            raise ScraperError(f"timeout fetching {url}") from exc
        except httpx.HTTPError as exc:
            raise ScraperError(f"transport error fetching {url}: {exc}") from exc

        if response.status_code in (401, 403, 429):
            raise ScraperBlocked(
                f"{self.name} returned HTTP {response.status_code} for {url}; "
                "back off and stop"
            )
        if response.status_code >= 500:
            raise ScraperError(f"{self.name} returned HTTP {response.status_code}")
        if response.status_code != 200:
            raise ScraperParseError(
                f"{self.name} returned unexpected HTTP {response.status_code}"
            )
        return response.text

    def fetch(self, request: ScrapeRequest) -> List[CompetitorPricePayload]:
        """Fetch and parse one request."""
        return self.parse(self.get_html(request), request)

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #


def get_scraper(settings: Optional[Settings] = None) -> CompetitorScraper:
    """Build the configured data source.

    Raises:
        ScraperDisabled: If a live scraper is named but real scraping has not
            been explicitly enabled. In practice
            :class:`~config.settings.IngestionSettings` rejects that combination
            first; this is the second lock on the same door, because a caller
            can construct settings programmatically.
    """
    settings = settings or get_settings()
    source = settings.ingestion.source

    if source is CompetitorSource.SYNTHETIC:
        from ingestion.synthetic_generator import SyntheticCompetitorGenerator

        return SyntheticCompetitorGenerator(settings)

    if source is CompetitorSource.DEMO_OTA:
        # Real HTTP, real robots.txt, real selectors -- against a site we ship.
        # Not gated on enable_real_scrapers: that flag is about third-party
        # terms of service, not about whether the network is involved.
        from ingestion.demo_ota_scraper import DemoOTAScraper

        return DemoOTAScraper(settings)

    if not settings.ingestion.enable_real_scrapers:
        raise ScraperDisabled(
            f"INGESTION_SOURCE={source.value} requires "
            "INGESTION_ENABLE_REAL_SCRAPERS=true. Scraping a third-party site "
            "is disabled by default; see ADR-004 in docs/architecture.md. For a "
            "working scraping pipeline use INGESTION_SOURCE=demo_ota."
        )

    if source is CompetitorSource.BOOKING:
        from ingestion.booking_scraper import BookingScraper

        return BookingScraper(settings)

    if source is CompetitorSource.EXPEDIA:
        from ingestion.expedia_scraper import ExpediaScraper

        return ExpediaScraper(settings)

    raise ScraperError(f"No implementation for competitor source {source!r}")


__all__ = [
    "Competitor",
    "CompetitorScraper",
    "HttpCompetitorScraper",
    "ScrapeRequest",
    "ScraperBlocked",
    "ScraperDisabled",
    "ScraperError",
    "ScraperParseError",
    "build_requests",
    "get_scraper",
]
