"""Scrapes the bundled demo OTA. The default working scraping path.

This is a real scraper. It reads ``robots.txt`` and obeys it, fetches HTML over
HTTP, parses it with CSS selectors, respects the rate limiter, retries transient
failures, and raises :class:`~ingestion.scraper_base.ScraperParseError` when the
markup stops matching. Every line of the scraping stack is exercised.

The only thing that is not third-party is the server. Booking.com and Expedia
both disallow the search paths their scrapers would need, so those scrapers --
correctly -- refuse to fetch anything. Rather than delete the robots check to
make a demo work, the project ships a site that genuinely grants permission. See
:mod:`demo_ota` for why that is the honest trade.

MAPPING PROPERTIES ONTO COMPETITOR BRANDS
-----------------------------------------
The demo site lists several properties per city; ``competitor_prices.competitor``
is one of four brands we track. Listing order maps onto those brands
deterministically, so a given property is always the same competitor. That keeps
``competitor_min_rate`` and ``competitor_max_rate`` meaningful -- they need a
spread across distinct sources to carry any information -- without adding an
enum value, which on a ``VARCHAR + CHECK`` column would mean a migration.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from database.models import Competitor, RoomType
from ingestion.scraper_base import (
    HttpCompetitorScraper,
    ScrapeRequest,
    ScraperParseError,
)
from monitoring.logging_config import get_logger
from streaming.events import CompetitorPricePayload

logger = get_logger(__name__)

#: Candidate selectors for a property card, most specific first. More than one
#: because that is how a scraper survives a *minor* markup change; a major one
#: is meant to fail loudly.
_CARD_SELECTORS: Tuple[str, ...] = (
    "div[data-testid='property-card']",
    "div.property-card",
)

#: Candidate selectors for the price element inside a card.
_PRICE_SELECTORS: Tuple[str, ...] = (
    "span[data-testid='price-and-discounted-price']",
    "span.rate",
)

#: Candidate selectors for the property name.
_NAME_SELECTORS: Tuple[str, ...] = (
    "h3[data-testid='title']",
    "h3.property-name",
)

#: Candidate selectors for the unit description.
_UNIT_SELECTORS: Tuple[str, ...] = (
    "p[data-testid='recommended-units']",
    "p.unit-name",
)

#: Matches "₹ 6,200", "₹&nbsp;6200", "INR 6200", "Rs. 6,200".
_PRICE_PATTERN = re.compile(r"(?:₹|INR|Rs\.?)\s*([\d][\d,.\s ]*)", re.IGNORECASE)

#: Keyword -> category, longest-specificity first. The site publishes prose unit
#: names ("Premium Executive Room, Club Access"), which is what a real listing
#: looks like and what the classifier has to recover a category from.
_ROOM_KEYWORDS: Tuple[Tuple[Tuple[str, ...], RoomType], ...] = (
    (("suite", "penthouse", "villa"), RoomType.SUITE),
    (("premium", "executive", "club"), RoomType.PREMIUM),
    (("deluxe", "luxury", "king"), RoomType.DELUXE),
    (("standard", "classic", "economy", "double", "twin"), RoomType.STANDARD),
)

#: Listing position -> competitor brand. See the module docstring.
_BRANDS: Tuple[Competitor, ...] = (
    Competitor.BOOKING,
    Competitor.EXPEDIA,
    Competitor.AGODA,
    Competitor.MAKEMYTRIP,
)


def parse_price(text: str) -> Optional[float]:
    """Extract a numeric rate from a price string, or ``None``.

    Returns ``None`` rather than raising: a card with no price is a sold-out
    property, which is normal and frequent. Only a page with *no parseable
    cards at all* is a parse failure.
    """
    match = _PRICE_PATTERN.search(text or "")
    if not match:
        return None
    cleaned = re.sub(r"[,\s ]", "", match.group(1)).rstrip(".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None


def classify_room(text: str) -> Optional[RoomType]:
    """Map a free-text unit name to one of our four categories, or ``None``."""
    lowered = (text or "").lower()
    for keywords, room_type in _ROOM_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return room_type
    return None


def _first_match(node: Any, selectors: Iterable[str]) -> Optional[Any]:
    for selector in selectors:
        found = node.select_one(selector)
        if found is not None:
            return found
    return None


def _first_match_all(soup: Any, selectors: Iterable[str]) -> List[Any]:
    for selector in selectors:
        found = soup.select(selector)
        if found:
            return found
    return []


class DemoOTAScraper(HttpCompetitorScraper):
    """Collects rates from the bundled demo OTA over HTTP."""

    name = "demo_ota"
    requires_network = True

    def __init__(self, settings: Optional[Any] = None) -> None:
        super().__init__(settings)
        # Unlike the third-party scrapers, whose host is a fixed brand address,
        # this one is wherever the operator is running it: a Compose service
        # name, a localhost port, or a container in CI.
        self.base_url = self.settings.ingestion.demo_ota_base_url.rstrip("/")

    def build_url(self, request: ScrapeRequest) -> Tuple[str, Dict[str, str]]:
        """Search URL and query parameters for one city, night and category."""
        params = {
            "city": request.city,
            "checkin": request.check_in_date.isoformat(),
            "room": request.room_type.value,
            # Normally "v1". Set INGESTION_DEMO_OTA_LAYOUT=v2 to make the site
            # serve redesigned markup and watch this scraper fail honestly.
            "layout": self.settings.ingestion.demo_ota_layout,
        }
        return f"{self.base_url}/search", params

    def parse(self, html: str, request: ScrapeRequest) -> List[CompetitorPricePayload]:
        """Extract rates from a results page.

        Raises:
            ScraperParseError: When no property cards match any known selector.
                That means the markup changed -- serve ``?layout=v2`` to see it
                happen -- and it must not be reported as an empty market. To the
                pricing engine, "our competitors published no rates" and "our
                parser broke" would otherwise be indistinguishable, and the
                second one would quietly widen every competitor band.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        cards = _first_match_all(soup, _CARD_SELECTORS)

        if not cards:
            raise ScraperParseError(
                f"no property cards matched {list(_CARD_SELECTORS)} on the demo "
                f"OTA results page for {request.city} {request.check_in_date}. "
                f"The markup has changed."
            )

        payloads: List[CompetitorPricePayload] = []
        sold_out = 0

        for position, card in enumerate(cards):
            price_node = _first_match(card, _PRICE_SELECTORS)
            if price_node is None:
                sold_out += 1
                continue

            price = parse_price(price_node.get_text(" ", strip=True))
            if price is None:
                sold_out += 1
                continue

            unit_node = _first_match(card, _UNIT_SELECTORS)
            room_type = classify_room(
                unit_node.get_text(" ", strip=True) if unit_node else ""
            )
            # An unclassifiable or mismatched unit is skipped, never guessed at:
            # a suite recorded as a standard room poisons the competitor average
            # for both categories.
            if room_type is not request.room_type:
                continue

            payloads.append(
                CompetitorPricePayload(
                    hotel_id=request.hotel_id,
                    competitor=_BRANDS[position % len(_BRANDS)],
                    room_type=request.room_type,
                    check_in_date=request.check_in_date,
                    price=round(price, 2),
                    currency=self.settings.pricing.currency,
                    is_available=True,
                    source=self.name,
                )
            )

        logger.info(
            "demo OTA: %d card(s), %d rate(s), %d sold out for %s %s %s",
            len(cards),
            len(payloads),
            sold_out,
            request.city,
            request.room_type.value,
            request.check_in_date,
        )
        return payloads


__all__ = ["DemoOTAScraper", "classify_room", "parse_price"]
