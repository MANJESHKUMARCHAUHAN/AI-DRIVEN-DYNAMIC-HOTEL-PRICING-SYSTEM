"""Booking.com rate collection. Opt-in, disabled by default.

READ THIS BEFORE ENABLING
-------------------------
Booking.com's terms of service restrict automated access, and their
``robots.txt`` disallows most search paths. This module honours robots.txt and
will refuse to fetch a disallowed URL -- which, at the time of writing, is
likely to be *every* URL it would want. That is the correct outcome, not a bug
to route around.

It exists because requirement 3 asks for a modular competitor ingestion
architecture with a ``BookingScraper`` in it, and because the point being
demonstrated is that swapping the data source is a subclass and an environment
variable. The supported production path is a licensed rate feed -- a Booking.com
Connectivity/Demand API partnership, RateGain, OTA Insight, or your channel
manager -- all of which return JSON and would replace :meth:`parse` with about
ten lines.

Enabling requires two settings, deliberately::

    INGESTION_SOURCE=booking
    INGESTION_ENABLE_REAL_SCRAPERS=true

**On the selectors.** Booking.com renders prices client-side and changes its
markup and its ``data-testid`` attributes frequently. The parser below tries
several known-plausible selectors in order and raises
:class:`~ingestion.scraper_base.ScraperParseError` when none match. It does not
return an empty list on a miss: to a pricing engine, "the competitive set has no
published rates" and "our parser broke" must never look the same.
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

#: Candidate CSS selectors for a property card, most specific first. Multiple
#: candidates because the markup is not stable; each is tried in turn.
_CARD_SELECTORS: Tuple[str, ...] = (
    "div[data-testid='property-card']",
    "div[data-testid='property-card-container']",
    "div.sr_property_block",
)

#: Candidate selectors for the price element inside a card.
_PRICE_SELECTORS: Tuple[str, ...] = (
    "span[data-testid='price-and-discounted-price']",
    "div[data-testid='price-and-discounted-price']",
    "span.prco-valign-middle-helper",
    "div.bui-price-display__value",
)

#: Candidate selectors for the room/unit name inside a card.
_ROOM_SELECTORS: Tuple[str, ...] = (
    "span[data-testid='recommended-units'] h4",
    "div[data-testid='recommended-units']",
    "div.room_link span",
)

#: Matches "₹ 6,200", "INR 6200", "6.200" and similar.
_PRICE_PATTERN = re.compile(r"(?:₹|INR|Rs\.?)\s*([\d][\d,.\s]*)", re.IGNORECASE)

#: Keyword -> room category. Booking.com names units freely ("Superior Double
#: Room with Balcony"), so classification is keyword-based and conservative.
_ROOM_KEYWORDS: Tuple[Tuple[Tuple[str, ...], RoomType], ...] = (
    (("suite", "penthouse", "villa"), RoomType.SUITE),
    (("premium", "executive", "club", "superior"), RoomType.PREMIUM),
    (("deluxe", "luxury"), RoomType.DELUXE),
    (("standard", "classic", "economy", "basic", "double", "twin"), RoomType.STANDARD),
)


def parse_price(text: str) -> Optional[float]:
    """Extract a numeric rate from a price string, or ``None``.

    Handles the thousands separators and currency prefixes the site uses.
    Returns ``None`` rather than raising, because a card with no price is a
    normal occurrence -- a sold-out property.
    """
    match = _PRICE_PATTERN.search(text or "")
    if not match:
        return None
    cleaned = re.sub(r"[,\s]", "", match.group(1)).rstrip(".")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None


def classify_room(text: str) -> Optional[RoomType]:
    """Map a free-text unit name to one of our four categories."""
    lowered = (text or "").lower()
    for keywords, room_type in _ROOM_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return room_type
    return None


def _first_match(node: Any, selectors: Iterable[str]) -> Optional[Any]:
    """First element matching any of ``selectors``, or ``None``."""
    for selector in selectors:
        found = node.select_one(selector)
        if found is not None:
            return found
    return None


def _first_match_all(soup: Any, selectors: Iterable[str]) -> List[Any]:
    """All elements for the first selector that matches anything."""
    for selector in selectors:
        found = soup.select(selector)
        if found:
            return found
    return []


class BookingScraper(HttpCompetitorScraper):
    """Collects published rates from Booking.com search results."""

    name = "booking"
    competitor = Competitor.BOOKING
    base_url = "https://www.booking.com"

    def build_url(self, request: ScrapeRequest) -> Tuple[str, Dict[str, str]]:
        """Search URL and query parameters for one city and one night."""
        params = {
            "ss": request.city,
            "checkin": request.check_in_date.isoformat(),
            "checkout": request.check_out_date.isoformat(),
            "group_adults": "2",
            "no_rooms": "1",
            "group_children": "0",
            "selected_currency": self.settings.pricing.currency,
        }
        return f"{self.base_url}/searchresults.html", params

    def parse(self, html: str, request: ScrapeRequest) -> List[CompetitorPricePayload]:
        """Extract rates for the requested room category from a results page.

        Raises:
            ScraperParseError: When no property cards are found at all, which
                means the markup changed or the response was a bot-check page.
        """
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        cards = _first_match_all(soup, _CARD_SELECTORS)

        if not cards:
            raise ScraperParseError(
                f"no property cards matched {list(_CARD_SELECTORS)} on the "
                f"Booking.com results page for {request.city} "
                f"{request.check_in_date}. The markup has changed, or the "
                f"response was an interstitial rather than results."
            )

        payloads: List[CompetitorPricePayload] = []
        for card in cards:
            price_node = _first_match(card, _PRICE_SELECTORS)
            if price_node is None:
                continue
            price = parse_price(price_node.get_text(" ", strip=True))
            if price is None:
                continue

            room_node = _first_match(card, _ROOM_SELECTORS)
            room_type = classify_room(
                room_node.get_text(" ", strip=True) if room_node else ""
            )
            # Only rates for the category we were asked about. An unclassifiable
            # unit name is skipped rather than guessed at -- a suite recorded as
            # a standard room would poison the competitor average for both.
            if room_type is not request.room_type:
                continue

            payloads.append(
                CompetitorPricePayload(
                    hotel_id=request.hotel_id,
                    competitor=self.competitor,
                    room_type=request.room_type,
                    check_in_date=request.check_in_date,
                    price=round(price, 2),
                    currency=self.settings.pricing.currency,
                    is_available=True,
                    source=self.name,
                )
            )

        logger.info(
            "Booking.com: %d card(s) parsed, %d rate(s) for %s %s",
            len(cards),
            len(payloads),
            request.room_type.value,
            request.check_in_date,
        )
        return payloads


__all__ = ["BookingScraper", "classify_room", "parse_price"]
