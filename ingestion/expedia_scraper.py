"""Expedia rate collection. Opt-in, disabled by default.

The same caveats as :mod:`ingestion.booking_scraper` apply in full: Expedia's
terms restrict automated access, ``robots.txt`` is honoured and will very
probably refuse, and the supported production route is a licensed feed --
Expedia's Rapid/Partner API, or a rate-shopping vendor. Enabling requires both
``INGESTION_SOURCE=expedia`` and ``INGESTION_ENABLE_REAL_SCRAPERS=true``.

What differs from the Booking implementation is instructive, which is why both
exist rather than one parameterised class:

* Expedia renders most of its results into a ``__NEXT_DATA__`` JSON blob. Where
  that blob is present, parsing it is far more robust than CSS selectors, so
  this scraper tries JSON first and falls back to markup.
* Its price strings carry a different set of qualifiers ("per night", "total").

Both differences are exactly the sort of thing a shared "generic OTA scraper"
would paper over with a pile of conditionals.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from database.models import Competitor
from ingestion.booking_scraper import classify_room, parse_price
from ingestion.scraper_base import (
    HttpCompetitorScraper,
    ScrapeRequest,
    ScraperParseError,
)
from monitoring.logging_config import get_logger
from streaming.events import CompetitorPricePayload

logger = get_logger(__name__)

#: Candidate CSS selectors for a property card, most specific first.
_CARD_SELECTORS: Tuple[str, ...] = (
    "div[data-stid='property-listing-results'] div[data-stid^='lodging-card']",
    "div[data-stid^='lodging-card']",
    "li[data-stid='property-listing-results-item']",
    "section.uitk-card",
)

_PRICE_SELECTORS: Tuple[str, ...] = (
    "div[data-test-id='price-summary'] span",
    "div[data-stid='price-summary'] span",
    "span.uitk-lockup-price",
)

_ROOM_SELECTORS: Tuple[str, ...] = (
    "div[data-stid='content-hotel-title']",
    "h3.uitk-heading",
    "h4.uitk-heading",
)

#: Matches "per night" so a total-stay price is not mistaken for a nightly rate.
_PER_NIGHT_PATTERN = re.compile(r"per\s*night|nightly|/\s*night", re.IGNORECASE)

#: Keys in the embedded JSON that plausibly hold a nightly rate.
_JSON_PRICE_KEYS = ("formattedDisplayPrice", "displayPrice", "lead", "amount")


def _walk(node: Any) -> Iterator[Any]:
    """Depth-first walk over a decoded JSON document."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def extract_next_data(html: str) -> Optional[Any]:
    """Return the decoded ``__NEXT_DATA__`` payload, or ``None`` if absent.

    Kept separate and pure so it can be unit-tested against a saved fixture
    without any network access.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    script = soup.find("script", id="__NEXT_DATA__")
    if script is None or not script.string:
        return None
    try:
        return json.loads(script.string)
    except json.JSONDecodeError as exc:
        raise ScraperParseError(f"__NEXT_DATA__ is not valid JSON: {exc}") from exc


def rates_from_json(document: Any) -> List[Tuple[str, float]]:
    """Pull ``(unit name, nightly price)`` pairs out of the embedded JSON.

    Structure-agnostic on purpose: it walks the document looking for objects
    that carry both a name and a price-shaped field, rather than hard-coding a
    path that changes with every deploy.
    """
    results: List[Tuple[str, float]] = []
    for node in _walk(document):
        name = node.get("name") or node.get("headingText") or node.get("title")
        if not isinstance(name, str):
            continue
        for key in _JSON_PRICE_KEYS:
            raw = node.get(key)
            if isinstance(raw, dict):
                raw = raw.get("formatted") or raw.get("amount")
            if isinstance(raw, (int, float)) and raw > 0:
                results.append((name, float(raw)))
                break
            if isinstance(raw, str):
                price = parse_price(raw)
                if price is not None:
                    results.append((name, price))
                    break
    return results


class ExpediaScraper(HttpCompetitorScraper):
    """Collects published rates from Expedia search results."""

    name = "expedia"
    competitor = Competitor.EXPEDIA
    base_url = "https://www.expedia.co.in"

    def build_url(self, request: ScrapeRequest) -> Tuple[str, Dict[str, str]]:
        """Search URL and query parameters for one city and one night."""
        params = {
            "destination": request.city,
            "startDate": request.check_in_date.isoformat(),
            "endDate": request.check_out_date.isoformat(),
            "rooms": "1",
            "adults": "2",
        }
        return f"{self.base_url}/Hotel-Search", params

    def parse(self, html: str, request: ScrapeRequest) -> List[CompetitorPricePayload]:
        """Extract rates, preferring the embedded JSON over the markup."""
        document = extract_next_data(html)
        if document is not None:
            pairs = rates_from_json(document)
            if pairs:
                logger.info(
                    "Expedia: parsed %d rate(s) from __NEXT_DATA__ for %s",
                    len(pairs),
                    request.check_in_date,
                )
                return self._to_payloads(pairs, request)
            logger.warning(
                "Expedia: __NEXT_DATA__ present but held no rates; falling back "
                "to markup parsing"
            )

        return self._parse_markup(html, request)

    def _parse_markup(
        self, html: str, request: ScrapeRequest
    ) -> List[CompetitorPricePayload]:
        from bs4 import BeautifulSoup

        from ingestion.booking_scraper import _first_match, _first_match_all

        soup = BeautifulSoup(html, "lxml")
        cards = _first_match_all(soup, _CARD_SELECTORS)
        if not cards:
            raise ScraperParseError(
                f"no lodging cards matched {list(_CARD_SELECTORS)} and no usable "
                f"__NEXT_DATA__ was present on the Expedia results page for "
                f"{request.city} {request.check_in_date}."
            )

        pairs: List[Tuple[str, float]] = []
        for card in cards:
            price_node = _first_match(card, _PRICE_SELECTORS)
            if price_node is None:
                continue
            price_text = price_node.get_text(" ", strip=True)
            price = parse_price(price_text)
            if price is None:
                continue
            # Expedia shows both nightly and total-stay prices. A total for a
            # multi-night stay recorded as a nightly rate would inflate the
            # competitor average by the length of stay.
            if request.nights > 1 and not _PER_NIGHT_PATTERN.search(price_text):
                continue

            room_node = _first_match(card, _ROOM_SELECTORS)
            pairs.append(
                (room_node.get_text(" ", strip=True) if room_node else "", price)
            )

        return self._to_payloads(pairs, request)

    def _to_payloads(
        self, pairs: List[Tuple[str, float]], request: ScrapeRequest
    ) -> List[CompetitorPricePayload]:
        """Filter to the requested category and wrap as validated payloads."""
        payloads: List[CompetitorPricePayload] = []
        for name, price in pairs:
            if classify_room(name) is not request.room_type:
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
        return payloads


__all__ = ["ExpediaScraper", "extract_next_data", "rates_from_json"]
