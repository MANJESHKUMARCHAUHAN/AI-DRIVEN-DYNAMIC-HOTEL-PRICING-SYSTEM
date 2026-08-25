"""The demo OTA: rate generation and the site it serves.

These test the *target*, not the scraper. If the site stops behaving like a
website -- non-deterministic rates, a robots.txt that permits what it should
refuse -- then every scraper test above it is measuring the wrong thing.
"""

from __future__ import annotations

from datetime import date, timedelta

import httpx
import pytest

from demo_ota.rates import ROOM_TYPES, known_cities, quote_rates

TOMORROW = date(2026, 9, 15)
TODAY = date(2026, 9, 1)


def _soup(html: str):
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "lxml")


def _cards(html: str) -> list:
    """Property cards as the scraper's v1 selectors would find them."""
    return _soup(html).select("div[data-testid='property-card']")


# --------------------------------------------------------------------------- #
# Rate generation
# --------------------------------------------------------------------------- #


class TestQuoteRates:
    def test_same_query_returns_the_same_rates(self) -> None:
        """Determinism is load-bearing, not a nicety.

        The dashboard previews a scrape and the background producer runs one
        moments later. If those disagreed, every "the rate I saw is not the rate
        that landed" report would be unfalsifiable.
        """
        first = quote_rates("Goa", TOMORROW, "deluxe", today=TODAY)
        second = quote_rates("Goa", TOMORROW, "deluxe", today=TODAY)
        assert [q.price for q in first] == [q.price for q in second]

    def test_every_listed_property_gets_a_quote(self) -> None:
        quotes = quote_rates("Jaipur", TOMORROW, "standard", today=TODAY)
        assert len(quotes) == 4
        assert len({q.property_name for q in quotes}) == 4

    def test_dearer_categories_cost_more(self) -> None:
        """Ordering must hold per property, not just on average."""
        by_room = {
            room: {q.property_name: q.price for q in quote_rates("Mumbai", TOMORROW, room, today=TODAY)}
            for room in ROOM_TYPES
        }
        shared = set.intersection(
            *[{name for name, price in mapping.items() if price} for mapping in by_room.values()]
        )
        assert shared, "no property was available in every category; widen the fixture"

        for name in shared:
            prices = [by_room[room][name] for room in ROOM_TYPES]
            assert prices == sorted(prices), f"{name} is not monotonic across categories"

    def test_weekends_cost_more_than_midweek(self) -> None:
        # 2026-09-18 is a Friday, 2026-09-16 a Wednesday.
        friday = quote_rates("Goa", date(2026, 9, 18), "deluxe", today=TODAY)
        wednesday = quote_rates("Goa", date(2026, 9, 16), "deluxe", today=TODAY)

        def mean(quotes):
            available = [q.price for q in quotes if q.price]
            return sum(available) / len(available)

        assert mean(friday) > mean(wednesday)

    def test_sold_out_properties_have_no_price(self) -> None:
        """A sold-out card must be representable, or the scraper never sees one."""
        seen_sold_out = False
        for offset in range(120):
            quotes = quote_rates(
                "Goa", TODAY + timedelta(days=offset), "suite", today=TODAY
            )
            if any(not q.is_available for q in quotes):
                seen_sold_out = True
                break
        assert seen_sold_out, "no sell-out in 120 nights; the availability model is broken"

    def test_sold_out_quote_reports_itself(self) -> None:
        quotes = quote_rates("Goa", TOMORROW, "deluxe", today=TODAY)
        for quote in quotes:
            assert quote.is_available is (quote.price is not None)

    def test_unknown_city_still_returns_inventory(self) -> None:
        """An OTA does not fail a search because a city is thin."""
        quotes = quote_rates("Kochi", TOMORROW, "deluxe", today=TODAY)
        assert quotes
        assert all(q.price is None or q.price > 0 for q in quotes)

    def test_unknown_room_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown room type"):
            quote_rates("Goa", TOMORROW, "presidential", today=TODAY)

    def test_lead_time_is_not_monotonic(self) -> None:
        """Cheap far out, firm near, spiking last-minute.

        A monotonic curve would make ``days_to_checkin`` a proxy for "cheaper",
        which is not how advance purchase actually behaves and would teach the
        model the wrong shape.
        """
        def mean_at(lead: int) -> float:
            quotes = quote_rates(
                "Bengaluru", TODAY + timedelta(days=lead), "deluxe", today=TODAY
            )
            available = [q.price for q in quotes if q.price]
            return sum(available) / len(available)

        far, mid, near = mean_at(90), mean_at(30), mean_at(1)
        assert near > mid, "last-minute rates should firm up, not keep falling"
        assert far < mid or near > far

    def test_rates_are_not_a_pure_function_of_the_inputs(self) -> None:
        """Two properties at the same position must not price identically.

        Without per-observation noise the competitor rate would be an exact
        function of the features, and a model would learn it perfectly -- the
        same failure ``search_demand`` had at 0.83 correlation.
        """
        quotes = [q.price for q in quote_rates("Udaipur", TOMORROW, "deluxe", today=TODAY) if q.price]
        assert len(set(quotes)) == len(quotes)


def test_known_cities_are_the_catalogue_cities() -> None:
    assert set(known_cities()) == {
        "Goa", "Jaipur", "Udaipur", "Bengaluru", "Mumbai", "New Delhi"
    }


# --------------------------------------------------------------------------- #
# The served site
# --------------------------------------------------------------------------- #


class TestDemoOTASite:
    def test_healthz(self, demo_ota_server: str) -> None:
        response = httpx.get(f"{demo_ota_server}/healthz", timeout=5)
        assert response.status_code == 200

    def test_robots_permits_search(self, demo_ota_server: str) -> None:
        body = httpx.get(f"{demo_ota_server}/robots.txt", timeout=5).text
        assert "Allow: /search" in body

    def test_robots_disallow_precedes_allow(self, demo_ota_server: str) -> None:
        """Rule order is the whole ballgame with ``urllib.robotparser``.

        It implements the 1994 first-match-wins standard, not Google's
        longest-match. An earlier version of this file put ``Allow: /`` above
        ``Disallow: /admin``, so ``/admin`` matched the blanket allow first and
        the disallow was dead text -- the scraper happily fetched a path it had
        been told not to. This asserts the ordering that makes the rule real.
        """
        body = httpx.get(f"{demo_ota_server}/robots.txt", timeout=5).text
        assert body.index("Disallow: /admin") < body.index("Allow: /search")
        assert "Allow: /\n" not in body, "a blanket allow would shadow every disallow"

    def test_search_returns_property_cards(self, demo_ota_server: str) -> None:
        response = httpx.get(
            f"{demo_ota_server}/search",
            params={"city": "Goa", "checkin": TOMORROW.isoformat(), "room": "deluxe"},
            timeout=5,
        )
        assert response.status_code == 200
        assert _cards(response.text)
        assert "&#8377;" in response.text or "₹" in response.text

    def test_v2_layout_shares_no_selectors_with_v1(self, demo_ota_server: str) -> None:
        """The redesign has to actually break things, or it proves nothing.

        Asserted with a CSS selector rather than a substring: the stylesheet
        mentions ``.property-card`` on every page, so ``"property-card" not in
        html`` is false even when the markup is completely different. The
        scraper matches elements, so the test has to match elements too.
        """
        params = {"city": "Goa", "checkin": TOMORROW.isoformat(), "room": "deluxe"}
        v2 = httpx.get(
            f"{demo_ota_server}/search", params={**params, "layout": "v2"}, timeout=5
        ).text

        assert not _cards(v2), "v1 selectors still match; the redesign is not a redesign"
        assert _soup(v2).select("article.listing"), "v2 markup missing"

    def test_unknown_room_type_returns_200_not_422(self, demo_ota_server: str) -> None:
        """A real OTA does not error because a filter matched nothing.

        The scraper has to be able to tell "no rates" from "page broke", and
        collapsing the first into an HTTP error removes that distinction.
        """
        response = httpx.get(
            f"{demo_ota_server}/search",
            params={"city": "Goa", "checkin": TOMORROW.isoformat(), "room": "igloo"},
            timeout=5,
        )
        assert response.status_code == 200
        assert not _cards(response.text)

    def test_admin_is_reachable_but_robots_disallowed(self, demo_ota_server: str) -> None:
        """robots.txt is a request, not an access control. Both halves matter."""
        assert httpx.get(f"{demo_ota_server}/admin", timeout=5).status_code == 200
