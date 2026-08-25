"""The demo OTA scraper, against a live server.

Real HTTP throughout. A mocked transport would exercise :meth:`parse` and skip
robots.txt, status-code mapping, connection reuse and the rate limiter -- which
between them are most of what a scraper *is*, and all of what tends to break.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from config import CompetitorSource, Settings
from database.models import Competitor, RoomType
from ingestion.demo_ota_scraper import DemoOTAScraper, classify_room, parse_price
from ingestion.scraper_base import (
    ScrapeRequest,
    ScraperBlocked,
    ScraperDisabled,
    ScraperParseError,
    get_scraper,
)


def _request(room_type: RoomType = RoomType.DELUXE, city: str = "Goa") -> ScrapeRequest:
    return ScrapeRequest(
        hotel_id="H001",
        city=city,
        room_type=room_type,
        check_in_date=date.today() + timedelta(days=7),
    )


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


class TestParsePrice:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("₹ 6,200", 6200.0),
            ("₹\xa06200", 6200.0),
            ("INR 6200", 6200.0),
            ("Rs. 12,450", 12450.0),
            ("₹ 1,23,456", 123456.0),          # Indian digit grouping
            ("₹ 6,200 per night", 6200.0),
        ],
    )
    def test_extracts_a_rate(self, text: str, expected: float) -> None:
        assert parse_price(text) == expected

    @pytest.mark.parametrize("text", ["Sold out", "", "no price here", "₹ 0"])
    def test_returns_none_rather_than_raising(self, text: str) -> None:
        """A card with no price is a sold-out property, not a parse failure.

        If this raised, every busy weekend would look like a broken parser.
        """
        assert parse_price(text) is None


class TestClassifyRoom:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("Standard Double Room", RoomType.STANDARD),
            ("Deluxe King Room with City View", RoomType.DELUXE),
            ("Premium Executive Room, Club Access", RoomType.PREMIUM),
            ("Junior Suite with Balcony", RoomType.SUITE),
        ],
    )
    def test_classifies_the_shipped_unit_names(self, text: str, expected: RoomType) -> None:
        assert classify_room(text) is expected

    def test_suite_beats_the_other_keywords(self) -> None:
        """Order matters: "Deluxe Suite" is a suite, not a deluxe."""
        assert classify_room("Deluxe Suite with King Bed") is RoomType.SUITE

    def test_unclassifiable_returns_none(self) -> None:
        assert classify_room("Room") is None
        assert classify_room("") is None


# --------------------------------------------------------------------------- #
# Live scraping
# --------------------------------------------------------------------------- #


class TestDemoOTAScraperLive:
    def test_factory_builds_it_without_the_third_party_flag(
        self, demo_ota_settings: Settings
    ) -> None:
        """demo_ota is not gated on INGESTION_ENABLE_REAL_SCRAPERS.

        That flag is about third-party terms of service. Gating a site we ship
        behind it would make the safe option the one hidden behind the scary
        switch.
        """
        assert demo_ota_settings.ingestion.enable_real_scrapers is False
        assert demo_ota_settings.ingestion.source is CompetitorSource.DEMO_OTA
        assert demo_ota_settings.ingestion.source.is_third_party is False

        scraper = get_scraper(demo_ota_settings)
        assert isinstance(scraper, DemoOTAScraper)
        scraper.close()

    def test_collects_rates_over_http(self, demo_ota_settings: Settings) -> None:
        with DemoOTAScraper(demo_ota_settings) as scraper:
            payloads = scraper.fetch(_request())

        assert payloads, "no rates scraped from a live page"
        for payload in payloads:
            assert payload.price > 0
            assert payload.source == "demo_ota"
            assert payload.currency == demo_ota_settings.pricing.currency
            assert payload.hotel_id == "H001"

    def test_returns_only_the_requested_category(self, demo_ota_settings: Settings) -> None:
        """A suite recorded as a standard room poisons both competitor averages."""
        with DemoOTAScraper(demo_ota_settings) as scraper:
            for room_type in RoomType:
                payloads = scraper.fetch(_request(room_type))
                assert all(p.room_type is room_type for p in payloads)

    def test_maps_properties_onto_distinct_brands(self, demo_ota_settings: Settings) -> None:
        """competitor_min_rate and competitor_max_rate need a real spread.

        Collapsing every listing onto one brand would leave those two features
        identical and carrying no information.
        """
        with DemoOTAScraper(demo_ota_settings) as scraper:
            payloads = scraper.fetch(_request())

        brands = [p.competitor for p in payloads]
        assert len(set(brands)) == len(brands), "a brand was reused within one page"
        assert set(brands) <= set(Competitor)

    def test_brand_assignment_is_stable(self, demo_ota_settings: Settings) -> None:
        """The same listing must always be the same competitor across runs."""
        with DemoOTAScraper(demo_ota_settings) as scraper:
            first = scraper.fetch(_request())
            second = scraper.fetch(_request())
        assert [p.competitor for p in first] == [p.competitor for p in second]
        assert [p.price for p in first] == [p.price for p in second]

    def test_sold_out_cards_are_skipped_not_fatal(self, demo_ota_settings: Settings) -> None:
        """Across many nights some pages carry sold-out cards; none may raise."""
        with DemoOTAScraper(demo_ota_settings) as scraper:
            total = 0
            for offset in range(1, 40):
                request = ScrapeRequest(
                    hotel_id="H001",
                    city="Goa",
                    room_type=RoomType.SUITE,
                    check_in_date=date.today() + timedelta(days=offset),
                )
                total += len(scraper.fetch(request))
        assert total > 0


class TestFailureModes:
    def test_redesigned_markup_raises_rather_than_returning_empty(
        self, demo_ota_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """"No rates" and "our parser broke" must not look the same.

        Returning an empty list on a redesign would tell the pricing engine the
        competitive set published nothing, and it would widen every competitor
        band accordingly -- confidently, and on no data at all.
        """
        monkeypatch.setattr(demo_ota_settings.ingestion, "demo_ota_layout", "v2")

        with DemoOTAScraper(demo_ota_settings) as scraper:
            with pytest.raises(ScraperParseError, match="no property cards matched"):
                scraper.fetch(_request())

    def test_collect_absorbs_the_parse_error(
        self, demo_ota_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One broken page must not take the whole collection run down."""
        monkeypatch.setattr(demo_ota_settings.ingestion, "demo_ota_layout", "v2")

        with DemoOTAScraper(demo_ota_settings) as scraper:
            assert scraper.collect(_request()) == []
            assert scraper.error_count > 0

    def test_robots_disallowed_path_is_refused(self, demo_ota_settings: Settings) -> None:
        with DemoOTAScraper(demo_ota_settings) as scraper:
            with pytest.raises(ScraperBlocked, match="robots.txt disallows"):
                scraper._check_robots(f"{scraper.base_url}/admin")

    def test_robots_allowed_path_is_permitted(self, demo_ota_settings: Settings) -> None:
        with DemoOTAScraper(demo_ota_settings) as scraper:
            scraper._check_robots(f"{scraper.base_url}/search")  # must not raise

    def test_unreadable_robots_is_treated_as_disallow(
        self, demo_ota_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fail closed. The permissive alternative sends traffic nobody allowed."""
        monkeypatch.setattr(
            demo_ota_settings.ingestion,
            "demo_ota_base_url",
            "http://127.0.0.1:9",  # discard port; nothing listens
        )
        scraper = DemoOTAScraper(demo_ota_settings)
        with pytest.raises(ScraperBlocked, match="could not read"):
            scraper._check_robots("http://127.0.0.1:9/search")
        scraper.close()

    def test_blocked_is_not_retried(
        self, demo_ota_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Hammering a site that just said no turns a soft block into a ban."""
        monkeypatch.setattr(demo_ota_settings.ingestion, "max_retries", 3)
        attempts = {"count": 0}

        scraper = DemoOTAScraper(demo_ota_settings)

        def _always_blocked(request):
            attempts["count"] += 1
            raise ScraperBlocked("nope")

        monkeypatch.setattr(scraper, "fetch", _always_blocked)
        with pytest.raises(ScraperBlocked):
            scraper.collect(_request())

        assert attempts["count"] == 1, "a block must not be retried"
        scraper.close()


class TestThirdPartyStillGated:
    """The demo path must not have loosened the third-party lock."""

    @pytest.mark.parametrize("source", ["booking", "expedia"])
    def test_third_party_needs_the_explicit_flag(
        self, source: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from config import IngestionSettings

        monkeypatch.setenv("INGESTION_SOURCE", source)
        monkeypatch.setenv("INGESTION_ENABLE_REAL_SCRAPERS", "false")
        with pytest.raises(ValueError, match="ENABLE_REAL_SCRAPERS"):
            IngestionSettings()

    def test_factory_refuses_a_disabled_third_party_source(
        self, demo_ota_settings: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            demo_ota_settings.ingestion, "source", CompetitorSource.BOOKING
        )
        monkeypatch.setattr(
            demo_ota_settings.ingestion, "enable_real_scrapers", False
        )
        with pytest.raises(ScraperDisabled, match="ENABLE_REAL_SCRAPERS"):
            get_scraper(demo_ota_settings)
