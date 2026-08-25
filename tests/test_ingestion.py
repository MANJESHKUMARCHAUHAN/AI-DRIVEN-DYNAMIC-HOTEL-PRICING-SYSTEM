"""Tests for the competitor ingestion layer.

Three things under test:

* **The synthetic generator**, which is the default data source and therefore
  the one the demo pipeline's correctness rests on.
* **The scraper interface and its safety interlocks** -- ADR-004 says live
  scraping is off unless two independent settings say otherwise, and that is
  worth a test rather than a comment.
* **The scrapers' pure parsing functions**, exercised against saved HTML
  fixtures. No test in this file makes a network request.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from config import CompetitorSource
from database.models import Competitor, RoomType
from ingestion.booking_scraper import BookingScraper, classify_room, parse_price
from ingestion.expedia_scraper import ExpediaScraper, extract_next_data, rates_from_json
from ingestion.scraper_base import (
    ScrapeRequest,
    ScraperDisabled,
    ScraperError,
    ScraperParseError,
    build_requests,
    get_scraper,
)
from ingestion.synthetic_generator import (
    DEFAULT_HORIZONS,
    SyntheticCompetitorGenerator,
    default_catalog,
)
from ingestion.validator import EventRejected, EventValidator, ReferenceData
from streaming.events import CompetitorPricePayload

AS_OF = date(2026, 8, 24)
STAY = date(2026, 9, 15)


def _request(**overrides) -> ScrapeRequest:
    row = dict(
        hotel_id="H001", city="Mumbai", room_type=RoomType.DELUXE, check_in_date=STAY
    )
    row.update(overrides)
    return ScrapeRequest(**row)


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #


class TestScrapeRequest:
    def test_derived_dates(self) -> None:
        request = _request()
        assert request.check_out_date == STAY + timedelta(days=1)

    def test_multi_night_checkout(self) -> None:
        assert _request(nights=3).check_out_date == STAY + timedelta(days=3)

    def test_build_requests_is_the_cartesian_product(self) -> None:
        requests = build_requests(
            [("H001", "Mumbai"), ("H004", "Goa")],
            horizons=[1, 7, 30],
            room_types=[RoomType.STANDARD, RoomType.SUITE],
            today=AS_OF,
        )
        assert len(requests) == 2 * 3 * 2
        assert {r.check_in_date for r in requests} == {
            AS_OF + timedelta(days=d) for d in (1, 7, 30)
        }


class TestScraperFactory:
    def test_default_source_is_synthetic(self, settings) -> None:
        """ADR-004: the shipped default must never reach a third-party site."""
        scraper = get_scraper(settings)
        assert isinstance(scraper, SyntheticCompetitorGenerator)
        assert scraper.requires_network is False

    def test_live_scraper_refused_without_the_second_switch(
        self, settings, monkeypatch
    ) -> None:
        """The config model rejects this combination, but a caller can build
        settings programmatically -- so the factory locks the door too."""
        monkeypatch.setattr(settings.ingestion, "source", CompetitorSource.BOOKING)
        monkeypatch.setattr(settings.ingestion, "enable_real_scrapers", False)

        with pytest.raises(ScraperDisabled, match="ENABLE_REAL_SCRAPERS"):
            get_scraper(settings)

    def test_live_scrapers_are_constructible_when_explicitly_enabled(
        self, settings, monkeypatch
    ) -> None:
        monkeypatch.setattr(settings.ingestion, "enable_real_scrapers", True)

        monkeypatch.setattr(settings.ingestion, "source", CompetitorSource.BOOKING)
        assert isinstance(get_scraper(settings), BookingScraper)

        monkeypatch.setattr(settings.ingestion, "source", CompetitorSource.EXPEDIA)
        assert isinstance(get_scraper(settings), ExpediaScraper)

    def test_configuration_rejects_the_unsafe_combination_outright(self) -> None:
        from config.settings import IngestionSettings

        with pytest.raises(ValueError, match="ENABLE_REAL_SCRAPERS"):
            IngestionSettings(
                source=CompetitorSource.BOOKING,
                enable_real_scrapers=False,
                _env_file=None,
            )


# --------------------------------------------------------------------------- #
# The synthetic generator
# --------------------------------------------------------------------------- #


class TestSyntheticGenerator:
    @pytest.fixture
    def generator(self, settings) -> SyntheticCompetitorGenerator:
        return SyntheticCompetitorGenerator(settings, seed=42)

    def test_emits_validated_payloads(self, generator) -> None:
        payloads = generator.fetch(_request(), as_of=AS_OF)
        assert payloads
        assert all(isinstance(p, CompetitorPricePayload) for p in payloads)
        assert all(p.price > 0 for p in payloads)
        assert all(p.room_type is RoomType.DELUXE for p in payloads)
        assert all(p.check_in_date == STAY for p in payloads)
        assert all(p.source == "synthetic" for p in payloads)

    def test_same_day_polling_is_reproducible(self, generator) -> None:
        first = generator.fetch(_request(), as_of=AS_OF)
        second = generator.fetch(_request(), as_of=AS_OF)
        assert [p.price for p in first] == [p.price for p in second]

    def test_the_market_moves_between_days(self, generator) -> None:
        """A feed that returns identical rates forever teaches a model nothing."""
        today = generator.fetch(_request(), as_of=AS_OF)
        tomorrow = generator.fetch(_request(), as_of=AS_OF + timedelta(days=1))
        assert [p.price for p in today] != [p.price for p in tomorrow]

    def test_coverage_is_incomplete(self, generator) -> None:
        """Missing competitor data is a case the pipeline must survive, so the
        generator produces it deliberately."""
        emitted = 0
        possible = 0
        for horizon in range(1, 61):
            request = _request(check_in_date=AS_OF + timedelta(days=horizon))
            emitted += len(generator.fetch(request, as_of=AS_OF))
            possible += len(Competitor)
        assert 0.7 < emitted / possible < 0.95

    def test_competitor_set_has_a_structural_price_spread(self, generator) -> None:
        """min and max rate must differ by something other than noise, or both
        features are the same column twice."""
        prices = {}
        for horizon in range(1, 40):
            request = _request(check_in_date=AS_OF + timedelta(days=horizon))
            for payload in generator.fetch(request, as_of=AS_OF):
                prices.setdefault(payload.competitor, []).append(payload.price)

        means = {c: sum(v) / len(v) for c, v in prices.items() if v}
        assert len(means) == len(Competitor)
        assert max(means.values()) / min(means.values()) > 1.05

    def test_peak_dates_are_priced_above_troughs(self, settings) -> None:
        """Goa on New Year's Eve versus Goa in the monsoon."""
        generator = SyntheticCompetitorGenerator(settings, seed=42)
        peak = generator.fetch(
            _request(hotel_id="H004", city="Goa", check_in_date=date(2026, 12, 30)),
            as_of=date(2026, 11, 1),
        )
        trough = generator.fetch(
            _request(hotel_id="H004", city="Goa", check_in_date=date(2026, 7, 15)),
            as_of=date(2026, 6, 1),
        )
        assert min(p.price for p in peak) > max(p.price for p in trough)

    def test_unknown_hotel_is_an_explicit_error(self, generator) -> None:
        with pytest.raises(ScraperError, match="no market profile"):
            generator.fetch(_request(hotel_id="H999"), as_of=AS_OF)

    def test_catalog_override_moves_the_price_level(self, settings) -> None:
        """scripts/run_producer.py passes the real room prices from the database."""
        cheap = SyntheticCompetitorGenerator(
            settings, seed=42, catalog={("H001", RoomType.DELUXE): 3_000.0}
        )
        expensive = SyntheticCompetitorGenerator(
            settings, seed=42, catalog={("H001", RoomType.DELUXE): 12_000.0}
        )
        cheap_prices = [p.price for p in cheap.fetch(_request(), as_of=AS_OF)]
        rich_prices = [p.price for p in expensive.fetch(_request(), as_of=AS_OF)]
        assert max(cheap_prices) < min(rich_prices)

    def test_default_catalog_covers_every_hotel_and_room(self) -> None:
        catalog = default_catalog()
        assert len(catalog) == 8 * len(RoomType)
        assert all(price > 0 for price in catalog.values())

    def test_request_plan_covers_the_configured_horizons(self, generator) -> None:
        requests = generator.requests(horizons=[1, 7], hotel_ids=["H001"], as_of=AS_OF)
        assert len(requests) == 2 * len(RoomType)
        assert {r.check_in_date for r in requests} == {
            AS_OF + timedelta(days=1), AS_OF + timedelta(days=7)
        }

    def test_stream_respects_max_events(self, generator) -> None:
        events = list(generator.stream(max_events=7, interval_seconds=0))
        assert len(events) == 7

    def test_stream_terminates_after_the_requested_passes(self, generator) -> None:
        events = list(
            generator.stream(
                horizons=[1], hotel_ids=["H001"], passes=1, interval_seconds=0
            )
        )
        assert 0 < len(events) <= len(RoomType) * len(Competitor)

    def test_default_horizons_span_short_and_long_lead(self) -> None:
        assert min(DEFAULT_HORIZONS) <= 3
        assert max(DEFAULT_HORIZONS) >= 30


# --------------------------------------------------------------------------- #
# Semantic validation
# --------------------------------------------------------------------------- #


class TestEventValidator:
    def _payload(self, **overrides) -> CompetitorPricePayload:
        row = dict(
            hotel_id="H001",
            competitor=Competitor.BOOKING,
            room_type=RoomType.DELUXE,
            check_in_date=STAY,
            price=6_200.0,
        )
        row.update(overrides)
        return CompetitorPricePayload(**row)

    @pytest.fixture
    def validator(self, seeded_session) -> EventValidator:
        validator = EventValidator(reference=ReferenceData())
        validator.reference.refresh(seeded_session)
        return validator

    def test_known_hotel_and_room_accepted(self, validator, seeded_session) -> None:
        validator.validate(self._payload(), seeded_session)
        assert validator.stats.accepted == 1

    def test_unknown_hotel_rejected(self, validator, seeded_session) -> None:
        with pytest.raises(EventRejected) as excinfo:
            validator.validate(self._payload(hotel_id="H999"), seeded_session)
        assert excinfo.value.reason == "unknown_hotel"
        assert validator.stats.by_reason == {"unknown_hotel": 1}

    def test_room_the_hotel_does_not_sell_rejected(
        self, validator, seeded_session
    ) -> None:
        from database.models import Room

        seeded_session.execute(
            Room.__table__.delete().where(Room.room_type == RoomType.PREMIUM)
        )
        seeded_session.commit()
        validator.reference.refresh(seeded_session)

        with pytest.raises(EventRejected) as excinfo:
            validator.validate(self._payload(room_type=RoomType.PREMIUM), seeded_session)
        assert excinfo.value.reason == "unknown_room_type"

    def test_empty_database_is_permissive_by_default(self, db_session) -> None:
        """A consumer that starts before the seeder finishes should not reject
        the whole stream over a start-up race; the foreign keys still catch it."""
        validator = EventValidator(reference=ReferenceData())
        validator.reference.refresh(db_session)
        validator.validate(self._payload(), db_session)
        assert validator.stats.accepted == 1

    def test_empty_database_is_fatal_in_strict_mode(self, db_session) -> None:
        validator = EventValidator(reference=ReferenceData(), strict_when_empty=True)
        validator.reference.refresh(db_session)
        with pytest.raises(EventRejected) as excinfo:
            validator.validate(self._payload(), db_session)
        assert excinfo.value.reason == "no_reference_data"

    def test_reference_data_is_cached_not_re_read_per_event(
        self, validator, seeded_session
    ) -> None:
        """At a few hundred events a second a per-message SELECT would be the
        bottleneck, and all it would ever learn is 'still the same hotels'."""
        before = validator.reference.refresh_count
        for _ in range(50):
            validator.validate(self._payload(), seeded_session)
        assert validator.reference.refresh_count == before

    def test_stale_cache_is_refreshed(self, seeded_session) -> None:
        validator = EventValidator(reference=ReferenceData(ttl_seconds=0.0))
        validator.reference.refresh(seeded_session)
        before = validator.reference.refresh_count
        validator.validate(self._payload(), seeded_session)
        assert validator.reference.refresh_count == before + 1

    def test_is_valid_returns_a_boolean(self, validator, seeded_session) -> None:
        assert validator.is_valid(self._payload(), seeded_session) is True
        assert validator.is_valid(self._payload(hotel_id="H999"), seeded_session) is False


# --------------------------------------------------------------------------- #
# Scraper parsing -- pure functions, saved fixtures, no network
# --------------------------------------------------------------------------- #

BOOKING_HTML = """
<html><body>
  <div data-testid="property-card">
    <span data-testid="recommended-units"><h4>Deluxe King Room</h4></span>
    <span data-testid="price-and-discounted-price">₹ 6,450</span>
  </div>
  <div data-testid="property-card">
    <span data-testid="recommended-units"><h4>Executive Suite</h4></span>
    <span data-testid="price-and-discounted-price">₹ 18,900</span>
  </div>
  <div data-testid="property-card">
    <span data-testid="recommended-units"><h4>Deluxe Twin Room</h4></span>
    <span data-testid="price-and-discounted-price">INR 5,980</span>
  </div>
  <div data-testid="property-card">
    <span data-testid="recommended-units"><h4>Deluxe Room</h4></span>
  </div>
</body></html>
"""

EXPEDIA_HTML = """
<html><body>
<script id="__NEXT_DATA__" type="application/json">
{"props": {"results": [
  {"name": "Deluxe Room, City View", "formattedDisplayPrice": "₹7,120"},
  {"name": "Presidential Suite", "formattedDisplayPrice": "₹24,500"},
  {"name": "Deluxe Room", "displayPrice": 6890}
]}}
</script>
</body></html>
"""


class TestPriceParsing:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("₹ 6,450", 6450.0),
            ("₹6450", 6450.0),
            ("INR 5,980", 5980.0),
            ("Rs. 12,300", 12300.0),
            ("₹ 1,04,500", 104500.0),
        ],
    )
    def test_recognised_formats(self, text: str, expected: float) -> None:
        assert parse_price(text) == expected

    @pytest.mark.parametrize("text", ["", "Sold out", "Price on request", "₹ 0"])
    def test_unparseable_returns_none(self, text: str) -> None:
        """A card with no price is a sold-out property, not an error."""
        assert parse_price(text) is None

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("Deluxe King Room", RoomType.DELUXE),
            ("Executive Suite", RoomType.SUITE),
            ("Superior Twin", RoomType.PREMIUM),
            ("Standard Double Room", RoomType.STANDARD),
            ("Luxury Room with Balcony", RoomType.DELUXE),
        ],
    )
    def test_room_classification(self, name: str, expected: RoomType) -> None:
        assert classify_room(name) is expected

    def test_unclassifiable_room_returns_none(self) -> None:
        """Better to skip a unit than to record a suite as a standard room and
        poison the competitor average for both categories."""
        assert classify_room("Pod 7B") is None


class TestBookingParser:
    @pytest.fixture
    def scraper(self, settings, monkeypatch) -> BookingScraper:
        monkeypatch.setattr(settings.ingestion, "enable_real_scrapers", True)
        return BookingScraper(settings)

    def test_extracts_only_the_requested_category(self, scraper) -> None:
        payloads = scraper.parse(BOOKING_HTML, _request(room_type=RoomType.DELUXE))
        assert [p.price for p in payloads] == [6450.0, 5980.0]
        assert all(p.competitor is Competitor.BOOKING for p in payloads)

    def test_suite_request_finds_the_suite(self, scraper) -> None:
        payloads = scraper.parse(BOOKING_HTML, _request(room_type=RoomType.SUITE))
        assert [p.price for p in payloads] == [18900.0]

    def test_cards_without_a_price_are_skipped(self, scraper) -> None:
        """The fourth fixture card has no price element."""
        assert len(scraper.parse(BOOKING_HTML, _request())) == 2

    def test_changed_markup_raises_instead_of_returning_nothing(self, scraper) -> None:
        """'No rates published' and 'our parser broke' must never look alike."""
        with pytest.raises(ScraperParseError, match="no property cards matched"):
            scraper.parse("<html><body><div class='v2-card'></div></body></html>",
                          _request())

    def test_search_url_carries_the_stay_dates(self, scraper) -> None:
        url, params = scraper.build_url(_request())
        assert url.startswith("https://www.booking.com")
        assert params["checkin"] == STAY.isoformat()
        assert params["checkout"] == (STAY + timedelta(days=1)).isoformat()
        assert params["ss"] == "Mumbai"


class TestExpediaParser:
    @pytest.fixture
    def scraper(self, settings, monkeypatch) -> ExpediaScraper:
        monkeypatch.setattr(settings.ingestion, "enable_real_scrapers", True)
        return ExpediaScraper(settings)

    def test_embedded_json_is_preferred_over_markup(self, scraper) -> None:
        payloads = scraper.parse(EXPEDIA_HTML, _request(room_type=RoomType.DELUXE))
        assert sorted(p.price for p in payloads) == [6890.0, 7120.0]

    def test_next_data_extraction(self) -> None:
        document = extract_next_data(EXPEDIA_HTML)
        assert document is not None
        assert len(rates_from_json(document)) == 3

    def test_absent_next_data_returns_none(self) -> None:
        assert extract_next_data("<html><body>nothing</body></html>") is None

    def test_malformed_next_data_raises(self) -> None:
        html = '<html><script id="__NEXT_DATA__">{not json}</script></html>'
        with pytest.raises(ScraperParseError, match="not valid JSON"):
            extract_next_data(html)

    def test_falls_back_to_markup_when_json_is_absent(self, scraper) -> None:
        html = """
        <html><body>
          <div data-stid="lodging-card-1">
            <h3 class="uitk-heading">Deluxe Garden Room</h3>
            <div data-test-id="price-summary"><span>₹ 8,100</span></div>
          </div>
        </body></html>
        """
        payloads = scraper.parse(html, _request(room_type=RoomType.DELUXE))
        assert [p.price for p in payloads] == [8100.0]

    def test_unrecognisable_page_raises(self, scraper) -> None:
        with pytest.raises(ScraperParseError, match="no lodging cards matched"):
            scraper.parse("<html><body><p>Access denied</p></body></html>", _request())

    def test_search_url_carries_the_stay_dates(self, scraper) -> None:
        url, params = scraper.build_url(_request())
        assert "expedia" in url
        assert params["startDate"] == STAY.isoformat()
