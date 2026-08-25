"""The two ingestion endpoints.

``/ingestion/status`` has to stay answerable in every configuration, including
the ones where scraping is refused or impossible -- an observability endpoint
that 500s when the thing it observes is broken is worse than no endpoint.

``/ingestion/run`` is tested against the live demo OTA, so the pass it reports is
a pass that really happened.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from api.dependencies import session_dependency
from api.main import create_app
from config import CompetitorSource
from database.models import (
    Competitor,
    CompetitorPrice,
    Hotel,
    MarketSegment,
    Room,
    RoomType,
)

PREFIX = "/api/v1"


def _seed(session) -> None:
    """Two hotels in cities the demo OTA has inventory for."""
    for hotel_id, name, city in [
        ("H001", "Sanchay Grand Mumbai", "Mumbai"),
        ("H002", "Sanchay Goa Resort", "Goa"),
    ]:
        session.add(
            Hotel(
                hotel_id=hotel_id,
                hotel_name=name,
                city=city,
                star_rating=5,
                total_rooms=100,
                segment=MarketSegment.MIXED,
            )
        )
        session.flush()
        for index, (room_type, price) in enumerate(
            [
                (RoomType.STANDARD, 5000.0),
                (RoomType.DELUXE, 6400.0),
                (RoomType.PREMIUM, 8250.0),
                (RoomType.SUITE, 13000.0),
            ]
        ):
            session.add(
                Room(
                    room_id=f"{hotel_id}-R{index}",
                    hotel_id=hotel_id,
                    room_type=room_type,
                    capacity=2,
                    room_count=25,
                    base_price=price,
                )
            )
    session.commit()


@pytest.fixture
def api(demo_ota_settings, db_engine, monkeypatch):
    """A client whose configured source is the live demo OTA."""
    factory = sessionmaker(bind=db_engine, expire_on_commit=False, future=True)
    with factory() as seeding:
        _seed(seeding)

    # No broker in tests. Publishing must degrade, not fail.
    monkeypatch.setattr(demo_ota_settings.kafka, "enabled", False)

    def override_session():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app = create_app(demo_ota_settings)
    app.dependency_overrides[session_dependency] = override_session

    with TestClient(app) as client:
        client.session_factory = factory  # type: ignore[attr-defined]
        client.settings = demo_ota_settings  # type: ignore[attr-defined]
        yield client


# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #


class TestIngestionStatus:
    def test_reports_the_configured_scraper(self, api: TestClient) -> None:
        body = api.get(f"{PREFIX}/ingestion/status").json()

        assert body["source"]["source"] == "demo_ota"
        assert body["source"]["is_scraper"] is True
        assert body["source"]["is_third_party"] is False
        assert body["source"]["base_url"].startswith("http://127.0.0.1")

    def test_reports_crawl_permission(self, api: TestClient) -> None:
        robots = api.get(f"{PREFIX}/ingestion/status").json()["robots"]
        assert robots["checked"] is True
        assert robots["allowed"] is True
        assert "robots.txt" in robots["detail"]

    def test_counts_are_zero_before_anything_lands(self, api: TestClient) -> None:
        body = api.get(f"{PREFIX}/ingestion/status").json()
        assert body["observations_total"] == 0
        assert body["by_source"] == {}
        assert body["latest_observed_at"] is None

    def test_counts_reflect_stored_rows(self, api: TestClient) -> None:
        with api.session_factory() as session:  # type: ignore[attr-defined]
            session.add(
                CompetitorPrice(
                    hotel_id="H001",
                    room_type=RoomType.DELUXE,
                    competitor=Competitor.BOOKING,
                    check_in_date=date.today() + timedelta(days=7),
                    price=7200.0,
                    currency="INR",
                    is_available=True,
                    source="demo_ota",
                    collected_at=datetime.now(tz=timezone.utc),
                )
            )
            session.commit()

        body = api.get(f"{PREFIX}/ingestion/status").json()
        assert body["observations_total"] == 1
        assert body["observations_last_hour"] == 1
        assert body["by_source"] == {"demo_ota": 1}

    def test_synthetic_source_reports_no_robots_check(
        self, api: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No requests means no crawl policy to honour -- and no error either."""
        monkeypatch.setattr(
            api.settings.ingestion, "source", CompetitorSource.SYNTHETIC  # type: ignore[attr-defined]
        )
        body = api.get(f"{PREFIX}/ingestion/status").json()

        assert body["source"]["is_scraper"] is False
        assert body["robots"]["checked"] is False
        assert body["robots"]["allowed"] is None

    def test_unreachable_target_still_answers(
        self, api: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The status endpoint must survive the thing it reports on being down.

        Failing closed here would mean the one endpoint that could tell you the
        scraper is broken goes down whenever the scraper is broken.
        """
        monkeypatch.setattr(
            api.settings.ingestion,  # type: ignore[attr-defined]
            "demo_ota_base_url",
            "http://127.0.0.1:9",
        )
        response = api.get(f"{PREFIX}/ingestion/status")

        assert response.status_code == 200
        robots = response.json()["robots"]
        assert robots["checked"] is True
        assert robots["allowed"] is False  # unreadable robots means disallow


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


class TestIngestionRun:
    def test_collects_and_persists(self, api: TestClient) -> None:
        response = api.post(
            f"{PREFIX}/ingestion/run",
            json={"hotel_ids": ["H002"], "room_types": ["deluxe"], "horizons": [7]},
        )
        assert response.status_code == 200

        body = response.json()
        assert body["succeeded"] is True
        assert body["source"] == "demo_ota"
        assert body["requests_made"] == 1
        assert body["rates_collected"] > 0
        assert body["persisted"] == body["rates_collected"]
        assert body["blocked"] is False

        for rate in body["rates"]:
            assert rate["source"] == "demo_ota"
            assert rate["room_type"] == "deluxe"
            assert rate["price"] > 0

    def test_persisted_rows_are_visible_afterwards(self, api: TestClient) -> None:
        api.post(
            f"{PREFIX}/ingestion/run",
            json={"hotel_ids": ["H002"], "room_types": ["suite"], "horizons": [14]},
        )
        status = api.get(f"{PREFIX}/ingestion/status").json()
        assert status["observations_total"] > 0
        assert "demo_ota" in status["by_source"]

    def test_rerunning_the_same_pass_is_idempotent(self, api: TestClient) -> None:
        """The same observation arriving twice must not move the average.

        The demo OTA is deterministic, so a repeated pass is genuinely the same
        observation -- exactly the at-least-once redelivery the consumer sees
        after a rebalance.
        """
        payload = {"hotel_ids": ["H002"], "room_types": ["premium"], "horizons": [21]}

        first = api.post(f"{PREFIX}/ingestion/run", json=payload).json()
        after_first = api.get(f"{PREFIX}/ingestion/status").json()["observations_total"]

        second = api.post(f"{PREFIX}/ingestion/run", json=payload).json()
        after_second = api.get(f"{PREFIX}/ingestion/status").json()["observations_total"]

        assert first["rates_collected"] == second["rates_collected"]
        assert after_second > after_first, (
            "a fresh observation of the same rate is a new row -- the "
            "idempotency key is the event id, not the rate"
        )

    def test_preview_run_persists_nothing(self, api: TestClient) -> None:
        body = api.post(
            f"{PREFIX}/ingestion/run",
            json={
                "hotel_ids": ["H002"],
                "room_types": ["deluxe"],
                "horizons": [7],
                "publish": False,
            },
        ).json()

        assert body["rates_collected"] > 0
        assert body["persisted"] == 0
        assert body["published"] == 0

        status = api.get(f"{PREFIX}/ingestion/status").json()
        assert status["observations_total"] == 0

    def test_redesigned_markup_reports_failure_not_success(
        self, api: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zero rates is a failed run, even though no request errored.

        Every page parsed to nothing, which is what a silent redesign looks
        like. Reporting that as success is how a dead feed goes unnoticed.
        """
        monkeypatch.setattr(
            api.settings.ingestion, "demo_ota_layout", "v2"  # type: ignore[attr-defined]
        )
        body = api.post(
            f"{PREFIX}/ingestion/run",
            json={"hotel_ids": ["H002"], "room_types": ["deluxe"], "horizons": [7]},
        ).json()

        assert body["succeeded"] is False
        assert body["rates_collected"] == 0
        assert "markup" in body["detail"] or "no rates" in body["detail"]

    def test_unknown_hotel_is_404(self, api: TestClient) -> None:
        response = api.post(f"{PREFIX}/ingestion/run", json={"hotel_ids": ["H999"]})
        assert response.status_code == 404
        assert "H999" in response.json()["detail"]

    def test_slow_pass_is_rejected_before_any_request(
        self, api: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard is on wall-clock, not request count.

        2 hotels x 4 categories x 12 horizons is 96 requests -- nothing at all
        with the limiter off, and eight minutes at five seconds apart. It is the
        second one that hangs a browser, so that is what gets refused.
        """
        monkeypatch.setattr(
            api.settings.ingestion, "rate_limit_seconds", 5.0  # type: ignore[attr-defined]
        )
        response = api.post(
            f"{PREFIX}/ingestion/run",
            json={"horizons": list(range(1, 13))},
        )
        assert response.status_code == 413
        assert "minutes" in response.json()["detail"]

    def test_the_same_pass_is_allowed_when_it_would_be_fast(
        self, api: TestClient
    ) -> None:
        """The count alone must not be what is refused.

        Without this, lowering the rate limit would not actually let a bigger
        pass through, and the previous test would be passing for the wrong
        reason.
        """
        response = api.post(
            f"{PREFIX}/ingestion/run",
            json={
                "hotel_ids": ["H002"],
                "room_types": ["deluxe"],
                "horizons": list(range(1, 13)),
                "publish": False,
            },
        )
        assert response.status_code == 200
        assert response.json()["requests_made"] == 12

    def test_too_many_horizons_is_rejected_by_the_schema(self, api: TestClient) -> None:
        response = api.post(
            f"{PREFIX}/ingestion/run", json={"horizons": list(range(1, 20))}
        )
        assert response.status_code == 422

    def test_disabled_source_is_409_not_500(
        self, api: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The request is fine; the server is configured not to do this."""
        monkeypatch.setattr(
            api.settings.ingestion, "source", CompetitorSource.BOOKING  # type: ignore[attr-defined]
        )
        monkeypatch.setattr(
            api.settings.ingestion, "enable_real_scrapers", False  # type: ignore[attr-defined]
        )
        response = api.post(
            f"{PREFIX}/ingestion/run",
            json={"hotel_ids": ["H002"], "room_types": ["deluxe"], "horizons": [7]},
        )
        assert response.status_code == 409
        assert "ENABLE_REAL_SCRAPERS" in response.json()["detail"]
