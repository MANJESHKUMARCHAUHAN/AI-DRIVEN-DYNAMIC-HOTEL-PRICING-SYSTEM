"""Tests for the nine business endpoints.

``tests/test_api.py`` covers the application shell -- CORS, correlation ids, the
error envelope, health. This file covers what the service actually does.

These run against in-memory SQLite with the session dependency overridden, and a
model registry pointed at an empty directory. That combination is not a
compromise: it *is* the degraded path, which is the one most likely to break
unnoticed. No artifacts, so pricing has to fall back to stored history and still
return a usable, guardrailed number. The tests that need models inject fakes.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from api.dependencies import get_registry_dependency, session_dependency
from api.main import create_app
from database.models import (
    Booking,
    Competitor,
    CompetitorPrice,
    DemandFeature,
    Hotel,
    MarketSegment,
    PricingDecision,
    Room,
    RoomType,
    Season,
)
from models.model_registry import LoadedModels, ModelRegistry

#: Far enough ahead to be a real pricing question, near enough to stay inside
#: the request schema's horizon.
STAY = date.today() + timedelta(days=21)


def seed_api_data(session) -> None:
    """One hotel, two room types, and enough history to price against."""
    session.add(
        Hotel(
            hotel_id="H001",
            hotel_name="Sanchay Grand Mumbai",
            city="Mumbai",
            star_rating=5,
            total_rooms=100,
            segment=MarketSegment.BUSINESS,
        )
    )
    session.flush()

    for index, (room_type, count, price) in enumerate(
        [(RoomType.STANDARD, 60, 5000.0), (RoomType.DELUXE, 40, 6400.0)]
    ):
        session.add(
            Room(
                room_id=f"H001-R{index}",
                hotel_id="H001",
                room_type=room_type,
                capacity=2,
                room_count=count,
                base_price=price,
                floor_price=price * 0.65,
                ceiling_price=price * 2.2,
            )
        )

    # The fact tables key on (hotel_id, room_type) through a composite foreign
    # key, and there is no ORM relationship telling SQLAlchemy about that
    # dependency -- so the rooms have to be on disk before the bookings are.
    session.flush()

    # Completed nights, so the historical fallback has something to average.
    for offset in range(1, 40):
        past = date.today() - timedelta(days=offset)
        session.add(
            Booking(
                hotel_id="H001",
                room_type=RoomType.DELUXE,
                booking_date=past - timedelta(days=7),
                check_in_date=past,
                check_out_date=past + timedelta(days=1),
                booking_count=26,
                cancellation_count=2,
                revenue=26 * 6200.0,
                adr=6200.0,
                lead_time_days=7,
            )
        )
        session.add(
            DemandFeature(
                hotel_id="H001",
                room_type=RoomType.DELUXE,
                stay_date=past,
                day_of_week=past.weekday(),
                is_weekend=past.weekday() >= 5,
                season=Season.MONSOON,
                holiday_flag=False,
                local_event_score=0.1,
                weather_score=0.6,
                search_demand=0.62,
                target_demand=0.60,
                occupancy_rate=0.55,
            )
        )

    # A competitive set for the night under test.
    for competitor, price in (
        (Competitor.BOOKING, 6800.0),
        (Competitor.EXPEDIA, 6500.0),
        (Competitor.AGODA, 6200.0),
    ):
        session.add(
            CompetitorPrice(
                hotel_id="H001",
                room_type=RoomType.DELUXE,
                competitor=competitor,
                check_in_date=STAY,
                price=price,
                collected_at=datetime.now(timezone.utc) - timedelta(days=1),
            )
        )
    session.commit()


class FakeProphetBundle:
    """Stands in for a loaded Prophet bundle."""

    def has(self, hotel_id, room_type) -> bool:
        return hotel_id == "H001"

    def demand_on(self, hotel_id, room_type, day):
        return {"forecast": 0.78, "lower": 0.70, "upper": 0.86, "trend": 0.75}

    def forecast_range(self, hotel_id, room_type, *, start, horizon_days=30):
        from models.prophet_model import ForecastResult

        frame = pd.DataFrame(
            {
                "ds": pd.date_range(start, periods=horizon_days, freq="D"),
                "yhat": 0.78,
                "yhat_lower": 0.70,
                "yhat_upper": 0.86,
                "trend": 0.75,
            }
        )
        return ForecastResult(
            series_key=(hotel_id, "deluxe"),
            frame=frame,
            horizon_days=horizon_days,
            fitted_at=datetime.now(timezone.utc),
        )


class FakeGBR:
    """Stands in for a loaded Gradient Boosting model."""

    model = object()  # the demand engine checks this to decide availability

    def predict_one(self, frame):
        return {"demand": 0.74, "lower": 0.66, "upper": 0.82, "confidence": 0.8}


@pytest.fixture
def api(settings, db_engine, tmp_path, monkeypatch):
    """A client wired to SQLite, with an empty artifact directory."""
    factory = sessionmaker(bind=db_engine, expire_on_commit=False, future=True)
    with factory() as seeding:
        seed_api_data(seeding)

    registry = ModelRegistry(settings)
    registry.artifact_dir = tmp_path / "artifacts"

    # No broker in the test environment. Publishing must degrade rather than
    # fail, and no route should care either way.
    monkeypatch.setattr(settings.kafka, "enabled", False)

    def override_session():
        # A generator *function*, not a lambda returning an iterator: FastAPI
        # decides how to treat a dependency by inspecting the callable, and a
        # lambda returning an iterator is injected as the iterator itself.
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app = create_app(settings)
    app.dependency_overrides[session_dependency] = override_session
    app.dependency_overrides[get_registry_dependency] = lambda: registry

    with TestClient(app) as client:
        client.registry = registry  # type: ignore[attr-defined]
        client.session_factory = factory  # type: ignore[attr-defined]
        yield client


def load_fakes(client) -> None:
    """Put fake models into the registry, as if a training run had happened."""
    client.registry._loaded = LoadedModels(
        version="v9",
        gbr=FakeGBR(),
        prophet=FakeProphetBundle(),
        loaded_at=datetime.now(timezone.utc),
    )


# --------------------------------------------------------------------------- #
# Hotels
# --------------------------------------------------------------------------- #


class TestHotelEndpoints:
    def test_list_hotels(self, api) -> None:
        body = api.get("/api/v1/hotels").json()
        assert [h["hotel_id"] for h in body] == ["H001"]
        assert body[0]["city"] == "Mumbai"

    def test_filter_by_city_is_case_insensitive(self, api) -> None:
        assert len(api.get("/api/v1/hotels", params={"city": "mumbai"}).json()) == 1
        assert api.get("/api/v1/hotels", params={"city": "Goa"}).json() == []

    def test_filter_by_star_rating(self, api) -> None:
        assert len(api.get("/api/v1/hotels", params={"star_rating": 5}).json()) == 1
        assert api.get("/api/v1/hotels", params={"star_rating": 3}).json() == []

    def test_hotel_detail_includes_rooms(self, api) -> None:
        body = api.get("/api/v1/hotels/H001").json()
        assert {r["room_type"] for r in body["rooms"]} == {"standard", "deluxe"}

    def test_hotel_detail_reports_recent_trading(self, api) -> None:
        """Occupancy, ADR and RevPAR -- the three numbers a hotel manages against."""
        body = api.get("/api/v1/hotels/H001").json()
        assert body["occupancy_last_30_days"] > 0
        assert body["adr_last_30_days"] > 0
        assert body["revpar_last_30_days"] == pytest.approx(
            body["occupancy_last_30_days"] * body["adr_last_30_days"], rel=0.01
        )

    def test_unknown_hotel_is_404(self, api) -> None:
        response = api.get("/api/v1/hotels/H999")
        assert response.status_code == 404
        assert "H999" in response.json()["detail"]


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #


class TestPricingEndpoint:
    def _body(self, **overrides) -> dict:
        payload = {
            "hotel_id": "H001",
            "room_type": "deluxe",
            "check_in_date": STAY.isoformat(),
            "current_price": 6000,
            "occupancy_rate": 0.72,
            "available_rooms": 11,
        }
        payload.update(overrides)
        return payload

    def test_returns_the_documented_shape(self, api) -> None:
        body = api.post("/api/v1/pricing/predict", json=self._body()).json()
        for field in (
            "hotel_id", "room_type", "check_in_date", "forecasted_demand",
            "predicted_demand", "base_price", "raw_recommended_price",
            "final_recommended_price", "price_change_percent", "competitor_rate",
            "confidence", "guardrails_applied", "model_version",
        ):
            assert field in body, field

    def test_prices_without_any_models(self, api) -> None:
        """The degraded path: no artifacts, so the historical fallback carries
        the request. It must still produce a guardrailed number."""
        body = api.post("/api/v1/pricing/predict", json=self._body()).json()
        assert body["final_recommended_price"] > 0
        assert body["demand"]["degraded"] is True
        assert body["model_version"] == "unversioned"

    def test_uses_the_models_when_they_are_loaded(self, api) -> None:
        load_fakes(api)
        body = api.post("/api/v1/pricing/predict", json=self._body()).json()
        assert body["forecasted_demand"] == pytest.approx(0.78)
        assert body["predicted_demand"] == pytest.approx(0.74)
        assert body["demand"]["degraded"] is False
        assert body["model_version"] == "v9"

    def test_only_the_three_required_fields_are_needed(self, api) -> None:
        """A caller who knows only hotel, room and date still gets a price."""
        response = api.post(
            "/api/v1/pricing/predict",
            json={"hotel_id": "H001", "room_type": "deluxe",
                  "check_in_date": STAY.isoformat()},
        )
        assert response.status_code == 200
        assert response.json()["final_recommended_price"] > 0

    def test_the_competitive_set_is_looked_up_when_omitted(self, api) -> None:
        body = api.post(
            "/api/v1/pricing/predict",
            json={"hotel_id": "H001", "room_type": "deluxe",
                  "check_in_date": STAY.isoformat()},
        ).json()
        # Seeded at 6800 / 6500 / 6200.
        assert body["competitor_rate"] == pytest.approx(6500.0, rel=0.01)

    def test_caller_values_beat_stored_ones(self, api) -> None:
        """The caller is describing the situation as it stands now, which is
        fresher than anything in the database."""
        body = api.post(
            "/api/v1/pricing/predict", json=self._body(competitor_rate=9000)
        ).json()
        assert body["competitor_rate"] == pytest.approx(9000.0)

    def test_the_response_explains_itself(self, api) -> None:
        body = api.post("/api/v1/pricing/predict", json=self._body()).json()
        assert len(body["adjustments"]) == 5
        assert all(a["reason"] for a in body["adjustments"])
        assert "Base price:" in body["explanation"]

    def test_the_price_respects_the_absolute_limits(self, api, settings) -> None:
        body = api.post(
            "/api/v1/pricing/predict", json=self._body(base_price=900_000)
        ).json()
        assert body["final_recommended_price"] <= settings.pricing.max_price

    def test_the_decision_is_persisted(self, api) -> None:
        body = api.post("/api/v1/pricing/predict", json=self._body()).json()
        with api.session_factory() as session:
            stored = session.query(PricingDecision).all()
        assert len(stored) == 1
        assert stored[0].final_recommended_price == pytest.approx(
            body["final_recommended_price"], rel=0.01
        )

    def test_persist_false_leaves_no_trace(self, api) -> None:
        """What-if queries must not pollute the audit trail."""
        api.post("/api/v1/pricing/predict", json=self._body(persist=False))
        with api.session_factory() as session:
            assert session.query(PricingDecision).count() == 0

    def test_unknown_hotel_is_404(self, api) -> None:
        response = api.post("/api/v1/pricing/predict", json=self._body(hotel_id="H999"))
        assert response.status_code == 404

    def test_a_room_the_hotel_does_not_sell_names_what_it_does(self, api) -> None:
        """'Unknown room type' is unhelpful; listing what it sells is not."""
        response = api.post("/api/v1/pricing/predict", json=self._body(room_type="suite"))
        assert response.status_code == 404
        assert "deluxe" in response.json()["detail"]

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("room_type", "penthouse"),
            ("occupancy_rate", 1.4),
            ("current_price", -100),
            ("check_in_date", "not-a-date"),
        ],
    )
    def test_invalid_input_is_422(self, api, field: str, value) -> None:
        response = api.post("/api/v1/pricing/predict", json=self._body(**{field: value}))
        assert response.status_code == 422

    def test_a_custom_validator_rejection_is_serialisable(self, api) -> None:
        """Regression: pydantic v2 puts the original exception object in
        ``ctx``, so echoing ``exc.errors()`` wholesale turned every 422 raised
        by one of our own field validators into a 500."""
        response = api.post(
            "/api/v1/pricing/predict", json=self._body(check_in_date="2099-01-01")
        )
        assert response.status_code == 422
        errors = response.json()["context"]["errors"]
        assert errors[0]["field"] == "check_in_date"
        assert "365 days" in errors[0]["message"]

    def test_the_error_envelope_does_not_echo_the_input(self, api) -> None:
        """A rejected payload should not end up in a log that was not designed
        to hold it."""
        response = api.post(
            "/api/v1/pricing/predict", json=self._body(current_price=-999_123)
        )
        assert "999123" not in response.text
        assert "-999123" not in response.text

    def test_an_inverted_competitor_band_is_rejected(self, api) -> None:
        response = api.post(
            "/api/v1/pricing/predict",
            json=self._body(competitor_min_rate=9000, competitor_max_rate=6000),
        )
        assert response.status_code == 422

    def test_history_returns_what_was_priced(self, api) -> None:
        api.post("/api/v1/pricing/predict", json=self._body())
        api.post("/api/v1/pricing/predict", json=self._body(current_price=6100))

        body = api.get("/api/v1/pricing/H001").json()
        assert body["count"] == 2
        assert all(item["hotel_id"] == "H001" for item in body["items"])
        assert all(item["prediction_id"] for item in body["items"])

    def test_history_can_be_filtered_by_room_type(self, api) -> None:
        api.post("/api/v1/pricing/predict", json=self._body())
        assert api.get(
            "/api/v1/pricing/H001", params={"room_type": "standard"}
        ).json()["count"] == 0

    def test_history_for_an_unknown_hotel_is_404(self, api) -> None:
        assert api.get("/api/v1/pricing/H999").status_code == 404


# --------------------------------------------------------------------------- #
# Forecasts
# --------------------------------------------------------------------------- #


class TestForecastEndpoint:
    def test_503_when_no_model_is_loaded(self, api) -> None:
        """A missing capability, not a fault -- and it resolves by training
        rather than by debugging."""
        response = api.get("/api/v1/forecast/H001")
        assert response.status_code == 503
        assert "train" in response.json()["detail"]

    def test_returns_the_requested_horizon(self, api) -> None:
        load_fakes(api)
        body = api.get(
            "/api/v1/forecast/H001", params={"room_type": "deluxe", "horizon_days": 7}
        ).json()
        assert len(body["points"]) == 7
        assert body["model_version"] == "v9"

    def test_the_forecast_starts_today_not_at_the_training_cutoff(self, api) -> None:
        """Regression: Prophet's own ``forecast()`` continues from the end of
        the *training* window, which the pipeline ends sixty days before today.
        'The next 7 nights' has to mean the next 7 nights."""
        load_fakes(api)
        body = api.get("/api/v1/forecast/H001", params={"horizon_days": 3}).json()
        assert body["points"][0]["date"] == date.today().isoformat()

    def test_an_explicit_start_date_is_honoured(self, api) -> None:
        load_fakes(api)
        body = api.get(
            "/api/v1/forecast/H001",
            params={"horizon_days": 2, "start_date": "2026-12-25"},
        ).json()
        assert [p["date"] for p in body["points"]] == ["2026-12-25", "2026-12-26"]

    def test_bounds_bracket_the_forecast(self, api) -> None:
        load_fakes(api)
        for point in api.get("/api/v1/forecast/H001").json()["points"]:
            assert point["lower"] <= point["forecast"] <= point["upper"]

    def test_an_absurd_horizon_is_422(self, api) -> None:
        assert api.get(
            "/api/v1/forecast/H001", params={"horizon_days": 5000}
        ).status_code == 422

    def test_a_room_the_hotel_does_not_sell_is_404(self, api) -> None:
        load_fakes(api)
        assert api.get(
            "/api/v1/forecast/H001", params={"room_type": "suite"}
        ).status_code == 404


# --------------------------------------------------------------------------- #
# Competitors
# --------------------------------------------------------------------------- #


class TestCompetitorEndpoints:
    def test_lists_observations_and_summarises_them(self, api) -> None:
        body = api.get(
            "/api/v1/competitors/H001",
            params={"start_date": STAY.isoformat(), "end_date": STAY.isoformat()},
        ).json()
        assert body["count"] == 3

        summary = body["summaries"][0]
        assert summary["competitor_count"] == 3
        assert summary["competitor_min_rate"] == 6200.0
        assert summary["competitor_max_rate"] == 6800.0
        assert summary["spread_percent"] > 0

    def test_unknown_hotel_is_404(self, api) -> None:
        assert api.get("/api/v1/competitors/H999").status_code == 404

    def test_submitting_an_event_persists_it(self, api) -> None:
        response = api.post(
            "/api/v1/competitors/events",
            json={
                "hotel_id": "H001",
                "competitor": "makemytrip",
                "room_type": "deluxe",
                "check_in_date": STAY.isoformat(),
                "price": 7100,
            },
        )
        assert response.status_code == 202
        assert response.json()["persisted"] is True

        with api.session_factory() as session:
            stored = (
                session.query(CompetitorPrice)
                .filter(CompetitorPrice.competitor == Competitor.MAKEMYTRIP)
                .one()
            )
        assert float(stored.price) == 7100.0

    def test_a_submitted_event_reaches_the_pricing_endpoint(self, api) -> None:
        """The whole point of the manual door: it feeds the same pipeline."""
        api.post(
            "/api/v1/competitors/events",
            json={
                "hotel_id": "H001",
                "competitor": "makemytrip",
                "room_type": "standard",
                "check_in_date": STAY.isoformat(),
                "price": 5500,
            },
        )
        body = api.post(
            "/api/v1/pricing/predict",
            json={"hotel_id": "H001", "room_type": "standard",
                  "check_in_date": STAY.isoformat()},
        ).json()
        assert body["competitor_rate"] == pytest.approx(5500.0)

    def test_kafka_being_down_does_not_lose_the_observation(self, api) -> None:
        """Kafka is disabled in this fixture. The rate is still stored, so a
        broker outage delays the stream rather than losing data."""
        response = api.post(
            "/api/v1/competitors/events",
            json={
                "hotel_id": "H001",
                "competitor": "agoda",
                "room_type": "standard",
                "check_in_date": STAY.isoformat(),
                "price": 5100,
            },
        ).json()
        assert response["published_to_kafka"] is False
        assert response["persisted"] is True
        assert "Kafka unavailable" in response["detail"]

    @pytest.mark.parametrize(
        ("field", "value"),
        [("price", -50), ("price", 0), ("competitor", "some_ota"), ("currency", "RUPEE")],
    )
    def test_invalid_events_are_422(self, api, field: str, value) -> None:
        payload = {
            "hotel_id": "H001",
            "competitor": "booking",
            "room_type": "deluxe",
            "check_in_date": STAY.isoformat(),
            "price": 6000,
        }
        payload[field] = value
        assert api.post("/api/v1/competitors/events", json=payload).status_code == 422

    def test_an_event_for_an_unknown_hotel_is_404(self, api) -> None:
        assert api.post(
            "/api/v1/competitors/events",
            json={
                "hotel_id": "H999",
                "competitor": "booking",
                "room_type": "deluxe",
                "check_in_date": STAY.isoformat(),
                "price": 6000,
            },
        ).status_code == 404


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class TestModelEndpoints:
    def test_reports_no_models_cleanly(self, api) -> None:
        body = api.get("/api/v1/models").json()
        assert body["active_version"] is None
        assert body["available"] == []
        assert body["versions"] == []
        assert body["feature_version"]

    def test_reports_the_loaded_version(self, api) -> None:
        load_fakes(api)
        body = api.get("/api/v1/models").json()
        assert body["active_version"] == "v9"
        assert set(body["available"]) == {"gradient_boosting", "prophet"}

    def test_training_without_data_is_409_not_500(self, api) -> None:
        """The request is well-formed; the *state* is wrong, and the fix is to
        seed and build features rather than to debug."""
        response = api.post("/api/v1/models/train", json={"test_days": 30})
        assert response.status_code == 409

    def test_train_request_bounds_are_enforced(self, api) -> None:
        assert api.post(
            "/api/v1/models/train", json={"test_days": 5000}
        ).status_code == 422


# --------------------------------------------------------------------------- #
# The documented contract
# --------------------------------------------------------------------------- #


class TestOpenApiContract:
    def test_every_endpoint_is_published_and_no_others(self, api) -> None:
        """The exact published surface -- no more and no fewer.

        A set, not a count, so that adding one endpoint while removing another
        cannot cancel out. The ten the original specification names, plus the
        two ingestion endpoints that came with the scraping pipeline.
        """
        paths = api.get("/openapi.json").json()["paths"]
        expected = {
            "/health",
            "/api/v1/hotels",
            "/api/v1/hotels/{hotel_id}",
            "/api/v1/pricing/predict",
            "/api/v1/pricing/{hotel_id}",
            "/api/v1/forecast/{hotel_id}",
            "/api/v1/competitors/{hotel_id}",
            "/api/v1/competitors/events",
            "/api/v1/ingestion/status",
            "/api/v1/ingestion/run",
            "/api/v1/models",
            "/api/v1/models/train",
        }
        assert expected == set(paths)

    def test_the_pricing_request_carries_a_worked_example(self, api) -> None:
        """A Swagger page whose example is `{}` is documentation in name only."""
        schema = api.get("/openapi.json").json()["components"]["schemas"]
        assert "example" in schema["PricingRequestSchema"]

    def test_error_responses_are_documented(self, api) -> None:
        paths = api.get("/openapi.json").json()["paths"]
        assert "404" in paths["/api/v1/hotels/{hotel_id}"]["get"]["responses"]
        assert "503" in paths["/api/v1/forecast/{hotel_id}"]["get"]["responses"]
