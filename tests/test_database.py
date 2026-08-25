"""Tests for the persistence layer.

Two things are being proven here, and they are different:

1. **The schema exists and is shaped correctly** -- nine tables, the right keys,
   the right indexes.
2. **The database refuses bad data.** Every business invariant that is expressed
   as a CHECK or a FOREIGN KEY gets a test that tries to violate it. A
   constraint nobody has ever seen fire is a constraint that might not work.

These run on in-memory SQLite with ``PRAGMA foreign_keys=ON``, so composite
foreign keys are genuinely enforced rather than quietly ignored.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.connection import create_db_engine, database_info, ping, session_scope
from database.init_db import create_schema, drop_schema, schema_status
from database.models import (
    TABLE_NAMES,
    Booking,
    Competitor,
    CompetitorPrice,
    DemandFeature,
    Hotel,
    MarketSegment,
    ModelType,
    ModelVersion,
    Prediction,
    PricingDecision,
    Room,
    RoomType,
    RunStatus,
    Season,
    TrainingRun,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

STAY = date(2026, 9, 15)


def _booking(**overrides) -> Booking:
    """A valid booking row, with named fields overridable per test."""
    row = dict(
        hotel_id="H001",
        room_type=RoomType.DELUXE,
        booking_date=STAY - timedelta(days=10),
        check_in_date=STAY,
        check_out_date=STAY + timedelta(days=1),
        booking_count=8,
        cancellation_count=1,
        revenue=44_800.0,
        adr=6_400.0,
        lead_time_days=10,
        channel="ota",
    )
    row.update(overrides)
    return Booking(**row)


def _competitor_price(**overrides) -> CompetitorPrice:
    row = dict(
        hotel_id="H001",
        room_type=RoomType.DELUXE,
        competitor=Competitor.BOOKING,
        check_in_date=STAY,
        price=6_800.0,
        currency="INR",
        collected_at=datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc),
    )
    row.update(overrides)
    return CompetitorPrice(**row)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #


class TestSchema:
    def test_creates_all_nine_tables(self, db_engine) -> None:
        assert sorted(inspect(db_engine).get_table_names()) == sorted(TABLE_NAMES)
        assert len(TABLE_NAMES) == 9

    def test_create_schema_is_idempotent(self, db_engine) -> None:
        assert create_schema(db_engine) == []
        assert schema_status(db_engine)["missing"] == []

    def test_drop_then_status_reports_everything_missing(self, db_engine) -> None:
        drop_schema(db_engine)
        status = schema_status(db_engine)
        assert status["present"] == []
        assert sorted(status["missing"]) == sorted(TABLE_NAMES)

    @pytest.mark.parametrize(
        ("table", "expected"),
        [
            ("bookings", "ix_bookings_hotel_checkin"),
            ("competitor_prices", "ix_competitor_prices_lookup"),
            ("demand_features", "ix_demand_features_hotel_stay"),
            ("predictions", "ix_predictions_hotel_created"),
            ("model_versions", "ix_model_versions_one_active"),
        ],
    )
    def test_query_path_indexes_exist(self, db_engine, table: str, expected: str) -> None:
        """The indexes the hot queries depend on are actually created."""
        names = {i["name"] for i in inspect(db_engine).get_indexes(table)}
        assert expected in names

    def test_enums_are_stored_as_portable_varchar(self, db_engine) -> None:
        """Not a native PostgreSQL ENUM -- see the module docstring in models.py."""
        columns = {c["name"]: c for c in inspect(db_engine).get_columns("rooms")}
        assert "VARCHAR" in str(columns["room_type"]["type"]).upper()


# --------------------------------------------------------------------------- #
# Reference data
# --------------------------------------------------------------------------- #


class TestHotelsAndRooms:
    def test_seeded_hotel_has_four_room_types(self, seeded_session: Session) -> None:
        hotel = seeded_session.execute(
            select(Hotel).where(Hotel.hotel_id == "H001")
        ).scalar_one()
        assert {r.room_type for r in hotel.rooms} == set(RoomType)

    def test_room_counts_sum_to_hotel_inventory(self, seeded_session: Session) -> None:
        hotel = seeded_session.execute(select(Hotel)).scalar_one()
        assert sum(r.room_count for r in hotel.rooms) == hotel.total_rooms

    def test_duplicate_hotel_id_rejected(self, seeded_session: Session) -> None:
        seeded_session.add(
            Hotel(
                hotel_id="H001",
                hotel_name="Impostor",
                city="Goa",
                star_rating=3,
                total_rooms=10,
                segment=MarketSegment.LEISURE,
            )
        )
        with pytest.raises(IntegrityError):
            seeded_session.flush()

    def test_duplicate_room_type_per_hotel_rejected(self, seeded_session: Session) -> None:
        """``(hotel_id, room_type)`` is the grain every fact table keys on."""
        seeded_session.add(
            Room(
                room_id="H001-DUP",
                hotel_id="H001",
                room_type=RoomType.DELUXE,
                capacity=2,
                room_count=5,
                base_price=6000.0,
            )
        )
        with pytest.raises(IntegrityError):
            seeded_session.flush()

    @pytest.mark.parametrize(
        ("field", "value"),
        [("star_rating", 9), ("star_rating", 0), ("total_rooms", 0)],
    )
    def test_hotel_check_constraints(self, db_session: Session, field: str, value) -> None:
        row = dict(
            hotel_id="H900",
            hotel_name="Bad",
            city="Goa",
            star_rating=4,
            total_rooms=10,
            segment=MarketSegment.LEISURE,
        )
        row[field] = value
        db_session.add(Hotel(**row))
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_deleting_a_hotel_cascades_to_its_rooms(self, seeded_session: Session) -> None:
        hotel = seeded_session.execute(select(Hotel)).scalar_one()
        seeded_session.delete(hotel)
        seeded_session.commit()
        assert seeded_session.execute(select(Room)).scalars().all() == []


# --------------------------------------------------------------------------- #
# Fact tables
# --------------------------------------------------------------------------- #


class TestBookingConstraints:
    def test_valid_booking_accepted(self, seeded_session: Session) -> None:
        seeded_session.add(_booking())
        seeded_session.commit()
        assert seeded_session.execute(select(Booking)).scalar_one().booking_count == 8

    def test_unknown_hotel_rejected_by_composite_fk(self, seeded_session: Session) -> None:
        seeded_session.add(_booking(hotel_id="H999"))
        with pytest.raises(IntegrityError):
            seeded_session.flush()

    def test_room_type_the_hotel_does_not_sell_is_rejected(
        self, db_session: Session
    ) -> None:
        """A hotel with only standard rooms cannot take a suite booking."""
        db_session.add(
            Hotel(
                hotel_id="H010",
                hotel_name="Standard Only",
                city="Goa",
                star_rating=3,
                total_rooms=20,
                segment=MarketSegment.LEISURE,
            )
        )
        db_session.add(
            Room(
                room_id="H010-STD",
                hotel_id="H010",
                room_type=RoomType.STANDARD,
                capacity=2,
                room_count=20,
                base_price=3000.0,
            )
        )
        db_session.commit()

        db_session.add(_booking(hotel_id="H010", room_type=RoomType.SUITE))
        with pytest.raises(IntegrityError):
            db_session.flush()

    @pytest.mark.parametrize(
        ("label", "patch"),
        [
            ("negative bookings", {"booking_count": -1}),
            ("negative cancellations", {"cancellation_count": -1}),
            ("cancellations exceed bookings", {"booking_count": 2, "cancellation_count": 3}),
            ("negative revenue", {"revenue": -1.0}),
            ("zero-length stay", {"check_out_date": STAY}),
            ("checkout before checkin", {"check_out_date": STAY - timedelta(days=1)}),
            ("booked after the stay", {"booking_date": STAY + timedelta(days=1)}),
            ("negative lead time", {"lead_time_days": -3}),
        ],
    )
    def test_invalid_bookings_rejected(
        self, seeded_session: Session, label: str, patch: dict
    ) -> None:
        seeded_session.add(_booking(**patch))
        with pytest.raises(IntegrityError, match=r"(?i)constraint"):
            seeded_session.flush()


class TestCompetitorPriceConstraints:
    def test_valid_row_accepted(self, seeded_session: Session) -> None:
        seeded_session.add(_competitor_price())
        seeded_session.commit()
        stored = seeded_session.execute(select(CompetitorPrice)).scalar_one()
        assert stored.competitor is Competitor.BOOKING
        assert stored.is_available is True

    @pytest.mark.parametrize("price", [0.0, -100.0])
    def test_non_positive_price_rejected(self, seeded_session: Session, price: float) -> None:
        seeded_session.add(_competitor_price(price=price))
        with pytest.raises(IntegrityError):
            seeded_session.flush()

    def test_duplicate_event_id_rejected(self, seeded_session: Session) -> None:
        """At-least-once delivery means the same event can arrive twice.

        The unique index is what turns a redelivery into a rejected insert
        instead of a duplicated observation skewing the competitor average.
        """
        seeded_session.add(_competitor_price(event_id="evt-1"))
        seeded_session.commit()
        seeded_session.add(_competitor_price(event_id="evt-1", price=7000.0))
        with pytest.raises(IntegrityError):
            seeded_session.flush()

    def test_null_event_ids_do_not_collide(self, seeded_session: Session) -> None:
        """Batch-seeded rows carry no event id; NULLs must not be unique-equal."""
        seeded_session.add(_competitor_price())
        seeded_session.add(_competitor_price(competitor=Competitor.AGODA))
        seeded_session.commit()
        assert len(seeded_session.execute(select(CompetitorPrice)).scalars().all()) == 2

    def test_all_four_competitors_are_storable(self, seeded_session: Session) -> None:
        for competitor in Competitor:
            seeded_session.add(_competitor_price(competitor=competitor))
        seeded_session.commit()
        stored = seeded_session.execute(select(CompetitorPrice.competitor)).scalars().all()
        assert set(stored) == set(Competitor)


class TestDemandFeatureConstraints:
    def _feature(self, **overrides) -> DemandFeature:
        row = dict(
            hotel_id="H001",
            room_type=RoomType.DELUXE,
            stay_date=STAY,
            day_of_week=STAY.weekday(),
            is_weekend=False,
            season=Season.AUTUMN,
            holiday_flag=False,
            local_event_score=0.0,
            weather_score=0.6,
            search_demand=0.5,
        )
        row.update(overrides)
        return DemandFeature(**row)

    def test_grain_is_unique(self, seeded_session: Session) -> None:
        seeded_session.add(self._feature())
        seeded_session.commit()
        seeded_session.add(self._feature(search_demand=0.9))
        with pytest.raises(IntegrityError):
            seeded_session.flush()

    def test_derived_columns_may_be_null_before_phase_four_runs(
        self, seeded_session: Session
    ) -> None:
        seeded_session.add(self._feature())
        seeded_session.commit()
        row = seeded_session.execute(select(DemandFeature)).scalar_one()
        assert row.occupancy_rate is None
        assert row.target_demand is None
        assert row.search_demand == 0.5

    @pytest.mark.parametrize(
        "patch",
        [
            {"occupancy_rate": 1.4},
            {"occupancy_rate": -0.1},
            {"day_of_week": 7},
            {"target_demand": -0.5},
            {"available_rooms": -2},
        ],
    )
    def test_out_of_range_values_rejected(self, seeded_session: Session, patch: dict) -> None:
        seeded_session.add(self._feature(**patch))
        with pytest.raises(IntegrityError):
            seeded_session.flush()


# --------------------------------------------------------------------------- #
# Serving artefacts
# --------------------------------------------------------------------------- #


class TestPredictionAndDecision:
    def _prediction(self, **overrides) -> Prediction:
        row = dict(
            prediction_id="pred-1",
            hotel_id="H001",
            room_type=RoomType.DELUXE,
            check_in_date=STAY,
            forecasted_demand=0.82,
            predicted_demand=0.79,
            blended_demand=0.805,
            confidence=0.87,
            model_version="v1.0",
            features={"occupancy_rate": 0.72},
        )
        row.update(overrides)
        return Prediction(**row)

    def test_prediction_and_decision_are_one_to_one(self, seeded_session: Session) -> None:
        prediction = self._prediction()
        seeded_session.add(prediction)
        seeded_session.flush()

        seeded_session.add(
            PricingDecision(
                prediction_id=prediction.id,
                hotel_id="H001",
                room_type=RoomType.DELUXE,
                check_in_date=STAY,
                base_price=6400.0,
                current_price=6000.0,
                demand_adjustment=0.12,
                occupancy_adjustment=0.08,
                competitor_adjustment=0.05,
                season_adjustment=0.02,
                event_adjustment=0.10,
                total_adjustment=0.37,
                raw_recommended_price=8768.0,
                final_recommended_price=6900.0,
                price_change_percent=15.0,
                guardrails_applied=["max_daily_change"],
                breakdown={"steps": ["base", "demand"]},
            )
        )
        seeded_session.commit()

        stored = seeded_session.execute(select(Prediction)).scalar_one()
        assert stored.decision is not None
        assert stored.decision.guardrails_applied == ["max_daily_change"]
        assert stored.features == {"occupancy_rate": 0.72}

    def test_confidence_outside_zero_one_rejected(self, seeded_session: Session) -> None:
        seeded_session.add(self._prediction(confidence=1.4))
        with pytest.raises(IntegrityError):
            seeded_session.flush()

    def test_duplicate_prediction_id_rejected(self, seeded_session: Session) -> None:
        seeded_session.add(self._prediction())
        seeded_session.commit()
        seeded_session.add(self._prediction())
        with pytest.raises(IntegrityError):
            seeded_session.flush()

    def test_deleting_a_prediction_removes_its_decision(
        self, seeded_session: Session
    ) -> None:
        prediction = self._prediction()
        seeded_session.add(prediction)
        seeded_session.flush()
        seeded_session.add(
            PricingDecision(
                prediction_id=prediction.id,
                hotel_id="H001",
                room_type=RoomType.DELUXE,
                check_in_date=STAY,
                base_price=6400.0,
                raw_recommended_price=7000.0,
                final_recommended_price=6900.0,
            )
        )
        seeded_session.commit()

        seeded_session.delete(prediction)
        seeded_session.commit()
        assert seeded_session.execute(select(PricingDecision)).scalars().all() == []


# --------------------------------------------------------------------------- #
# MLOps tables
# --------------------------------------------------------------------------- #


class TestModelRegistryTables:
    def _version(self, **overrides) -> ModelVersion:
        row = dict(
            version="v1.0",
            model_type=ModelType.GRADIENT_BOOSTING,
            artifact_path="models/artifacts/gbr_v1.joblib",
            is_active=True,
            mae=0.041,
            rmse=0.058,
            mape=6.2,
            feature_list=["occupancy_rate", "competitor_rate"],
            hyperparameters={"n_estimators": 300},
            training_rows=9_000,
            trained_at=datetime(2026, 8, 24, 6, 0, tzinfo=timezone.utc),
        )
        row.update(overrides)
        return ModelVersion(**row)

    def test_only_one_active_version_per_model_type(self, db_session: Session) -> None:
        """The partial unique index is the guarantee that "the active model" is
        a single, unambiguous row."""
        db_session.add(self._version(version="v1.0"))
        db_session.commit()
        db_session.add(self._version(version="v2.0"))
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_many_inactive_versions_allowed(self, db_session: Session) -> None:
        db_session.add(self._version(version="v1.0", is_active=False))
        db_session.add(self._version(version="v2.0", is_active=False))
        db_session.add(self._version(version="v3.0", is_active=True))
        db_session.commit()
        assert len(db_session.execute(select(ModelVersion)).scalars().all()) == 3

    def test_each_model_type_has_its_own_active_slot(self, db_session: Session) -> None:
        db_session.add(self._version(model_type=ModelType.GRADIENT_BOOSTING))
        db_session.add(self._version(model_type=ModelType.PROPHET))
        db_session.commit()
        active = db_session.execute(
            select(ModelVersion).where(ModelVersion.is_active.is_(True))
        ).scalars().all()
        assert {v.model_type for v in active} == set(ModelType)

    def test_duplicate_version_string_per_type_rejected(self, db_session: Session) -> None:
        db_session.add(self._version(is_active=False))
        db_session.commit()
        db_session.add(self._version(is_active=False))
        with pytest.raises(IntegrityError):
            db_session.flush()

    def test_failed_runs_are_recorded_not_discarded(self, db_session: Session) -> None:
        db_session.add(
            TrainingRun(
                run_id="run-1",
                model_type=ModelType.PROPHET,
                status=RunStatus.FAILED,
                error_message="insufficient history: 12 rows",
                triggered_by="cli",
            )
        )
        db_session.commit()
        run = db_session.execute(select(TrainingRun)).scalar_one()
        assert run.status is RunStatus.FAILED
        assert run.model_version_id is None
        assert "insufficient history" in run.error_message

    def test_run_links_to_its_model_version(self, db_session: Session) -> None:
        version = self._version()
        db_session.add(version)
        db_session.flush()
        db_session.add(
            TrainingRun(
                run_id="run-2",
                model_type=ModelType.GRADIENT_BOOSTING,
                status=RunStatus.SUCCEEDED,
                model_version_id=version.id,
                metrics={"mae": 0.041},
            )
        )
        db_session.commit()
        assert db_session.execute(select(ModelVersion)).scalar_one().runs[0].run_id == "run-2"


# --------------------------------------------------------------------------- #
# Connection layer
# --------------------------------------------------------------------------- #


class TestConnection:
    def test_session_scope_commits_on_success(self, db_engine) -> None:
        from sqlalchemy.orm import sessionmaker

        factory = sessionmaker(bind=db_engine, expire_on_commit=False, future=True)
        with session_scope(factory) as session:
            session.add(
                Hotel(
                    hotel_id="H100",
                    hotel_name="Committed",
                    city="Goa",
                    star_rating=4,
                    total_rooms=50,
                    segment=MarketSegment.LEISURE,
                )
            )

        with session_scope(factory) as session:
            assert session.execute(select(Hotel)).scalar_one().hotel_id == "H100"

    def test_session_scope_rolls_back_on_error(self, db_engine) -> None:
        from sqlalchemy.orm import sessionmaker

        factory = sessionmaker(bind=db_engine, expire_on_commit=False, future=True)
        with pytest.raises(RuntimeError):
            with session_scope(factory) as session:
                session.add(
                    Hotel(
                        hotel_id="H101",
                        hotel_name="Doomed",
                        city="Goa",
                        star_rating=4,
                        total_rooms=50,
                        segment=MarketSegment.LEISURE,
                    )
                )
                session.flush()
                raise RuntimeError("business rule failed")

        with session_scope(factory) as session:
            assert session.execute(select(Hotel)).scalars().all() == []

    def test_sqlite_foreign_keys_are_enabled(self, db_engine) -> None:
        """Without the PRAGMA, every FK test in this file would pass vacuously."""
        with db_engine.connect() as conn:
            from sqlalchemy import text

            assert conn.execute(text("PRAGMA foreign_keys")).scalar_one() == 1

    def test_ping_and_info_report_a_live_database(self, db_engine) -> None:
        assert ping(db_engine) is True
        info = database_info(db_engine)
        assert info["reachable"] is True
        assert info["dialect"] == "sqlite"

    def test_ping_returns_false_for_an_unreachable_database(self) -> None:
        """Health checks must degrade to False, never raise."""
        engine = create_db_engine("sqlite+pysqlite:////nonexistent-dir/x/y.db")
        assert ping(engine) is False


# --------------------------------------------------------------------------- #
# Timezone discipline
# --------------------------------------------------------------------------- #


def test_timestamp_columns_are_timezone_aware() -> None:
    """Requirement: every timestamp is TIMESTAMPTZ, stored UTC.

    Asserted against the model metadata rather than a live row, because SQLite
    has no timestamp type at all -- the property that matters is what the DDL
    says, since that is what PostgreSQL will build.
    """
    from database.models import ALL_TABLES

    offenders = [
        f"{table.name}.{column.name}"
        for table in ALL_TABLES
        for column in table.columns
        if column.type.__class__.__name__ == "TIMESTAMP"
        and not getattr(column.type, "timezone", False)
    ]
    assert offenders == []
