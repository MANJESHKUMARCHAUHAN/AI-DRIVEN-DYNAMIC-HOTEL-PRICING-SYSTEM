"""Tests for the monitoring layer and the model registry.

Monitoring is the code most likely to be written once and never exercised,
because it only speaks up when something is wrong -- and the thing it warns
about is by definition not happening in a healthy test environment. So the tests
here mostly *construct* the unhealthy states: a stale competitor feed, a
collapsed prediction distribution, a feature store with the wrong version in it.

The PSI tests are the exception. They check the arithmetic directly, because a
drift index that quietly returns zero for everything is indistinguishable from a
system with no drift.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest
from sqlalchemy.orm import Session

from database.models import (
    Booking,
    Competitor,
    CompetitorPrice,
    DemandFeature,
    Prediction,
    PricingDecision,
    RoomType,
    Season,
)
from features.feature_engineering import FEATURE_VERSION
from models.model_registry import ModelRegistry, reset_registry
from monitoring.data_monitor import (
    MIN_COMPETITOR_COVERAGE,
    DataMonitor,
    DataQualityReport,
    Severity,
)
from monitoring.model_monitor import (
    PSI_MODERATE,
    PSI_NO_SHIFT,
    ModelMonitor,
    population_stability_index,
    psi_severity,
)

TODAY = date.today()


# --------------------------------------------------------------------------- #
# PSI
# --------------------------------------------------------------------------- #


class TestPopulationStabilityIndex:
    def test_identical_distributions_score_near_zero(self) -> None:
        rng = np.random.default_rng(0)
        sample = rng.normal(0.6, 0.15, 2_000)
        assert population_stability_index(sample, sample) < 0.01

    def test_a_shifted_distribution_scores_high(self) -> None:
        rng = np.random.default_rng(0)
        reference = rng.normal(0.60, 0.15, 2_000)
        shifted = rng.normal(0.85, 0.15, 2_000)
        assert population_stability_index(reference, shifted) > PSI_MODERATE

    def test_psi_grows_with_the_size_of_the_shift(self) -> None:
        rng = np.random.default_rng(1)
        reference = rng.normal(0.60, 0.15, 3_000)
        values = [
            population_stability_index(reference, rng.normal(mean, 0.15, 3_000))
            for mean in (0.62, 0.70, 0.85)
        ]
        assert values == sorted(values)

    def test_a_vanished_region_is_strong_drift(self) -> None:
        """Empty bins are floored, not dropped. Dropping them would make a
        category disappearing entirely *reduce* the index."""
        rng = np.random.default_rng(2)
        reference = rng.uniform(0.0, 1.0, 2_000)
        truncated = rng.uniform(0.0, 0.4, 2_000)
        assert population_stability_index(reference, truncated) > PSI_MODERATE

    def test_too_small_a_sample_is_nan_not_zero(self) -> None:
        """Returning 0.0 would read as 'no drift' when the truth is 'no idea'."""
        assert np.isnan(population_stability_index([0.5] * 10, [0.9] * 10))

    def test_a_constant_reference_is_undefined(self) -> None:
        assert np.isnan(population_stability_index([0.5] * 500, [0.9] * 500))

    def test_non_finite_values_are_dropped(self) -> None:
        rng = np.random.default_rng(3)
        clean = rng.normal(0.6, 0.1, 500)
        dirty = np.concatenate([clean, [np.nan, np.inf, -np.inf]])
        assert population_stability_index(clean, dirty) == pytest.approx(
            population_stability_index(clean, clean), abs=0.02
        )

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.05, Severity.OK),
            (PSI_NO_SHIFT, Severity.WARNING),
            (0.20, Severity.WARNING),
            (PSI_MODERATE, Severity.CRITICAL),
            (1.5, Severity.CRITICAL),
            (float("nan"), Severity.OK),
        ],
    )
    def test_severity_bands(self, value: float, expected: Severity) -> None:
        assert psi_severity(value) is expected


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #


def seed_healthy(session: Session, *, days: int = 60) -> None:
    """A database that should pass every data quality check."""
    now = datetime.now(timezone.utc)
    for offset in range(days):
        stay = TODAY - timedelta(days=offset)
        session.add(
            Booking(
                hotel_id="H001",
                room_type=RoomType.DELUXE,
                booking_date=stay - timedelta(days=7),
                check_in_date=stay,
                check_out_date=stay + timedelta(days=1),
                booking_count=20,
                cancellation_count=2,
                revenue=20 * 6000.0,
                adr=6000.0,
                lead_time_days=7,
            )
        )
        session.add(
            DemandFeature(
                hotel_id="H001",
                room_type=RoomType.DELUXE,
                stay_date=stay,
                day_of_week=stay.weekday(),
                is_weekend=stay.weekday() >= 5,
                season=Season.MONSOON,
                holiday_flag=False,
                local_event_score=0.1,
                weather_score=0.6,
                search_demand=0.6,
                occupancy_rate=0.55,
                competitor_rate=6400.0,
                historical_demand=0.6,
                current_room_price=6200.0,
                lead_time=7.0,
                booking_count=18,
                target_demand=0.6,
                feature_version=FEATURE_VERSION,
                computed_at=now,
            )
        )

    # Competitor rates covering the whole forward horizon.
    for offset in range(0, 31):
        session.add(
            CompetitorPrice(
                hotel_id="H001",
                room_type=RoomType.DELUXE,
                competitor=Competitor.BOOKING,
                check_in_date=TODAY + timedelta(days=offset),
                price=6500.0,
                collected_at=now - timedelta(hours=1),
            )
        )
    session.commit()


@pytest.fixture
def healthy(seeded_session: Session) -> Session:
    seed_healthy(seeded_session)
    return seeded_session


class TestDataMonitorHappyPath:
    def test_a_healthy_database_passes_everything(self, healthy) -> None:
        report = DataMonitor().run(healthy)
        assert report.severity is Severity.OK, [c.message for c in report.failures]

    def test_the_report_counts_its_checks(self, healthy) -> None:
        report = DataMonitor().run(healthy)
        assert report.as_dict()["passed"] == report.as_dict()["total"]
        assert "checks passed" in report.summary()


class TestDataMonitorFindsProblems:
    def test_an_empty_database_is_critical(self, db_session) -> None:
        report = DataMonitor().run(db_session)
        assert report.severity is Severity.CRITICAL

    def test_a_hotel_without_rooms_cannot_be_priced(self, db_session) -> None:
        from database.models import Hotel, MarketSegment

        db_session.add(
            Hotel(
                hotel_id="H500",
                hotel_name="Roomless",
                city="Goa",
                star_rating=3,
                total_rooms=10,
                segment=MarketSegment.LEISURE,
            )
        )
        db_session.commit()

        check = DataMonitor.check_reference_data(db_session)
        assert check.severity is Severity.CRITICAL
        assert "no rooms" in check.message

    def test_a_stale_competitor_feed_warns(self, healthy) -> None:
        """The check most likely to fire in practice: a scraper gets blocked and
        the engine goes blind to the market while everything else keeps working."""
        healthy.query(CompetitorPrice).update(
            {CompetitorPrice.collected_at: datetime.now(timezone.utc) - timedelta(days=5)}
        )
        healthy.commit()

        check = DataMonitor().check_competitor_freshness(healthy)
        assert check.severity is Severity.WARNING
        assert "stale" in check.message

    def test_no_competitor_data_at_all_is_critical(self, healthy) -> None:
        healthy.query(CompetitorPrice).delete()
        healthy.commit()

        check = DataMonitor().check_competitor_freshness(healthy)
        assert check.severity is Severity.CRITICAL

    def test_thin_forward_coverage_warns(self, healthy) -> None:
        healthy.query(CompetitorPrice).filter(
            CompetitorPrice.check_in_date > TODAY + timedelta(days=3)
        ).delete()
        healthy.commit()

        check = DataMonitor.check_competitor_coverage(healthy)
        assert check.severity is Severity.WARNING
        assert check.value < MIN_COMPETITOR_COVERAGE

    def test_coverage_can_never_exceed_one(self, healthy) -> None:
        """Regression: an earlier version divided by the feature-store night
        count, which stops at today while competitor rates run forward, and
        reported 200% coverage."""
        for offset in range(200):
            healthy.add(
                CompetitorPrice(
                    hotel_id="H001",
                    room_type=RoomType.DELUXE,
                    competitor=Competitor.EXPEDIA,
                    check_in_date=TODAY + timedelta(days=offset),
                    price=6000.0,
                    collected_at=datetime.now(timezone.utc),
                )
            )
        healthy.commit()

        assert DataMonitor.check_competitor_coverage(healthy).value <= 1.0

    def test_a_stale_feature_build_warns(self, healthy) -> None:
        healthy.query(DemandFeature).update(
            {DemandFeature.computed_at: datetime.now(timezone.utc) - timedelta(days=4)}
        )
        healthy.commit()

        check = DataMonitor.check_feature_freshness(healthy)
        assert check.severity is Severity.WARNING

    def test_a_mixed_feature_version_warns(self, healthy) -> None:
        """Serving v1 features to code that now produces v2 is train/serve skew
        that no accuracy metric will reveal."""
        row = healthy.query(DemandFeature).first()
        row.feature_version = "v0"
        healthy.commit()

        check = DataMonitor.check_feature_version(healthy)
        assert check.severity is Severity.WARNING
        assert "v0" in check.message

    def test_nulls_in_a_required_column_are_reported(self, healthy) -> None:
        rows = healthy.query(DemandFeature).limit(30).all()
        for row in rows:
            row.occupancy_rate = None
        healthy.commit()

        check = DataMonitor.check_feature_nulls(healthy)
        assert check.severity in {Severity.WARNING, Severity.CRITICAL}
        assert "occupancy_rate" in check.message

    def test_an_impossible_target_is_critical(self, healthy) -> None:
        """Overbooking makes values above 1.0 legitimate; 3.0 means the
        inventory denominator is wrong."""
        row = healthy.query(DemandFeature).first()
        row.target_demand = 3.0
        healthy.commit()

        check = DataMonitor.check_target_range(healthy)
        assert check.severity is Severity.CRITICAL
        assert "denominator" in check.message

    def test_a_gap_in_bookings_warns(self, seeded_session) -> None:
        seed_healthy(seeded_session)
        seeded_session.query(Booking).filter(
            Booking.check_in_date > TODAY - timedelta(days=20)
        ).delete()
        seeded_session.commit()

        check = DataMonitor.check_booking_recency(seeded_session)
        assert check.severity is Severity.WARNING


class TestReportSeverity:
    def test_worst_severity_wins(self) -> None:
        from monitoring.data_monitor import CheckResult

        report = DataQualityReport(
            checks=[
                CheckResult("a", Severity.OK, ""),
                CheckResult("b", Severity.WARNING, ""),
                CheckResult("c", Severity.CRITICAL, ""),
            ]
        )
        assert report.severity is Severity.CRITICAL
        assert len(report.failures) == 2

    def test_all_ok_is_ok(self) -> None:
        from monitoring.data_monitor import CheckResult

        report = DataQualityReport(checks=[CheckResult("a", Severity.OK, "")])
        assert report.severity is Severity.OK


# --------------------------------------------------------------------------- #
# Model monitoring
# --------------------------------------------------------------------------- #


def seed_predictions(session: Session, *, n: int = 120, constant: bool = False) -> None:
    rng = np.random.default_rng(4)
    for index in range(n):
        demand = 0.7 if constant else float(np.clip(rng.normal(0.65, 0.12), 0.05, 1.0))
        prediction = Prediction(
            prediction_id=f"pred-{index}",
            hotel_id="H001",
            room_type=RoomType.DELUXE,
            check_in_date=TODAY - timedelta(days=index % 30),
            blended_demand=demand,
            confidence=0.8,
            model_version="v1",
            latency_ms=25.0,
        )
        session.add(prediction)
        session.flush()
        session.add(
            PricingDecision(
                prediction_id=prediction.id,
                hotel_id="H001",
                room_type=RoomType.DELUXE,
                check_in_date=prediction.check_in_date,
                base_price=6400.0,
                raw_recommended_price=6400.0 * (1 + demand - 0.65),
                final_recommended_price=6400.0 * (1 + demand - 0.65),
                price_change_percent=(demand - 0.65) * 100,
                guardrails_applied=["MAX_DAILY_RISE"] if index % 10 == 0 else [],
            )
        )
    session.commit()


class TestModelMonitor:
    def test_too_few_predictions_is_not_an_alarm(self, healthy) -> None:
        """Two predictions from a smoke test have near-zero variance, which is
        not evidence the model returns a constant. A monitor that cries wolf on
        an idle service gets muted."""
        seed_predictions(healthy, n=3)
        _, checks = ModelMonitor().prediction_health(healthy)

        names = {c.name for c in checks}
        assert "prediction_volume" in names
        assert all(c.severity is Severity.OK for c in checks)

    def test_a_healthy_prediction_distribution_passes(self, healthy) -> None:
        seed_predictions(healthy)
        stats, checks = ModelMonitor().prediction_health(healthy)

        assert stats["n"] == 120
        assert stats["demand_std"] > 0.02
        assert all(c.severity is Severity.OK for c in checks)

    def test_a_collapsed_distribution_is_critical(self, healthy) -> None:
        """The classic sign of a broken artifact or an all-null input."""
        seed_predictions(healthy, constant=True)
        _, checks = ModelMonitor().prediction_health(healthy)

        variance = next(c for c in checks if c.name == "prediction_variance")
        assert variance.severity is Severity.CRITICAL
        assert "constant" in variance.message

    def test_guardrail_pressure_is_counted(self, healthy) -> None:
        seed_predictions(healthy)
        stats, counts, checks = ModelMonitor().pricing_health(healthy)

        assert counts["MAX_DAILY_RISE"] == 12
        assert stats["clamped_share"] == pytest.approx(0.1)
        assert all(c.severity is Severity.OK for c in checks)

    def test_constant_guardrail_firing_warns(self, healthy) -> None:
        """A guardrail firing on most decisions means the model wants prices the
        business will not allow -- a retuning signal, not a success."""
        seed_predictions(healthy, n=60)
        healthy.query(PricingDecision).update(
            {PricingDecision.guardrails_applied: ["MAX_DAILY_RISE"]},
            synchronize_session=False,
        )
        healthy.commit()

        _, _, checks = ModelMonitor().pricing_health(healthy)
        pressure = next(c for c in checks if c.name == "guardrail_pressure")
        assert pressure.severity is Severity.WARNING

    def test_the_seasonality_caveat_fires_on_short_history(self, healthy) -> None:
        """The most important caveat in the layer: on under two years of data,
        PSI cannot separate drift from season."""
        caveat = ModelMonitor.seasonality_caveat(healthy, window_days=30)
        assert caveat is not None
        assert caveat.severity is Severity.WARNING
        assert "seasons" in caveat.message

    def test_realised_accuracy_needs_completed_nights(self, healthy) -> None:
        seed_predictions(healthy, n=10)
        assert ModelMonitor().realised_accuracy(healthy) is None

    def test_the_full_run_produces_a_report(self, healthy) -> None:
        seed_predictions(healthy)
        report = ModelMonitor().run(healthy)

        payload = report.as_dict()
        assert "severity" in payload
        assert "prediction_stats" in payload
        assert json.loads(json.dumps(payload, default=str))


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #


class TestModelRegistry:
    @pytest.fixture(autouse=True)
    def _clean(self):
        reset_registry()
        yield
        reset_registry()

    def test_an_empty_directory_is_not_fatal(self, settings, tmp_path) -> None:
        """Missing artifacts are a degraded service, not a failed boot. An API
        that will not start before a training run cannot be deployed before it
        is trained."""
        registry = ModelRegistry(settings)
        registry.artifact_dir = tmp_path / "artifacts"

        loaded = registry.load()
        assert loaded.any_loaded is False
        assert registry.is_loaded is False
        assert "registry" in loaded.errors

    def test_discovers_versions_from_filenames(self, settings, tmp_path) -> None:
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        for name in ("gbr_v1.joblib", "prophet_v1.joblib", "gbr_v3.joblib"):
            (artifacts / name).write_text("x", encoding="utf-8")
        (artifacts / "training_report_v1.json").write_text(
            json.dumps({"version": "v1", "steps": []}), encoding="utf-8"
        )

        registry = ModelRegistry(settings)
        registry.artifact_dir = artifacts

        assert set(registry.discover()) == {"v1", "v3"}
        assert registry.latest_version() == "v3"

    def test_versions_are_ordered_numerically_not_lexically(self, settings, tmp_path) -> None:
        """v10 is newer than v9, which a string comparison gets backwards."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        for version in ("v2", "v9", "v10"):
            (artifacts / f"gbr_{version}.joblib").write_text("x", encoding="utf-8")

        registry = ModelRegistry(settings)
        registry.artifact_dir = artifacts
        assert registry.latest_version() == "v10"

    def test_an_explicit_version_beats_the_latest(self, settings, tmp_path) -> None:
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        for version in ("v1", "v5"):
            (artifacts / f"gbr_{version}.joblib").write_text("x", encoding="utf-8")

        registry = ModelRegistry(settings)
        registry.artifact_dir = artifacts
        assert registry.resolve_version("v1") == "v1"

    def test_configuration_can_pin_a_version(self, settings, tmp_path, monkeypatch) -> None:
        """'Whatever trained last' is fine on a laptop and unacceptable in
        production."""
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        for version in ("v1", "v5"):
            (artifacts / f"gbr_{version}.joblib").write_text("x", encoding="utf-8")

        monkeypatch.setattr(settings.model, "model_active_version", "v1")
        registry = ModelRegistry(settings)
        registry.artifact_dir = artifacts
        assert registry.resolve_version() == "v1"

    def test_a_corrupt_artifact_is_recorded_not_raised(self, settings, tmp_path) -> None:
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "gbr_v1.joblib").write_text("not a joblib file", encoding="utf-8")

        registry = ModelRegistry(settings)
        registry.artifact_dir = artifacts

        loaded = registry.load()
        assert loaded.gbr is None
        assert "gradient_boosting" in loaded.errors

    def test_status_is_serialisable(self, settings, tmp_path) -> None:
        registry = ModelRegistry(settings)
        registry.artifact_dir = tmp_path / "artifacts"
        registry.load()

        assert json.loads(json.dumps(registry.status(), default=str))

    def test_the_catalogue_is_newest_first(self, settings, tmp_path) -> None:
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        for version in ("v1", "v2", "v3"):
            (artifacts / f"gbr_{version}.joblib").write_text("x", encoding="utf-8")

        registry = ModelRegistry(settings)
        registry.artifact_dir = artifacts
        assert [row["version"] for row in registry.catalogue()] == ["v3", "v2", "v1"]
