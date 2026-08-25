"""Tests for the training pipeline.

Two layers:

* **Orchestration**, tested against a small real database. The pipeline's job is
  to sequence steps, version artifacts and record what happened, and the
  property that matters most is that a failure in one model does not destroy the
  other.
* **Bookkeeping** -- versioning, reports, summaries -- tested as pure functions.

The integration test deliberately trains real models on a small dataset rather
than mocking them. A pipeline test with both models mocked out verifies only
that the mocks were called.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from database.models import (
    Booking,
    Competitor,
    CompetitorPrice,
    DemandFeature,
    RoomType,
    Season,
)
from features.feature_engineering import FEATURE_VERSION, FeatureConfig
from features.feature_store import FeatureStore, load_feature_list
from models.gradient_boosting_model import GradientBoostingConfig, GradientBoostingDemandModel
from models.prophet_model import ProphetBundle
from training.pipeline import (
    MIN_TRAINING_ROWS,
    StepOutcome,
    TrainingFailed,
    TrainingPipeline,
    TrainingResult,
    latest_report,
)

#: Long enough for Prophet (which needs 90+ days) after a 45-day holdout, and
#: for the pipeline's 500-row floor across two room types.
HISTORY_DAYS = 320
LAST_STAY = date(2026, 6, 30)


def seed_feature_store(session: Session, *, days: int = HISTORY_DAYS) -> None:
    """Insert enough raw data for a genuine, if small, training run."""
    from database.models import Hotel, MarketSegment, Room

    session.add(
        Hotel(
            hotel_id="H900",
            hotel_name="Pipeline Test Hotel",
            city="Mumbai",
            star_rating=4,
            total_rooms=80,
            segment=MarketSegment.MIXED,
        )
    )
    session.flush()
    for index, (room_type, count) in enumerate(
        [(RoomType.STANDARD, 50), (RoomType.DELUXE, 30)]
    ):
        session.add(
            Room(
                room_id=f"H900-R{index}",
                hotel_id="H900",
                room_type=room_type,
                capacity=2,
                room_count=count,
                base_price=5000.0 + index * 1500,
            )
        )
    session.flush()

    for offset in range(days):
        stay = LAST_STAY - timedelta(days=offset)
        # A weekly cycle plus a slow drift, so there is something to learn.
        weekend_lift = 8 if stay.weekday() >= 5 else 0
        drift = offset // 60

        for room_type in (RoomType.STANDARD, RoomType.DELUXE):
            for lead, count in ((30, 4), (21, 5), (14, 6), (7, 7), (2, 5)):
                session.add(
                    Booking(
                        hotel_id="H900",
                        room_type=room_type,
                        booking_date=stay - timedelta(days=lead),
                        check_in_date=stay,
                        check_out_date=stay + timedelta(days=1),
                        booking_count=count + weekend_lift + drift,
                        cancellation_count=1,
                        revenue=(count + weekend_lift) * 5200.0,
                        adr=5200.0,
                        lead_time_days=lead,
                    )
                )
            session.add(
                CompetitorPrice(
                    hotel_id="H900",
                    room_type=room_type,
                    competitor=Competitor.BOOKING,
                    check_in_date=stay,
                    price=5800.0 + weekend_lift * 40,
                    collected_at=datetime.combine(
                        stay - timedelta(days=10), datetime.min.time(),
                        tzinfo=timezone.utc,
                    ),
                )
            )
            session.add(
                DemandFeature(
                    hotel_id="H900",
                    room_type=room_type,
                    stay_date=stay,
                    day_of_week=stay.weekday(),
                    is_weekend=stay.weekday() >= 5,
                    season=Season.MONSOON,
                    holiday_flag=False,
                    local_event_score=0.1,
                    weather_score=0.6,
                    search_demand=0.5 + weekend_lift / 40.0,
                )
            )
    session.commit()

    FeatureStore(
        FeatureConfig(incomplete_tail_days=0)
    ).build_and_store(session)


@pytest.fixture
def trained_session(db_session: Session, tmp_path: Path) -> Session:
    seed_feature_store(db_session)
    return db_session


@pytest.fixture
def pipeline(settings, tmp_path: Path) -> TrainingPipeline:
    pipe = TrainingPipeline(settings, gbr_config=GradientBoostingConfig(n_estimators=60))
    pipe.artifact_dir = tmp_path / "artifacts"
    return pipe


# --------------------------------------------------------------------------- #
# Versioning and bookkeeping
# --------------------------------------------------------------------------- #


class TestVersioning:
    def test_first_version_is_v1(self, pipeline) -> None:
        assert pipeline.next_version() == "v1"

    def test_version_increments_past_existing_reports(self, pipeline) -> None:
        pipeline.artifact_dir.mkdir(parents=True, exist_ok=True)
        for name in ("training_report_v1.json", "training_report_v7.json"):
            (pipeline.artifact_dir / name).write_text("{}", encoding="utf-8")
        assert pipeline.next_version() == "v8"

    def test_unrelated_files_do_not_affect_the_version(self, pipeline) -> None:
        pipeline.artifact_dir.mkdir(parents=True, exist_ok=True)
        (pipeline.artifact_dir / "gbr_v99.joblib").write_text("x", encoding="utf-8")
        (pipeline.artifact_dir / "training_report_vX.json").write_text("{}", encoding="utf-8")
        assert pipeline.next_version() == "v1"


class TestTrainingResult:
    def _result(self, **overrides) -> TrainingResult:
        payload = dict(
            version="v1",
            feature_version=FEATURE_VERSION,
            started_at="2026-08-24T10:00:00+00:00",
            finished_at="2026-08-24T10:01:00+00:00",
            duration_seconds=60.0,
            n_rows=1000,
            n_train=900,
            n_test=100,
            train_window=["2025-09-01", "2026-05-01"],
            test_window=["2026-05-02", "2026-06-30"],
            dataset_hash="abc",
        )
        payload.update(overrides)
        return TrainingResult(**payload)

    def test_succeeds_when_any_step_succeeds(self) -> None:
        """One flaky Stan fit must not mean 'no models today'."""
        result = self._result(
            steps=[
                StepOutcome("gradient_boosting", True, metrics={"mae": 0.06}),
                StepOutcome("prophet", False, error="InsufficientHistory"),
            ]
        )
        assert result.succeeded is True

    def test_fails_when_no_step_succeeds(self) -> None:
        result = self._result(
            steps=[StepOutcome("gradient_boosting", False, error="boom")]
        )
        assert result.succeeded is False

    def test_step_lookup(self) -> None:
        result = self._result(steps=[StepOutcome("prophet", True)])
        assert result.step("prophet") is not None
        assert result.step("absent") is None

    def test_summary_quotes_the_baseline_gain(self) -> None:
        result = self._result(
            steps=[
                StepOutcome(
                    "gradient_boosting", True,
                    metrics={"mae": 0.07}, baseline={"mae": 0.14},
                )
            ]
        )
        assert "50% better than baseline" in result.summary()

    def test_summary_names_a_failed_step(self) -> None:
        result = self._result(steps=[StepOutcome("prophet", False, error="Stan died")])
        assert "FAILED" in result.summary()
        assert "Stan died" in result.summary()

    def test_as_dict_is_json_serialisable(self) -> None:
        result = self._result(steps=[StepOutcome("prophet", True)])
        assert json.loads(json.dumps(result.as_dict(), default=str))["succeeded"] is True


# --------------------------------------------------------------------------- #
# Guard rails
# --------------------------------------------------------------------------- #


class TestPreconditions:
    def test_empty_feature_store_is_refused(self, db_session, pipeline) -> None:
        with pytest.raises(ValueError, match="build_features"):
            pipeline.load_features(db_session)

    def test_too_little_data_is_refused_with_a_number(
        self, db_session, pipeline
    ) -> None:
        seed_feature_store(db_session, days=40)
        with pytest.raises(TrainingFailed, match=str(MIN_TRAINING_ROWS)):
            pipeline.load_features(db_session)


# --------------------------------------------------------------------------- #
# The pipeline end to end
# --------------------------------------------------------------------------- #


class TestPipelineRun:
    @pytest.fixture
    def result(self, trained_session, pipeline) -> TrainingResult:
        return pipeline.run(
            trained_session, test_days=45, backtest_folds=0
        )

    def test_both_models_are_trained(self, result) -> None:
        assert {s.name for s in result.steps} == {"gradient_boosting", "prophet"}
        assert all(s.succeeded for s in result.steps), [
            s.error for s in result.steps if not s.succeeded
        ]

    def test_artifacts_are_written(self, result, pipeline) -> None:
        for name in ("gradient_boosting", "prophet", "feature_list", "report"):
            assert Path(result.artifacts[name]).is_file(), name

    def test_saved_models_load_and_predict(self, result, trained_session) -> None:
        """An artifact that cannot be loaded is not an artifact."""
        gbr = GradientBoostingDemandModel.load(Path(result.artifacts["gradient_boosting"]))
        matrix = FeatureStore.load_model_matrix(trained_session)
        assert len(gbr.predict(matrix)) == len(matrix)

        bundle = ProphetBundle.load(Path(result.artifacts["prophet"]))
        assert bundle.demand_on("H900", RoomType.DELUXE, LAST_STAY) is not None

    def test_feature_list_matches_the_running_code(self, result) -> None:
        """The train/serve contract, written to disk beside the model."""
        from features.feature_engineering import FEATURE_COLUMNS
        from features.feature_store import validate_feature_list

        saved = load_feature_list(Path(result.artifacts["feature_list"]).parent)
        validate_feature_list(saved)
        assert saved["features"] == list(FEATURE_COLUMNS)

    def test_split_is_chronological_and_recorded(self, result) -> None:
        assert result.train_window[1] <= result.test_window[0]
        assert result.n_train + result.n_test == result.n_rows

    def test_dataset_hash_is_recorded(self, result) -> None:
        assert result.dataset_hash and len(result.dataset_hash) == 64

    def test_both_models_beat_the_baseline_on_the_same_holdout(self, result) -> None:
        """Comparable numbers are the whole point of scoring them alike -- the
        blend weight between the two models has no basis otherwise."""
        for name in ("gradient_boosting", "prophet"):
            step = result.step(name)
            assert step.metrics["mae"] < step.baseline["mae"], name

    def test_importances_and_horizon_breakdown_are_reported(self, result) -> None:
        assert result.feature_importance
        assert {"feature", "importance"} <= set(result.feature_importance[0])
        assert result.per_horizon

    def test_report_json_is_complete(self, result) -> None:
        payload = json.loads(
            Path(result.artifacts["report"]).read_text(encoding="utf-8")
        )
        assert payload["version"] == result.version
        assert payload["succeeded"] is True
        assert len(payload["steps"]) == 2

    def test_latest_report_finds_the_newest_run(self, result, pipeline) -> None:
        assert latest_report(pipeline.artifact_dir)["version"] == result.version

    def test_latest_report_is_none_when_nothing_is_trained(self, tmp_path) -> None:
        assert latest_report(tmp_path) is None


class TestPartialFailure:
    def test_prophet_failure_does_not_lose_the_gbr(
        self, trained_session, pipeline, monkeypatch
    ) -> None:
        """The property this pipeline exists to guarantee."""
        def _explode(*args, **kwargs):
            raise RuntimeError("stan backend unavailable")

        monkeypatch.setattr(ProphetBundle, "fit_all", _explode)

        result = pipeline.run(trained_session, test_days=45, backtest_folds=0)

        assert result.succeeded is True
        assert result.step("gradient_boosting").succeeded is True
        assert result.step("prophet").succeeded is False
        assert "stan backend unavailable" in result.step("prophet").error
        assert Path(result.artifacts["gradient_boosting"]).is_file()

    def test_gbr_failure_does_not_lose_prophet(
        self, trained_session, pipeline, monkeypatch
    ) -> None:
        def _explode(*args, **kwargs):
            raise RuntimeError("out of memory")

        monkeypatch.setattr(GradientBoostingDemandModel, "fit", _explode)

        result = pipeline.run(trained_session, test_days=45, backtest_folds=0)

        assert result.succeeded is True
        assert result.step("prophet").succeeded is True
        assert result.step("gradient_boosting").succeeded is False

    def test_both_failing_raises(self, trained_session, pipeline, monkeypatch) -> None:
        def _explode(*args, **kwargs):
            raise RuntimeError("nope")

        monkeypatch.setattr(GradientBoostingDemandModel, "fit", _explode)
        monkeypatch.setattr(ProphetBundle, "fit_all", _explode)

        with pytest.raises(TrainingFailed, match="no model could be trained"):
            pipeline.run(trained_session, test_days=45, backtest_folds=0)

    def test_selective_training(self, trained_session, pipeline) -> None:
        result = pipeline.run(
            trained_session, test_days=45, train_prophet=False, backtest_folds=0
        )
        assert [s.name for s in result.steps] == ["gradient_boosting"]
        assert "prophet" not in result.artifacts
