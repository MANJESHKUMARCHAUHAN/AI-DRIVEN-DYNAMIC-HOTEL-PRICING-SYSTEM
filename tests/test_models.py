"""Tests for the metrics module and the Prophet forecaster.

Prophet fits are genuinely slow (roughly half a second each), so the series
fixtures are module-scoped and deliberately small. The tests that need a *fitted*
model share one; the rest assert on pure functions.

The most important test in this file is
:meth:`TestProphetQuality.test_beats_the_predict_the_mean_baseline`. Everything
else checks that the wrapper behaves; that one checks that the model is worth
having at all. An earlier configuration of this forecaster passed every
structural test in this file while being 66% *worse* than predicting the mean --
which is precisely the failure a structural test cannot see.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from models.metrics import (
    MAPE_FLOOR,
    RegressionMetrics,
    baseline_metrics,
    bias,
    evaluate,
    interval_coverage,
    mae,
    mape,
    mape_coverage,
    r2,
    rmse,
    smape,
    weighted_mape,
)
from models.prophet_model import (
    MIN_TRAINING_DAYS,
    InsufficientHistory,
    ProphetBundle,
    ProphetConfig,
    ProphetDemandForecaster,
    holiday_frame,
)

SERIES_START = date(2025, 6, 1)
SERIES_DAYS = 420


def make_series(
    *,
    days: int = SERIES_DAYS,
    level: float = 0.65,
    weekly_amplitude: float = 0.18,
    trend: float = 0.05,
    noise: float = 0.03,
    seed: int = 7,
    weekend_positive: bool = True,
) -> pd.DataFrame:
    """A demand series with a level, a weekly cycle, a trend and noise.

    Deterministic, and structured enough that a forecaster which fails to beat
    the mean on it is broken rather than unlucky.
    """
    rng = np.random.default_rng(seed)
    days_index = pd.date_range(SERIES_START, periods=days, freq="D")
    weekday = days_index.dayofweek.to_numpy()

    weekend = np.isin(weekday, (5, 6))
    weekly = np.where(weekend, weekly_amplitude, -weekly_amplitude / 2.0)
    if not weekend_positive:
        weekly = -weekly

    drift = np.linspace(0.0, trend, days)
    values = level + weekly + drift + rng.normal(0.0, noise, days)
    return pd.DataFrame({"ds": days_index, "y": np.clip(values, 0.02, 1.2)})


@pytest.fixture(scope="module")
def series() -> pd.DataFrame:
    return make_series()


@pytest.fixture(scope="module")
def fitted(series: pd.DataFrame) -> ProphetDemandForecaster:
    return ProphetDemandForecaster(("H001", "deluxe")).fit(series)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


class TestPointMetrics:
    def test_perfect_prediction_scores_zero_error(self) -> None:
        actual = [0.5, 0.6, 0.7]
        assert mae(actual, actual) == 0.0
        assert rmse(actual, actual) == 0.0
        assert mape(actual, actual) == 0.0
        assert r2(actual, actual) == 1.0
        assert bias(actual, actual) == 0.0

    def test_mae_is_the_mean_absolute_difference(self) -> None:
        assert mae([1.0, 2.0, 3.0], [1.5, 2.0, 2.0]) == pytest.approx(0.5)

    def test_rmse_punishes_large_errors_harder(self) -> None:
        spread = [0.0, 0.0, 3.0]
        even = [1.0, 1.0, 1.0]
        actual = [0.0, 0.0, 0.0]
        assert mae(actual, spread) == pytest.approx(mae(actual, even))
        assert rmse(actual, spread) > rmse(actual, even)

    def test_bias_signals_systematic_over_prediction(self) -> None:
        """Good MAE with a large bias means consistently over-charging."""
        assert bias([0.5, 0.5], [0.6, 0.6]) == pytest.approx(0.1)
        assert bias([0.5, 0.5], [0.4, 0.4]) == pytest.approx(-0.1)

    def test_r2_is_negative_when_worse_than_the_mean(self) -> None:
        """Not clipped: 'worse than a constant' is exactly what we need to see."""
        actual = [0.2, 0.5, 0.8]
        assert r2(actual, [0.8, 0.5, 0.2]) < 0

    def test_mismatched_lengths_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="different shapes"):
            mae([1.0, 2.0], [1.0])

    def test_empty_input_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            mae([], [])

    def test_non_finite_pairs_are_dropped(self) -> None:
        assert mae([1.0, np.nan, 3.0], [1.0, 5.0, 3.0]) == 0.0

    def test_all_non_finite_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="no finite"):
            mae([np.nan, np.nan], [1.0, 2.0])


class TestPercentageMetrics:
    def test_mape_skips_near_zero_actuals(self) -> None:
        """Dividing by 0.001 demand produces a 40,000% error and a useless
        headline number."""
        actual = [0.5, 0.001, 0.5]
        predicted = [0.55, 0.4, 0.45]
        assert mape(actual, predicted) == pytest.approx(10.0)

    def test_mape_reports_its_own_coverage(self) -> None:
        """A MAPE computed over a third of the rows must be visibly so."""
        actual = [0.5, 0.001, 0.001]
        assert mape_coverage(actual) == pytest.approx(1 / 3)

    def test_mape_is_nan_when_nothing_clears_the_floor(self) -> None:
        assert math.isnan(mape([0.001, 0.002], [0.5, 0.5]))

    def test_smape_stays_bounded_near_zero(self) -> None:
        """Where MAPE explodes, sMAPE degrades gracefully."""
        assert smape([0.001], [0.5]) <= 200.0

    def test_weighted_mape_prioritises_busy_nights(self) -> None:
        """Being wrong about a full hotel matters more than about an empty one."""
        actual = [1.0, 0.1]
        big_miss_when_busy = [0.5, 0.1]
        big_miss_when_quiet = [1.0, 0.05]
        assert weighted_mape(actual, big_miss_when_busy) > weighted_mape(
            actual, big_miss_when_quiet
        )

    def test_floor_is_configurable(self) -> None:
        actual = [0.5, 0.05]
        predicted = [0.5, 0.10]
        assert mape(actual, predicted, floor=0.01) > mape(actual, predicted, floor=0.1)

    def test_default_floor_is_one_room_in_fifty(self) -> None:
        assert MAPE_FLOOR == pytest.approx(0.02)


class TestIntervalCoverage:
    def test_full_coverage(self) -> None:
        assert interval_coverage([0.5, 0.6], [0.4, 0.5], [0.6, 0.7]) == 1.0

    def test_no_coverage(self) -> None:
        assert interval_coverage([0.9, 0.9], [0.4, 0.5], [0.6, 0.7]) == 0.0

    def test_boundaries_count_as_covered(self) -> None:
        assert interval_coverage([0.4, 0.6], [0.4, 0.5], [0.6, 0.7]) == 1.0


class TestEvaluateAndBaseline:
    def test_evaluate_returns_every_metric(self) -> None:
        actual = [0.5, 0.6, 0.7, 0.8]
        predicted = [0.52, 0.58, 0.72, 0.78]
        metrics = evaluate(actual, predicted)
        assert isinstance(metrics, RegressionMetrics)
        assert metrics.n == 4
        assert metrics.interval_coverage is None
        assert set(metrics.as_dict()) >= {"mae", "rmse", "mape", "r2", "bias"}

    def test_evaluate_adds_coverage_when_bounds_are_given(self) -> None:
        metrics = evaluate([0.5], [0.5], lower=[0.4], upper=[0.6])
        assert metrics.interval_coverage == 1.0

    def test_summary_is_one_readable_line(self) -> None:
        summary = evaluate([0.5, 0.7], [0.55, 0.65]).summary()
        assert "MAE=" in summary and "R2=" in summary and "\n" not in summary

    def test_baseline_predicts_the_mean(self) -> None:
        """Every model number should be read against this."""
        actual = [0.4, 0.6, 0.8]
        assert baseline_metrics(actual).mae == pytest.approx(
            mae(actual, [0.6, 0.6, 0.6])
        )

    def test_baseline_r2_is_zero_by_definition(self) -> None:
        assert baseline_metrics([0.4, 0.6, 0.8]).r2 == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Prophet -- structure
# --------------------------------------------------------------------------- #


class TestHolidayFrame:
    def test_has_the_columns_prophet_requires(self) -> None:
        frame = holiday_frame([2026])
        assert {"holiday", "ds", "lower_window", "upper_window", "prior_scale"} <= set(
            frame.columns
        )

    def test_contains_the_major_indian_festivals(self) -> None:
        names = set(holiday_frame([2026])["holiday"])
        assert {"Diwali", "Holi", "Independence Day", "Christmas"} <= names

    def test_prior_scale_is_constant_per_holiday_name(self) -> None:
        """Prophet raises 'does not have consistent prior scale specification'
        otherwise, and our calendar lets significance vary by year."""
        frame = holiday_frame(range(2024, 2028))
        per_name = frame.groupby("holiday")["prior_scale"].nunique()
        assert (per_name == 1).all()

    def test_significant_holidays_get_a_larger_prior(self) -> None:
        """Mahavir Jayanti should not be fitted as hard as Diwali."""
        frame = holiday_frame([2026]).drop_duplicates("holiday").set_index("holiday")
        assert frame.loc["Diwali", "prior_scale"] > frame.loc["Republic Day", "prior_scale"]
        assert (
            frame.loc["Republic Day", "prior_scale"]
            > frame.loc["Makar Sankranti / Pongal", "prior_scale"]
        )

    def test_window_spreads_the_effect_around_the_day(self) -> None:
        frame = holiday_frame([2026], window=(-2, 2))
        assert (frame["lower_window"] == -2).all()
        assert (frame["upper_window"] == 2).all()


class TestProphetForecaster:
    def test_fit_records_the_training_window(self, fitted) -> None:
        assert fitted.n_observations == SERIES_DAYS
        assert fitted.history_start == SERIES_START
        assert fitted.history_end == SERIES_START + timedelta(days=SERIES_DAYS - 1)

    def test_forecast_returns_only_future_dates(self, fitted) -> None:
        """Including the in-sample fit is how a backtest quietly becomes an
        in-sample score."""
        result = fitted.forecast(14)
        assert len(result.frame) == 14
        assert result.frame["ds"].min().date() > fitted.history_end

    def test_forecast_columns(self, fitted) -> None:
        assert set(fitted.forecast(7).frame.columns) == {
            "ds", "yhat", "yhat_lower", "yhat_upper", "trend"
        }

    def test_forecast_is_non_negative(self, fitted) -> None:
        """Demand is a fraction; a negative lower bound is an artefact of the
        additive error term, not a possible outcome."""
        frame = fitted.forecast(30).frame
        assert (frame[["yhat", "yhat_lower", "yhat_upper"]] >= 0).all().all()

    def test_interval_brackets_the_point_forecast(self, fitted) -> None:
        frame = fitted.forecast(30).frame
        assert (frame["yhat_lower"] <= frame["yhat"]).all()
        assert (frame["yhat"] <= frame["yhat_upper"]).all()

    def test_horizons_are_nested_slices_of_one_forecast(self, fitted) -> None:
        """Day 7 must not depend on whether we asked for 7 days or 30."""
        results = fitted.forecast_horizons([7, 14, 30])
        assert [len(r.frame) for r in results.values()] == [7, 14, 30]
        pd.testing.assert_frame_equal(
            results[7].frame, results[30].frame.head(7).reset_index(drop=True)
        )

    def test_specified_horizons_match_the_specification(self) -> None:
        from models.prophet_model import DEFAULT_HORIZONS

        assert DEFAULT_HORIZONS == (7, 14, 30)

    def test_predict_for_dates_handles_in_sample_dates(self, fitted) -> None:
        days = [SERIES_START + timedelta(days=n) for n in (10, 20, 30)]
        assert len(fitted.predict_for_dates(days)) == 3

    def test_short_history_is_refused(self) -> None:
        with pytest.raises(InsufficientHistory, match=str(MIN_TRAINING_DAYS)):
            ProphetDemandForecaster(("H001", "suite")).fit(make_series(days=40))

    def test_missing_columns_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="missing column"):
            ProphetDemandForecaster(("H001", "suite")).fit(
                pd.DataFrame({"date": [], "value": []})
            )

    def test_forecasting_before_fitting_is_an_error(self) -> None:
        with pytest.raises(RuntimeError, match="has not been fitted"):
            ProphetDemandForecaster(("H001", "suite")).forecast(7)

    def test_gaps_in_the_calendar_are_filled(self) -> None:
        """'Last week' should always mean seven rows."""
        series = make_series(days=200)
        with_gaps = series.drop(series.index[50:60])
        forecaster = ProphetDemandForecaster(("H001", "deluxe")).fit(with_gaps)
        assert forecaster.n_observations == 200

    def test_short_history_disables_yearly_and_holidays(self, fitted) -> None:
        """Terms fitted to a single observation are noise Prophet extrapolates
        with full confidence."""
        assert fitted.yearly_enabled is False
        assert fitted.holidays_enabled is False

    def test_config_can_force_the_seasonal_terms_on(self) -> None:
        config = ProphetConfig(yearly_seasonality=True, use_holidays=True)
        forecaster = ProphetDemandForecaster(("H001", "deluxe"), config).fit(
            make_series(days=200)
        )
        assert forecaster.yearly_enabled is True
        assert forecaster.holidays_enabled is True


class TestProphetSerialisation:
    def test_state_round_trip_preserves_predictions(self, fitted) -> None:
        """Prophet objects hold a Stan backend and do not pickle across
        versions; the state carries JSON instead.

        ``yhat`` and ``trend`` are deterministic and must match exactly. The
        interval bounds are *not*: Prophet estimates them by Monte Carlo
        sampling, so they move by ~0.3% between calls on the same model. That is
        Prophet's behaviour rather than a serialisation defect -- and it is why
        the confidence score derived from interval width is rounded before it
        reaches a caller.
        """
        restored = ProphetDemandForecaster.from_state(fitted.to_state())
        days = [fitted.history_end + timedelta(days=n) for n in (1, 5, 10)]

        original = fitted.predict_for_dates(days)
        copy = restored.predict_for_dates(days)

        pd.testing.assert_frame_equal(
            original[["ds", "yhat", "trend"]], copy[["ds", "yhat", "trend"]]
        )
        for column in ("yhat_lower", "yhat_upper"):
            assert copy[column].to_numpy() == pytest.approx(
                original[column].to_numpy(), rel=0.05
            )

    def test_state_carries_the_training_window(self, fitted) -> None:
        state = fitted.to_state()
        assert state["n_observations"] == SERIES_DAYS
        assert isinstance(state["model_json"], str)

    def test_unfitted_model_cannot_be_serialised(self) -> None:
        with pytest.raises(RuntimeError, match="has not been fitted"):
            ProphetDemandForecaster(("H001", "suite")).to_state()


# --------------------------------------------------------------------------- #
# Prophet -- does it actually work
# --------------------------------------------------------------------------- #


class TestProphetQuality:
    def test_beats_the_predict_the_mean_baseline(self, series) -> None:
        """The test that matters.

        An earlier configuration of this forecaster passed every structural test
        above while scoring 66% *worse* than the mean, because yearly
        seasonality was being fitted on less than one cycle of history. Only a
        quality assertion catches that.
        """
        forecaster = ProphetDemandForecaster(("H001", "deluxe"))
        metrics = forecaster.backtest(series, horizon_days=30, folds=2)
        baseline = baseline_metrics(series["y"].iloc[-60:])

        assert metrics.mae < baseline.mae * 0.75, (
            f"Prophet MAE {metrics.mae:.4f} vs baseline {baseline.mae:.4f} -- "
            "the forecaster is not earning its place"
        )
        assert metrics.r2 > 0.0

    def test_uncertainty_intervals_are_calibrated(self, series) -> None:
        """The pricing engine derives confidence from interval width, so an
        80% band that covers 30% of outcomes would produce confident bad prices."""
        metrics = ProphetDemandForecaster(("H001", "deluxe")).backtest(
            series, horizon_days=30, folds=2
        )
        assert 0.6 <= metrics.interval_coverage <= 0.95

    def test_learns_the_weekly_shape(self, series) -> None:
        forecaster = ProphetDemandForecaster(("H001", "deluxe")).fit(series)
        frame = forecaster.forecast(28).frame
        frame["dow"] = frame["ds"].dt.dayofweek

        weekend = frame[frame["dow"] >= 5]["yhat"].mean()
        weekday = frame[frame["dow"] < 5]["yhat"].mean()
        assert weekend > weekday + 0.05

    def test_learns_an_inverted_weekly_shape_too(self) -> None:
        """A business hotel empties at the weekend. A forecaster that has learnt
        'weekend = busy' globally is wrong half the time."""
        forecaster = ProphetDemandForecaster(("H003", "standard")).fit(
            make_series(weekend_positive=False, seed=11)
        )
        frame = forecaster.forecast(28).frame
        frame["dow"] = frame["ds"].dt.dayofweek
        assert frame[frame["dow"] >= 5]["yhat"].mean() < frame[frame["dow"] < 5]["yhat"].mean()

    def test_backtest_refuses_impossible_fold_counts(self, series) -> None:
        with pytest.raises(InsufficientHistory, match="backtesting"):
            ProphetDemandForecaster(("H001", "deluxe")).backtest(
                series, horizon_days=90, folds=10
            )


# --------------------------------------------------------------------------- #
# Bundle
# --------------------------------------------------------------------------- #


class TestProphetBundle:
    @pytest.fixture(scope="class")
    def daily(self) -> pd.DataFrame:
        frames = []
        for index, (hotel, room) in enumerate(
            [("H001", "deluxe"), ("H001", "suite"), ("H004", "standard")]
        ):
            series = make_series(days=200, level=0.5 + index * 0.1, seed=index)
            frames.append(
                series.assign(hotel_id=hotel, room_type=room, stay_date=series["ds"])
                .rename(columns={"y": "target_demand"})
                .drop(columns=["ds"])
            )
        return ProphetBundle.daily_demand(pd.concat(frames, ignore_index=True))

    @pytest.fixture(scope="class")
    def bundle(self, daily: pd.DataFrame) -> ProphetBundle:
        return ProphetBundle().fit_all(daily)

    def test_daily_demand_reshapes_the_feature_matrix(self, daily) -> None:
        assert set(daily.columns) == {"hotel_id", "room_type", "ds", "y"}
        assert daily.groupby(["hotel_id", "room_type"]).ngroups == 3

    def test_daily_demand_rejects_a_frame_without_the_target(self) -> None:
        with pytest.raises(ValueError, match="missing"):
            ProphetBundle.daily_demand(pd.DataFrame({"hotel_id": ["H001"]}))

    def test_one_model_per_series(self, bundle) -> None:
        assert len(bundle.models) == 3
        assert bundle.has("H001", "deluxe")
        assert not bundle.has("H999", "deluxe")

    def test_room_type_enum_and_string_are_equivalent(self, bundle) -> None:
        from database.models import RoomType

        assert bundle.has("H001", RoomType.DELUXE)

    def test_a_failing_series_does_not_stop_the_others(self, daily) -> None:
        """One hotel with three weeks of history must not block thirty-one
        other models from training."""
        short = daily[daily["hotel_id"] == "H001"].groupby(
            ["hotel_id", "room_type"], group_keys=False
        ).head(30)
        mixed = pd.concat([daily[daily["hotel_id"] == "H004"], short])

        bundle = ProphetBundle().fit_all(mixed)
        report = bundle.report_frame()

        assert len(bundle.models) == 1
        assert report["fitted"].sum() == 1
        assert (~report["fitted"]).sum() == 2
        assert report[~report["fitted"]]["error"].str.contains("observation").all()

    def test_unknown_series_raises_on_forecast(self, bundle) -> None:
        with pytest.raises(KeyError, match="no Prophet model"):
            bundle.forecast("H999", "deluxe")

    def test_demand_on_returns_none_for_an_unknown_series(self, bundle) -> None:
        """The pricing engine degrades to the GBR prediction alone; an
        exception here would take a price request down."""
        assert bundle.demand_on("H999", "deluxe", date(2026, 7, 1)) is None

    def test_demand_on_returns_the_forecast_for_one_night(self, bundle) -> None:
        value = bundle.demand_on("H001", "deluxe", SERIES_START + timedelta(days=210))
        assert value is not None
        assert value["lower"] <= value["forecast"] <= value["upper"]

    def test_save_and_load_round_trip(self, bundle, tmp_path) -> None:
        path = tmp_path / "prophet.joblib"
        bundle.save(path)
        reloaded = ProphetBundle.load(path)

        assert set(reloaded.models) == set(bundle.models)
        day = SERIES_START + timedelta(days=205)

        # The point forecast is deterministic; the bounds are Monte Carlo
        # sampled by Prophet and move slightly between calls.
        before = bundle.demand_on("H001", "deluxe", day)
        after = reloaded.demand_on("H001", "deluxe", day)
        assert after["forecast"] == pytest.approx(before["forecast"])
        assert after["trend"] == pytest.approx(before["trend"])
        assert after["lower"] == pytest.approx(before["lower"], rel=0.05)

    def test_refuses_to_save_an_empty_bundle(self, tmp_path) -> None:
        with pytest.raises(RuntimeError, match="no fitted models"):
            ProphetBundle().save(tmp_path / "empty.joblib")

    def test_missing_artifact_gives_an_actionable_error(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="train_models"):
            ProphetBundle.load(tmp_path / "absent.joblib")

    def test_wrong_artifact_format_is_rejected(self, tmp_path) -> None:
        import joblib

        path = tmp_path / "other.joblib"
        joblib.dump({"format": "gbr-v1"}, path)
        with pytest.raises(ValueError, match="not a Prophet bundle"):
            ProphetBundle.load(path)

    def test_aggregate_metrics_is_none_without_a_backtest(self, bundle) -> None:
        assert bundle.aggregate_metrics() is None

    def test_aggregate_metrics_summarises_the_backtests(self, daily) -> None:
        bundle = ProphetBundle().fit_all(daily, backtest_folds=1, backtest_horizon=30)
        aggregate = bundle.aggregate_metrics()
        assert aggregate is not None
        assert aggregate.n == 90  # 3 series x 1 fold x 30 days
        assert aggregate.mae > 0

    def test_report_frame_lists_every_series(self, bundle) -> None:
        report = bundle.report_frame()
        assert len(report) == 3
        assert set(report.columns) >= {"hotel_id", "room_type", "fitted", "error"}


# --------------------------------------------------------------------------- #
# Gradient Boosting
# --------------------------------------------------------------------------- #

from features.feature_engineering import FEATURE_COLUMNS, TARGET_COLUMN  # noqa: E402
from features.feature_store import FeatureVersionMismatch  # noqa: E402
from models.gradient_boosting_model import (  # noqa: E402
    ARTIFACT_FORMAT,
    GradientBoostingConfig,
    GradientBoostingDemandModel,
    dataset_hash,
    time_based_split,
    train_gradient_boosting,
)


def make_matrix(*, days: int = 300, seed: int = 3) -> pd.DataFrame:
    """A feature matrix with real structure, small enough to train in a second.

    The target is a genuine function of occupancy, horizon and search interest
    plus noise, so a model that fails to beat the mean on it is broken rather
    than unlucky -- and the importance assertions have something true to find.
    """
    rng = np.random.default_rng(seed)
    horizons = np.array([0, 3, 7, 14, 30, 60])

    rows = []
    for offset in range(days):
        stay = SERIES_START + timedelta(days=offset)
        for room in ("standard", "deluxe"):
            horizon = float(rng.choice(horizons))
            # On-the-books occupancy: high near check-in, near zero far out.
            progress = 1.0 - horizon / 70.0
            final = float(np.clip(rng.beta(6, 3), 0.05, 1.0))
            occupancy = float(np.clip(final * progress + rng.normal(0, 0.03), 0, 1))
            search = float(np.clip(final + rng.normal(0, 0.15), 0, 1))

            rows.append(
                {
                    "hotel_id": "H001",
                    "room_type": room,
                    "stay_date": pd.Timestamp(stay),
                    "days_to_checkin": horizon,
                    "occupancy_rate": occupancy,
                    "search_demand": search,
                    TARGET_COLUMN: final,
                }
            )

    frame = pd.DataFrame(rows)

    # Fill the rest of the contract with plausible, mostly uninformative values.
    defaults = {
        "available_rooms": lambda f: (1 - f["occupancy_rate"]) * 40,
        "total_rooms": lambda f: 40.0,
        "booking_count": lambda f: f["occupancy_rate"] * 8,
        "cancellation_count": lambda f: 5.0,
        "competitor_rate": lambda f: 6000.0 + f["search_demand"] * 2000,
        "competitor_min_rate": lambda f: 5800.0,
        "competitor_max_rate": lambda f: 6600.0,
        "competitor_count": lambda f: 3.0,
        "lead_time": lambda f: f["days_to_checkin"] + 5,
        "historical_demand": lambda f: 0.65,
        "current_room_price": lambda f: 6200.0,
        "is_weekend": lambda f: (f["stay_date"].dt.dayofweek >= 5).astype(float),
        "day_of_week": lambda f: f["stay_date"].dt.dayofweek.astype(float),
        "holiday_flag": lambda f: 0.0,
        "local_event_score": lambda f: 0.1,
        "weather_score": lambda f: 0.6,
        "holiday_proximity": lambda f: 0.0,
        "competitor_missing": lambda f: 0.0,
        "price_to_competitor": lambda f: f["current_room_price"] / f["competitor_rate"],
        "competitor_spread": lambda f: 0.13,
        "pickup_velocity": lambda f: f["booking_count"] / 7.0,
        "occupancy_x_lead": lambda f: f["occupancy_rate"]
        * (1 - f["days_to_checkin"] / 60.0),
        "demand_pressure": lambda f: f["search_demand"] * 1.1,
        "season_winter": lambda f: 0.0,
        "season_summer": lambda f: 1.0,
        "season_monsoon": lambda f: 0.0,
        "season_autumn": lambda f: 0.0,
    }
    for column, builder in defaults.items():
        if column not in frame.columns:
            value = builder(frame)
            frame[column] = value if not callable(value) else value

    missing = [c for c in FEATURE_COLUMNS if c not in frame.columns]
    assert not missing, f"fixture is missing {missing}"
    return frame


@pytest.fixture(scope="module")
def matrix() -> pd.DataFrame:
    return make_matrix()


@pytest.fixture(scope="module")
def trained(matrix):
    model, split, metrics = train_gradient_boosting(matrix, test_days=45)
    return model, split, metrics


class TestTimeBasedSplit:
    def test_holds_out_the_most_recent_window(self, matrix) -> None:
        """A random split would put next Tuesday in training and last Tuesday
        in test, and report a score the model can never achieve."""
        split = time_based_split(matrix, test_days=45)
        assert split.train["stay_date"].max() <= split.test["stay_date"].min()

    def test_no_stay_date_appears_on_both_sides(self, matrix) -> None:
        split = time_based_split(matrix, test_days=45)
        assert not set(split.train["stay_date"]) & set(split.test["stay_date"])

    def test_every_row_lands_somewhere(self, matrix) -> None:
        split = time_based_split(matrix, test_days=45)
        assert len(split.train) + len(split.test) == len(matrix)

    def test_impossible_holdout_is_rejected(self, matrix) -> None:
        with pytest.raises(ValueError, match="history spans"):
            time_based_split(matrix, test_days=10_000)

    def test_missing_date_column_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="no 'stay_date'"):
            time_based_split(pd.DataFrame({"a": [1]}))


class TestDatasetHash:
    def test_same_data_same_hash(self, matrix) -> None:
        X, y = matrix[list(FEATURE_COLUMNS)], matrix[TARGET_COLUMN]
        assert dataset_hash(X, y) == dataset_hash(X, y)

    def test_changed_target_changes_the_hash(self, matrix) -> None:
        X, y = matrix[list(FEATURE_COLUMNS)], matrix[TARGET_COLUMN]
        assert dataset_hash(X, y) != dataset_hash(X, y * 1.01)

    def test_reordered_columns_change_the_hash(self, matrix) -> None:
        """Column order is part of the contract; a model reading a shifted
        matrix produces plausible, wrong numbers."""
        X, y = matrix[list(FEATURE_COLUMNS)], matrix[TARGET_COLUMN]
        shuffled = X[[FEATURE_COLUMNS[1], FEATURE_COLUMNS[0], *FEATURE_COLUMNS[2:]]]
        assert dataset_hash(X, y) != dataset_hash(shuffled, y)


class TestGradientBoostingModel:
    def test_predicts_one_value_per_row(self, trained) -> None:
        model, split, _ = trained
        assert model.predict(split.test).shape == (len(split.test),)

    def test_predictions_are_never_negative(self, trained) -> None:
        """A negative demand is an ensemble artefact, and a price adjustment
        computed from one would move the wrong way."""
        model, split, _ = trained
        assert (model.predict(split.test) >= 0).all()

    def test_metadata_describes_the_run(self, trained) -> None:
        model, split, _ = trained
        meta = model.metadata
        assert meta.n_train == len(split.train)
        assert meta.features == list(FEATURE_COLUMNS)
        assert meta.dataset_hash and len(meta.dataset_hash) == 64
        assert meta.residual_std and meta.residual_std > 0
        assert meta.train_end < meta.test_start

    def test_metadata_records_the_baseline_alongside_the_score(self, trained) -> None:
        """A model number with no baseline next to it is unreadable."""
        model, _, _ = trained
        assert model.metadata.metrics["mae"] < model.metadata.baseline_metrics["mae"]

    def test_early_stopping_is_recorded(self, trained) -> None:
        model, _, _ = trained
        assert 0 < model.metadata.n_estimators_used <= model.config.n_estimators

    def test_predict_one_returns_a_band_and_a_confidence(self, trained) -> None:
        model, split, _ = trained
        result = model.predict_one(split.test.head(1))
        assert result["lower"] <= result["demand"] <= result["upper"]
        assert 0.0 <= result["confidence"] <= 1.0

    def test_confidence_falls_as_spread_grows(self) -> None:
        tight = GradientBoostingDemandModel.confidence_from_spread(0.01, 0.7)
        loose = GradientBoostingDemandModel.confidence_from_spread(0.30, 0.7)
        assert tight > loose
        assert GradientBoostingDemandModel.confidence_from_spread(0.0, 0.7) == 1.0

    def test_missing_feature_is_rejected_loudly(self, trained) -> None:
        """Train/serve skew is invisible without this check."""
        model, split, _ = trained
        with pytest.raises(FeatureVersionMismatch, match="absent from the input"):
            model.predict(split.test.drop(columns=["occupancy_rate"]))

    def test_extra_columns_are_harmless(self, trained) -> None:
        model, split, _ = trained
        model.predict(split.test.assign(some_new_column=1.0))

    def test_fitting_without_a_target_is_rejected(self, matrix) -> None:
        with pytest.raises(ValueError, match="no 'target_demand'"):
            GradientBoostingDemandModel().fit(matrix.drop(columns=[TARGET_COLUMN]))

    def test_non_finite_features_are_rejected(self, matrix) -> None:
        broken = matrix.copy()
        broken.loc[broken.index[0], "occupancy_rate"] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            GradientBoostingDemandModel().fit(broken)

    def test_predicting_before_fitting_is_an_error(self, matrix) -> None:
        with pytest.raises(RuntimeError, match="has not been fitted"):
            GradientBoostingDemandModel().predict(matrix)


class TestGradientBoostingQuality:
    def test_beats_the_predict_the_mean_baseline(self, trained) -> None:
        model, split, metrics = trained
        baseline = baseline_metrics(split.test[TARGET_COLUMN])
        assert metrics.mae < baseline.mae * 0.7
        assert metrics.r2 > 0.4

    def test_is_not_systematically_biased(self, trained) -> None:
        """Good MAE with a large bias means consistently over- or under-charging."""
        _, _, metrics = trained
        assert abs(metrics.bias) < 0.03

    def test_occupancy_is_among_the_strongest_features(self, trained) -> None:
        """The revenue-management story rests on this.

        Regression guard: a misaligned on-the-books join once left occupancy
        ranked below `weather_score`, and the headline metrics still looked
        respectable.
        """
        model, split, _ = trained
        ranked = model.permutation_importance(split.test, n_repeats=3)
        top = ranked.head(5)["feature"].tolist()
        assert "occupancy_rate" in top, ranked.head(8).to_string()

    def test_short_horizons_are_predicted_better_than_long_ones(self, trained) -> None:
        """Near check-in the booking curve has already resolved most of the
        answer; sixty days out it has not. A model that scores the same at both
        is not using the curve."""
        model, split, _ = trained
        by_horizon = model.evaluate_by(split.test, "days_to_checkin").set_index(
            "days_to_checkin"
        )
        assert by_horizon.loc[0, "mae"] < by_horizon.loc[60, "mae"]

    def test_per_group_evaluation_beats_the_baseline_everywhere(self, trained) -> None:
        """A good headline can hide a model that is useless on one segment."""
        model, split, _ = trained
        by_room = model.evaluate_by(split.test, "room_type")
        assert (by_room["mae"] < by_room["baseline_mae"]).all()


class TestGradientBoostingImportance:
    def test_impurity_importances_sum_to_one(self, trained) -> None:
        model, _, _ = trained
        assert model.feature_importance()["importance"].sum() == pytest.approx(1.0)

    def test_every_feature_is_listed(self, trained) -> None:
        model, _, _ = trained
        assert set(model.feature_importance()["feature"]) == set(FEATURE_COLUMNS)

    def test_permutation_importance_reports_a_spread(self, trained) -> None:
        model, split, _ = trained
        ranked = model.permutation_importance(split.test, n_repeats=3)
        assert {"feature", "importance", "std"} <= set(ranked.columns)
        assert (ranked["std"] >= 0).all()

    def test_permutation_ranking_is_sorted(self, trained) -> None:
        model, split, _ = trained
        importances = model.permutation_importance(split.test, n_repeats=3)["importance"]
        assert importances.is_monotonic_decreasing


class TestGradientBoostingPersistence:
    def test_round_trip_preserves_predictions_exactly(self, trained, tmp_path) -> None:
        """Unlike Prophet's sampled intervals, a tree ensemble is deterministic."""
        model, split, _ = trained
        path = model.save(tmp_path / "gbr.joblib")
        reloaded = GradientBoostingDemandModel.load(path)

        np.testing.assert_array_equal(
            model.predict(split.test), reloaded.predict(split.test)
        )

    def test_round_trip_preserves_the_metadata(self, trained, tmp_path) -> None:
        model, _, _ = trained
        reloaded = GradientBoostingDemandModel.load(model.save(tmp_path / "gbr.joblib"))
        assert reloaded.metadata.dataset_hash == model.metadata.dataset_hash
        assert reloaded.features == model.features

    def test_missing_artifact_gives_an_actionable_error(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="train_models"):
            GradientBoostingDemandModel.load(tmp_path / "absent.joblib")

    def test_a_prophet_bundle_is_not_a_gbr_artifact(self, tmp_path) -> None:
        import joblib

        path = tmp_path / "wrong.joblib"
        joblib.dump({"format": "prophet-bundle-v1"}, path)
        with pytest.raises(ValueError, match="not a Gradient Boosting artifact"):
            GradientBoostingDemandModel.load(path)

    def test_unfitted_model_cannot_be_saved(self, tmp_path) -> None:
        with pytest.raises(RuntimeError, match="has not been fitted"):
            GradientBoostingDemandModel().save(tmp_path / "empty.joblib")

    def test_artifact_format_tag_is_stable(self) -> None:
        assert ARTIFACT_FORMAT == "gbr-model-v1"


class TestGradientBoostingConfig:
    def test_sklearn_kwargs_are_accepted_by_the_estimator(self) -> None:
        from sklearn.ensemble import GradientBoostingRegressor

        GradientBoostingRegressor(**GradientBoostingConfig().to_sklearn())

    def test_depth_allows_three_way_interactions(self) -> None:
        """occupancy given lead time given season is a depth-3 question."""
        assert GradientBoostingConfig().max_depth >= 4

    def test_subsampling_is_on(self) -> None:
        assert 0 < GradientBoostingConfig().subsample < 1.0
