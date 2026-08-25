"""Demand forecasting with Prophet.

Prophet's job in this system is narrow and worth stating precisely: it forecasts
the **shape of demand over time** for a hotel and room type -- trend, weekly
rhythm, yearly season, holiday spikes -- for dates nobody has booked yet.

That is the complement to the Gradient Boosting model, which forecasts the
*level* of demand for a specific night given everything currently known about it
(on-the-books occupancy, the competitive set, pickup velocity). The two answer
different questions:

============================  ===========================================
Prophet                       Gradient Boosting
============================  ===========================================
Time series, one per series   Cross-sectional, one model for everything
Sees only the date            Sees 30 features about the night
Works 30+ days out            Best inside the booking window
Gives an uncertainty band     Gives a point estimate
============================  ===========================================

The pricing engine blends them (``MODEL_PROPHET_BLEND_WEIGHT``), which is why
both are trained on the *same* target -- realised demand as a fraction of
inventory -- so their outputs are directly comparable numbers rather than two
different quantities that happen to be averaged.

**Serialisation.** Prophet objects hold a compiled Stan backend and do not
pickle reliably across versions. Each fitted model is therefore serialised with
Prophet's own ``model_to_json`` and the *strings* are what joblib stores. The
artifact survives a Prophet upgrade; a naive ``joblib.dump(model)`` would not.

**One model per (hotel, room type).** A single pooled model cannot represent a
business hotel that empties at the weekend and a resort that fills, at the same
time. Thirty-two small models fit in under a minute and each one is
interpretable on its own.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from database.models import RoomType
from features.calendars import _load_holidays  # noqa: PLC2701 - intentional reuse
from models.metrics import RegressionMetrics, baseline_metrics, evaluate
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

def quiet_stan_logging() -> None:
    """Silence cmdstanpy's per-fit chatter.

    Setting this once at import is not enough: ``cmdstanpy`` sets its own logger
    to DEBUG when *it* is imported, and this module imports Prophet lazily
    inside :meth:`ProphetDemandForecaster.fit`. Whatever level we chose earlier
    is therefore overwritten before the first fit runs. Re-asserting it at the
    point of use is the only ordering that holds.

    Without this, a 32-series training run emits several hundred DEBUG lines
    (``TBB already found in load path``, ``cmd: where.exe tbb.dll``) and buries
    the metrics it exists to report.
    """
    for name in ("cmdstanpy", "prophet", "prophet.models", "prophet.forecaster"):
        logger_ = logging.getLogger(name)
        logger_.setLevel(logging.WARNING)
        logger_.propagate = False


quiet_stan_logging()

#: Forecast horizons the API exposes, per the specification.
DEFAULT_HORIZONS: Tuple[int, ...] = (7, 14, 30)

#: Shorter than this and there is not enough signal to fit even a weekly term.
MIN_TRAINING_DAYS = 90

#: Yearly seasonality is enabled only above this much history -- two full cycles.
#:
#: This threshold was measured, not guessed. On a 60-day holdout across four
#: series, with ~300 training days available:
#:
#:   yearly OFF -> MAE 0.066, 80% interval covers 87% of outcomes
#:   yearly ON  -> MAE 0.097, 80% interval covers 53% of outcomes
#:   baseline   -> MAE 0.113  (predict the mean)
#:
#: With less than one full cycle in the training window the yearly Fourier term
#: is unidentifiable: Prophet fits it to noise and then extrapolates that noise
#: into the forecast, which is worse than having no yearly term at all. Two
#: cycles is the point at which the term is separable from the trend.
MIN_DAYS_FOR_YEARLY = 730

#: Holiday regressors need the same treatment for the same reason: with one year
#: of data each holiday occurs exactly once, so its coefficient is fitted to a
#: single observation and is indistinguishable from that day's noise. Measured
#: cost of enabling them on one year of history: MAE 0.066 -> 0.068.
MIN_DAYS_FOR_HOLIDAYS = 730


@dataclass
class ProphetConfig:
    """Hyperparameters, with the reasoning for the non-obvious ones.

    Every default here was chosen by measurement on a 60-day holdout, not by
    copying a tutorial. The grid that produced them is reproduced in
    :data:`MIN_DAYS_FOR_YEARLY`.

    Attributes:
        changepoint_prior_scale: Trend flexibility. 0.05 measured best; 0.02 was
            too stiff to follow the slow drift in the data and 0.30 chased noise.
        seasonality_prior_scale: Seasonal amplitude. Generous, because the weekly
            swing between a business hotel's Wednesday and its Saturday is large
            and real.
        interval_width: Width of the uncertainty band. 0.80 rather than 0.95 --
            the pricing engine turns interval width into a confidence score, and
            a 95% band is so wide that every night looks equally uncertain. At
            0.80 the measured empirical coverage is 0.87, so the band is honest.
        seasonality_mode: Multiplicative. Demand swings scale with the level: a
            peak-season weekend adds more rooms than an off-season one.
        growth: Linear. Logistic is theoretically nicer -- occupancy has a hard
            ceiling -- but measured no better (MAE 0.067 vs 0.066) while adding a
            capacity parameter that has to be guessed. Forecasts are clipped to
            ``[0, cap]`` afterwards regardless, which gets the ceiling without
            the extra machinery.
        yearly_fourier_order: 6 rather than Prophet's 10. Even with enough
            history, ten harmonics on annual hotel demand fit holiday-week
            wiggles that the holiday regressors already explain.
    """

    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    holidays_prior_scale: float = 10.0
    interval_width: float = 0.80
    seasonality_mode: str = "multiplicative"
    growth: str = "linear"
    #: Ceiling used for clipping, and for logistic growth if selected: a
    #: multiple of observed peak demand.
    capacity_headroom: float = 1.15
    weekly_seasonality: bool = True
    #: ``None`` means "decide from the series length"; see MIN_DAYS_FOR_YEARLY.
    yearly_seasonality: Optional[bool] = None
    yearly_fourier_order: int = 6
    #: ``None`` means "decide from the series length"; see MIN_DAYS_FOR_HOLIDAYS.
    use_holidays: Optional[bool] = None
    daily_seasonality: bool = False
    n_changepoints: int = 20
    holiday_window: Tuple[int, int] = (-1, 1)
    mcmc_samples: int = 0  # MAP estimation; MCMC is minutes per series

    def as_dict(self) -> Dict[str, Any]:
        return {
            "changepoint_prior_scale": self.changepoint_prior_scale,
            "seasonality_prior_scale": self.seasonality_prior_scale,
            "holidays_prior_scale": self.holidays_prior_scale,
            "interval_width": self.interval_width,
            "seasonality_mode": self.seasonality_mode,
            "growth": self.growth,
            "n_changepoints": self.n_changepoints,
            "weekly_seasonality": self.weekly_seasonality,
            "yearly_fourier_order": self.yearly_fourier_order,
            "daily_seasonality": self.daily_seasonality,
            "capacity_headroom": self.capacity_headroom,
            "holiday_window": list(self.holiday_window),
        }


class InsufficientHistory(ValueError):
    """A series is too short to fit a seasonal model on."""


# --------------------------------------------------------------------------- #
# Holidays
# --------------------------------------------------------------------------- #


def holiday_frame(
    years: Iterable[int], window: Tuple[int, int] = (-1, 1)
) -> pd.DataFrame:
    """Prophet's holiday table, built from the project's own calendar.

    Reusing :mod:`features.calendars` rather than the ``holidays`` package keeps
    one definition of "Diwali" across the data generator, the feature pipeline
    and the forecaster. A holiday Prophet knows about but the features do not
    would put a spike in the forecast that nothing else in the system explains.

    ``lower_window``/``upper_window`` spread each holiday's effect over the days
    around it, because that is when the travel actually happens.
    """
    years = sorted(set(years))

    # Prophet requires one prior scale per holiday *name*, constant across every
    # occurrence -- it raises "does not have consistent prior scale
    # specification" otherwise. Our calendar allows a holiday's significance to
    # differ by year (Dussehra is 0.75 in 2024 and 0.80 afterwards), so the
    # scale is resolved per name first, taking the strongest year.
    strength: Dict[str, float] = {}
    for year in years:
        for holiday in _load_holidays(year).values():
            strength[holiday.name] = max(
                strength.get(holiday.name, 0.0), holiday.significance
            )

    rows: List[Dict[str, Any]] = []
    for year in years:
        for day, holiday in _load_holidays(year).items():
            rows.append(
                {
                    "holiday": holiday.name,
                    "ds": pd.Timestamp(day),
                    "lower_window": window[0],
                    "upper_window": window[1],
                    # Scaling the prior by significance stops Mahavir Jayanti
                    # from being fitted as hard as Diwali.
                    "prior_scale": max(1.0, 10.0 * strength[holiday.name]),
                }
            )
    return pd.DataFrame(rows).sort_values("ds").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Single-series forecaster
# --------------------------------------------------------------------------- #


@dataclass
class ForecastResult:
    """A forecast for one series, plus what produced it."""

    series_key: Tuple[str, str]
    frame: pd.DataFrame  # ds, yhat, yhat_lower, yhat_upper, trend
    horizon_days: int
    fitted_at: datetime

    def as_records(self) -> List[Dict[str, Any]]:
        """JSON-ready rows, for the API."""
        return [
            {
                "date": row.ds.date().isoformat(),
                "forecast": round(float(row.yhat), 4),
                "lower": round(float(row.yhat_lower), 4),
                "upper": round(float(row.yhat_upper), 4),
                "trend": round(float(row.trend), 4),
            }
            for row in self.frame.itertuples()
        ]

    def value_on(self, day: date) -> Optional[Dict[str, float]]:
        """The forecast for one date, or ``None`` if outside the horizon."""
        match = self.frame[self.frame["ds"] == pd.Timestamp(day)]
        if match.empty:
            return None
        row = match.iloc[0]
        return {
            "forecast": float(row["yhat"]),
            "lower": float(row["yhat_lower"]),
            "upper": float(row["yhat_upper"]),
            "trend": float(row["trend"]),
        }


class ProphetDemandForecaster:
    """Wraps Prophet for one ``(hotel, room type)`` demand series.

    Example::

        forecaster = ProphetDemandForecaster(("H001", "deluxe"))
        forecaster.fit(history)          # ds, y
        result = forecaster.forecast(30)
    """

    def __init__(
        self,
        series_key: Tuple[str, str],
        config: Optional[ProphetConfig] = None,
    ) -> None:
        self.series_key = series_key
        self.config = config or ProphetConfig()
        self.model: Any = None
        self.fitted_at: Optional[datetime] = None
        self.history_start: Optional[date] = None
        self.history_end: Optional[date] = None
        self.n_observations: int = 0
        self.yearly_enabled: bool = False
        self.holidays_enabled: bool = False
        self._cap: float = 1.0
        self._floor: float = 0.0

    # -- fitting ------------------------------------------------------------ #

    def fit(self, history: pd.DataFrame) -> "ProphetDemandForecaster":
        """Fit on a daily series.

        Args:
            history: Columns ``ds`` (date) and ``y`` (demand as a fraction of
                inventory). Gaps are filled by interpolation -- Prophet tolerates
                missing days, but the weekly term is cleaner on a complete
                calendar.

        Raises:
            InsufficientHistory: Fewer than :data:`MIN_TRAINING_DAYS` points.
        """
        from prophet import Prophet

        quiet_stan_logging()  # cmdstanpy resets its own level on import
        frame = self._prepare(history)

        if len(frame) < MIN_TRAINING_DAYS:
            raise InsufficientHistory(
                f"{self.series_key}: {len(frame)} observation(s), need at least "
                f"{MIN_TRAINING_DAYS} to fit weekly and yearly seasonality"
            )

        # Both of these terms are *disabled* on short history rather than fitted
        # badly. A term whose coefficient comes from one observation is not a
        # model of anything, and Prophet extrapolates it with full confidence.
        yearly = (
            self.config.yearly_seasonality
            if self.config.yearly_seasonality is not None
            else len(frame) >= MIN_DAYS_FOR_YEARLY
        )
        use_holidays = (
            self.config.use_holidays
            if self.config.use_holidays is not None
            else len(frame) >= MIN_DAYS_FOR_HOLIDAYS
        )
        if not yearly or not use_holidays:
            logger.debug(
                "%s: %d day(s) of history -> yearly=%s holidays=%s",
                self.series_key,
                len(frame),
                yearly,
                use_holidays,
            )

        holidays = (
            holiday_frame(
                range(frame["ds"].dt.year.min(), frame["ds"].dt.year.max() + 3),
                window=self.config.holiday_window,
            )
            if use_holidays
            else None
        )

        model = Prophet(
            growth=self.config.growth,
            changepoint_prior_scale=self.config.changepoint_prior_scale,
            seasonality_prior_scale=self.config.seasonality_prior_scale,
            holidays_prior_scale=self.config.holidays_prior_scale,
            interval_width=self.config.interval_width,
            seasonality_mode=self.config.seasonality_mode,
            weekly_seasonality=self.config.weekly_seasonality,
            # Added explicitly below when enabled, so the Fourier order is ours.
            yearly_seasonality=False,
            daily_seasonality=self.config.daily_seasonality,
            n_changepoints=min(self.config.n_changepoints, max(len(frame) // 10, 1)),
            mcmc_samples=self.config.mcmc_samples,
            holidays=holidays,
        )
        if yearly:
            model.add_seasonality(
                name="yearly",
                period=365.25,
                fourier_order=self.config.yearly_fourier_order,
            )
        self.yearly_enabled = yearly
        self.holidays_enabled = use_holidays

        # Prophet warns about seasonality it cannot identify; we have already
        # decided that above, so the warning is noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(frame)

        self.model = model
        self.fitted_at = datetime.now(timezone.utc)
        self.history_start = frame["ds"].min().date()
        self.history_end = frame["ds"].max().date()
        self.n_observations = len(frame)
        return self

    def _prepare(self, history: pd.DataFrame) -> pd.DataFrame:
        """Validate, sort, gap-fill and attach the logistic bounds."""
        missing = {"ds", "y"} - set(history.columns)
        if missing:
            raise ValueError(f"history is missing column(s): {sorted(missing)}")

        frame = history[["ds", "y"]].copy()
        frame["ds"] = pd.to_datetime(frame["ds"]).dt.normalize()
        frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
        frame = frame.dropna(subset=["ds"]).sort_values("ds")
        frame = frame.groupby("ds", as_index=False)["y"].mean()

        # Complete the calendar so "last week" always means seven rows.
        full = pd.DataFrame(
            {"ds": pd.date_range(frame["ds"].min(), frame["ds"].max(), freq="D")}
        )
        frame = full.merge(frame, on="ds", how="left")
        gaps = int(frame["y"].isna().sum())
        if gaps:
            logger.info("%s: interpolating %d missing day(s)", self.series_key, gaps)
        frame["y"] = frame["y"].interpolate(limit_direction="both")
        frame = frame.dropna(subset=["y"])

        # The ceiling is computed whatever the growth mode: logistic growth
        # needs it as a parameter, and linear growth needs it to clip forecasts.
        # Derived from the observed peak with headroom rather than fixed at 1.0,
        # because overbooking is real.
        self._cap = float(max(frame["y"].max() * self.config.capacity_headroom, 0.1))
        self._floor = 0.0

        if self.config.growth == "logistic":
            frame["cap"] = self._cap
            frame["floor"] = self._floor
        return frame

    # -- forecasting -------------------------------------------------------- #

    def forecast(self, horizon_days: int = 30) -> ForecastResult:
        """Forecast the next ``horizon_days`` days.

        Returns:
            Only the future rows -- the in-sample fit is not a forecast and
            including it is how backtests accidentally become in-sample scores.
        """
        self._require_fitted()

        future = self.model.make_future_dataframe(periods=horizon_days, freq="D")
        if self.config.growth == "logistic":
            future["cap"] = self._cap
            future["floor"] = self._floor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted = self.model.predict(future)

        columns = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
        cutoff = pd.Timestamp(self.history_end)
        future_only = predicted.loc[predicted["ds"] > cutoff, columns].copy()

        # Demand is a non-negative fraction; a negative lower bound is an
        # artefact of the additive error term, not a possible outcome.
        for column in ("yhat", "yhat_lower", "yhat_upper"):
            future_only[column] = future_only[column].clip(lower=0.0, upper=self._cap)

        return ForecastResult(
            series_key=self.series_key,
            frame=future_only.reset_index(drop=True),
            horizon_days=horizon_days,
            fitted_at=self.fitted_at,
        )

    def forecast_horizons(
        self, horizons: Sequence[int] = DEFAULT_HORIZONS
    ) -> Dict[int, ForecastResult]:
        """Forecast several horizons, computing the longest once and slicing it.

        Prophet's forecast for day 7 is identical whether the frame was built
        for 7 days or 30, so fitting once and slicing is both faster and
        guaranteed self-consistent.
        """
        longest = self.forecast(max(horizons))
        results: Dict[int, ForecastResult] = {}
        for horizon in sorted(horizons):
            results[horizon] = ForecastResult(
                series_key=self.series_key,
                frame=longest.frame.head(horizon).reset_index(drop=True),
                horizon_days=horizon,
                fitted_at=self.fitted_at,
            )
        return results

    def predict_for_dates(self, days: Sequence[date]) -> pd.DataFrame:
        """Forecast specific dates, including ones inside the training window."""
        self._require_fitted()
        future = pd.DataFrame({"ds": pd.to_datetime(sorted(set(days)))})
        if self.config.growth == "logistic":
            future["cap"] = self._cap
            future["floor"] = self._floor

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predicted = self.model.predict(future)

        columns = ["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]
        result = predicted[columns].copy()
        for column in ("yhat", "yhat_lower", "yhat_upper"):
            result[column] = result[column].clip(lower=0.0, upper=self._cap)
        return result

    # -- evaluation --------------------------------------------------------- #

    def backtest(
        self, history: pd.DataFrame, *, horizon_days: int = 30, folds: int = 3
    ) -> RegressionMetrics:
        """Rolling-origin backtest: fit on the past, score on the future.

        Each fold moves the cutoff forward by one horizon, refits from scratch,
        and scores the untouched window after it. No shuffling, no random split:
        a randomly split time series leaks tomorrow into yesterday's training
        set and reports a score the model could never achieve.

        Args:
            folds: Number of cutoffs to evaluate.

        Raises:
            InsufficientHistory: If the series cannot support the requested
                number of folds.
        """
        frame = self._prepare(history)
        needed = MIN_TRAINING_DAYS + horizon_days * folds
        if len(frame) < needed:
            raise InsufficientHistory(
                f"{self.series_key}: backtesting {folds} fold(s) of {horizon_days} "
                f"days needs {needed} observations, have {len(frame)}"
            )

        actuals: List[float] = []
        predictions: List[float] = []
        lowers: List[float] = []
        uppers: List[float] = []

        for fold in range(folds, 0, -1):
            cutoff_index = len(frame) - horizon_days * fold
            train = frame.iloc[:cutoff_index]
            test = frame.iloc[cutoff_index : cutoff_index + horizon_days]
            if test.empty:
                continue

            fold_model = ProphetDemandForecaster(self.series_key, self.config)
            fold_model.fit(train[["ds", "y"]])
            predicted = fold_model.predict_for_dates(test["ds"].dt.date.tolist())

            merged = test.merge(predicted, on="ds", how="inner")
            actuals.extend(merged["y"].tolist())
            predictions.extend(merged["yhat"].tolist())
            lowers.extend(merged["yhat_lower"].tolist())
            uppers.extend(merged["yhat_upper"].tolist())

        if not actuals:
            raise InsufficientHistory(f"{self.series_key}: backtest produced no folds")

        return evaluate(actuals, predictions, lower=lowers, upper=uppers)

    # -- serialisation ------------------------------------------------------ #

    def to_state(self) -> Dict[str, Any]:
        """Serialisable state, with the Prophet object as JSON not as a pickle."""
        from prophet.serialize import model_to_json

        self._require_fitted()
        return {
            "series_key": self.series_key,
            "config": self.config.as_dict(),
            "model_json": model_to_json(self.model),
            "fitted_at": self.fitted_at.isoformat() if self.fitted_at else None,
            "history_start": self.history_start.isoformat() if self.history_start else None,
            "history_end": self.history_end.isoformat() if self.history_end else None,
            "n_observations": self.n_observations,
            "yearly_enabled": self.yearly_enabled,
            "holidays_enabled": self.holidays_enabled,
            "cap": self._cap,
            "floor": self._floor,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "ProphetDemandForecaster":
        """Rebuild a forecaster from :meth:`to_state`."""
        from prophet.serialize import model_from_json

        forecaster = cls(tuple(state["series_key"]))
        forecaster.model = model_from_json(state["model_json"])
        forecaster.fitted_at = (
            datetime.fromisoformat(state["fitted_at"]) if state.get("fitted_at") else None
        )
        forecaster.history_start = (
            date.fromisoformat(state["history_start"]) if state.get("history_start") else None
        )
        forecaster.history_end = (
            date.fromisoformat(state["history_end"]) if state.get("history_end") else None
        )
        forecaster.n_observations = int(state.get("n_observations", 0))
        forecaster.yearly_enabled = bool(state.get("yearly_enabled", False))
        forecaster.holidays_enabled = bool(state.get("holidays_enabled", False))
        forecaster._cap = float(state.get("cap", 1.0))
        forecaster._floor = float(state.get("floor", 0.0))
        return forecaster

    def _require_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError(
                f"{self.series_key}: forecaster has not been fitted; call fit() first"
            )


# --------------------------------------------------------------------------- #
# Multi-series bundle
# --------------------------------------------------------------------------- #


@dataclass
class SeriesReport:
    """Per-series outcome of a training run, successes and failures alike."""

    series_key: Tuple[str, str]
    fitted: bool
    n_observations: int = 0
    metrics: Optional[RegressionMetrics] = None
    baseline: Optional[RegressionMetrics] = None
    error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hotel_id": self.series_key[0],
            "room_type": self.series_key[1],
            "fitted": self.fitted,
            "n_observations": self.n_observations,
            "metrics": self.metrics.as_dict() if self.metrics else None,
            "baseline_mae": self.baseline.mae if self.baseline else None,
            "error": self.error,
        }


class ProphetBundle:
    """A fitted forecaster per ``(hotel, room type)``, saved as one artifact.

    Example::

        bundle = ProphetBundle().fit_all(daily_demand)
        bundle.save(Path("models/artifacts/prophet_v1.joblib"))
        forecast = bundle.forecast("H001", RoomType.DELUXE, horizon_days=14)
    """

    def __init__(self, config: Optional[ProphetConfig] = None) -> None:
        self.config = config or ProphetConfig()
        self.models: Dict[Tuple[str, str], ProphetDemandForecaster] = {}
        self.reports: List[SeriesReport] = []
        self.trained_at: Optional[datetime] = None

    # -- training ----------------------------------------------------------- #

    @staticmethod
    def daily_demand(features: pd.DataFrame) -> pd.DataFrame:
        """Reshape the feature matrix into Prophet's ``ds``/``y`` per series.

        Prophet sees only the date and the realised demand -- deliberately. Its
        contribution is the calendar structure the Gradient Boosting model
        cannot express for dates with no bookings yet.
        """
        required = {"hotel_id", "room_type", "stay_date", "target_demand"}
        missing = required - set(features.columns)
        if missing:
            raise ValueError(f"feature frame is missing: {sorted(missing)}")

        frame = features[list(required)].copy()
        frame["ds"] = pd.to_datetime(frame["stay_date"]).dt.normalize()
        frame["room_type"] = frame["room_type"].map(
            lambda v: v.value if isinstance(v, RoomType) else str(v)
        )
        frame = frame.rename(columns={"target_demand": "y"})
        return (
            frame.dropna(subset=["y"])
            .groupby(["hotel_id", "room_type", "ds"], as_index=False)["y"]
            .mean()
            .sort_values(["hotel_id", "room_type", "ds"])
        )

    def fit_all(
        self,
        daily: pd.DataFrame,
        *,
        backtest_folds: int = 0,
        backtest_horizon: int = 30,
    ) -> "ProphetBundle":
        """Fit one model per series.

        A series that cannot be fitted is *recorded and skipped*, not fatal: one
        hotel with three weeks of history must not stop the other thirty-one
        models from training.

        Args:
            backtest_folds: Rolling-origin folds per series. Zero skips
                backtesting, which is what the fast path uses; the training
                script turns it on.
        """
        self.models.clear()
        self.reports.clear()

        for (hotel_id, room_type), group in daily.groupby(["hotel_id", "room_type"]):
            key = (str(hotel_id), str(room_type))
            history = group[["ds", "y"]].reset_index(drop=True)

            try:
                forecaster = ProphetDemandForecaster(key, self.config).fit(history)
            except (InsufficientHistory, ValueError) as exc:
                logger.warning("Skipping %s: %s", key, exc)
                self.reports.append(
                    SeriesReport(key, fitted=False, n_observations=len(history),
                                 error=str(exc))
                )
                continue
            except Exception as exc:  # pragma: no cover - Stan failures are rare
                logger.error("Fit failed for %s: %s: %s", key, type(exc).__name__, exc)
                self.reports.append(
                    SeriesReport(key, fitted=False, n_observations=len(history),
                                 error=f"{type(exc).__name__}: {exc}")
                )
                continue

            metrics: Optional[RegressionMetrics] = None
            if backtest_folds > 0:
                try:
                    metrics = forecaster.backtest(
                        history, horizon_days=backtest_horizon, folds=backtest_folds
                    )
                except InsufficientHistory as exc:
                    logger.warning("No backtest for %s: %s", key, exc)

            self.models[key] = forecaster
            self.reports.append(
                SeriesReport(
                    key,
                    fitted=True,
                    n_observations=forecaster.n_observations,
                    metrics=metrics,
                    baseline=baseline_metrics(history["y"]) if metrics else None,
                )
            )
            logger.info(
                "Fitted %s on %d observation(s)%s",
                key,
                forecaster.n_observations,
                f" | {metrics.summary()}" if metrics else "",
            )

        self.trained_at = datetime.now(timezone.utc)
        logger.info(
            "Prophet bundle: %d/%d series fitted",
            len(self.models),
            len(self.reports),
        )
        return self

    # -- inference ---------------------------------------------------------- #

    def has(self, hotel_id: str, room_type: Any) -> bool:
        return self._key(hotel_id, room_type) in self.models

    def forecast(
        self, hotel_id: str, room_type: Any, *, horizon_days: int = 30
    ) -> ForecastResult:
        """Forecast one series.

        Raises:
            KeyError: If no model was fitted for that series.
        """
        key = self._key(hotel_id, room_type)
        if key not in self.models:
            raise KeyError(
                f"no Prophet model for {key}; fitted series are "
                f"{sorted(self.models)[:5]}{'...' if len(self.models) > 5 else ''}"
            )
        return self.models[key].forecast(horizon_days)

    def forecast_range(
        self,
        hotel_id: str,
        room_type: Any,
        *,
        start: date,
        horizon_days: int = 30,
    ) -> ForecastResult:
        """Forecast ``horizon_days`` nights beginning at ``start``.

        Distinct from :meth:`forecast`, which continues from the *end of the
        training window*. Those two are the same thing only if the model was
        trained right up to today -- and it never is, because the training
        pipeline holds out the most recent sixty days.

        Serving "the next 7 nights" with :meth:`forecast` would therefore return
        dates two months in the past. This method asks for the dates the caller
        actually means.

        Raises:
            KeyError: If no model was fitted for that series.
        """
        key = self._key(hotel_id, room_type)
        forecaster = self.models.get(key)
        if forecaster is None:
            raise KeyError(f"no Prophet model for {key}")

        days = [start + timedelta(days=offset) for offset in range(horizon_days)]
        frame = forecaster.predict_for_dates(days)

        return ForecastResult(
            series_key=key,
            frame=frame.reset_index(drop=True),
            horizon_days=horizon_days,
            fitted_at=forecaster.fitted_at,
        )

    def demand_on(
        self, hotel_id: str, room_type: Any, day: date
    ) -> Optional[Dict[str, float]]:
        """The forecast for a single night, or ``None`` if unavailable.

        Returns ``None`` rather than raising: the pricing engine degrades to the
        Gradient Boosting prediction alone when Prophet has nothing to say, and
        an exception here would take a price request down over a missing series.
        """
        key = self._key(hotel_id, room_type)
        forecaster = self.models.get(key)
        if forecaster is None:
            return None
        try:
            predicted = forecaster.predict_for_dates([day])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Prophet prediction failed for %s on %s: %s", key, day, exc)
            return None
        if predicted.empty:
            return None
        row = predicted.iloc[0]
        return {
            "forecast": float(row["yhat"]),
            "lower": float(row["yhat_lower"]),
            "upper": float(row["yhat_upper"]),
            "trend": float(row["trend"]),
        }

    def evaluate_on(self, features: pd.DataFrame) -> Optional[RegressionMetrics]:
        """Score every fitted series against a held-out labelled frame.

        This is the number worth quoting, and it is deliberately computed on the
        *same* chronological holdout the Gradient Boosting model is scored on.
        The alternative -- reporting Prophet's internal rolling-origin backtest --
        measures a different thing on a different window, so the two models
        cannot be compared and the blend weight between them has no basis.

        Args:
            features: Rows with ``hotel_id``, ``room_type``, ``stay_date`` and
                ``target_demand``.

        Returns:
            Metrics across every series, or ``None`` if nothing could be scored.
        """
        daily = self.daily_demand(features)
        actuals: List[float] = []
        predictions: List[float] = []
        lowers: List[float] = []
        uppers: List[float] = []

        for (hotel_id, room_type), group in daily.groupby(["hotel_id", "room_type"]):
            forecaster = self.models.get((str(hotel_id), str(room_type)))
            if forecaster is None:
                continue
            predicted = forecaster.predict_for_dates(group["ds"].dt.date.tolist())
            merged = group.merge(predicted, on="ds", how="inner")
            actuals.extend(merged["y"].tolist())
            predictions.extend(merged["yhat"].tolist())
            lowers.extend(merged["yhat_lower"].tolist())
            uppers.extend(merged["yhat_upper"].tolist())

        if not actuals:
            logger.warning("Prophet could not be scored: no overlapping series")
            return None
        return evaluate(actuals, predictions, lower=lowers, upper=uppers)

    @staticmethod
    def _key(hotel_id: str, room_type: Any) -> Tuple[str, str]:
        value = room_type.value if isinstance(room_type, RoomType) else str(room_type)
        return (str(hotel_id), value)

    # -- reporting ---------------------------------------------------------- #

    def report_frame(self) -> pd.DataFrame:
        """Per-series training outcomes as a table."""
        return pd.DataFrame([r.as_dict() for r in self.reports])

    def aggregate_metrics(self) -> Optional[RegressionMetrics]:
        """Mean of the per-series backtest metrics, weighted by observations."""
        scored = [r for r in self.reports if r.metrics is not None]
        if not scored:
            return None

        weights = np.array([r.metrics.n for r in scored], dtype=float)
        def _weighted(attribute: str) -> float:
            values = np.array([getattr(r.metrics, attribute) for r in scored], dtype=float)
            usable = np.isfinite(values)
            if not usable.any():
                return float("nan")
            return float(np.average(values[usable], weights=weights[usable]))

        return RegressionMetrics(
            mae=_weighted("mae"),
            rmse=_weighted("rmse"),
            mape=_weighted("mape"),
            smape=_weighted("smape"),
            weighted_mape=_weighted("weighted_mape"),
            r2=_weighted("r2"),
            bias=_weighted("bias"),
            n=int(weights.sum()),
            mape_coverage=_weighted("mape_coverage"),
            interval_coverage=_weighted("interval_coverage"),
        )

    # -- persistence --------------------------------------------------------- #

    def save(self, path: Path) -> Path:
        """Write the bundle to disk."""
        import joblib

        if not self.models:
            raise RuntimeError("refusing to save a bundle with no fitted models")

        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format": "prophet-bundle-v1",
                "config": self.config.as_dict(),
                "trained_at": self.trained_at.isoformat() if self.trained_at else None,
                "series": {
                    "|".join(key): model.to_state()
                    for key, model in self.models.items()
                },
                "reports": [r.as_dict() for r in self.reports],
            },
            path,
            compress=3,
        )
        size_mb = path.stat().st_size / 1_048_576
        logger.info(
            "Saved %d Prophet model(s) to %s (%.1f MB)", len(self.models), path, size_mb
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "ProphetBundle":
        """Read a bundle back."""
        import joblib

        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing. Train first: python scripts/train_models.py"
            )

        payload = joblib.load(path)
        if payload.get("format") != "prophet-bundle-v1":
            raise ValueError(
                f"{path} is not a Prophet bundle (format={payload.get('format')!r})"
            )

        bundle = cls(ProphetConfig(**{
            k: v for k, v in payload["config"].items()
            if k in ProphetConfig.__dataclass_fields__ and k != "holiday_window"
        }))
        for flat_key, state in payload["series"].items():
            hotel_id, room_type = flat_key.split("|", 1)
            bundle.models[(hotel_id, room_type)] = ProphetDemandForecaster.from_state(
                state
            )
        bundle.trained_at = (
            datetime.fromisoformat(payload["trained_at"])
            if payload.get("trained_at")
            else None
        )
        logger.info("Loaded %d Prophet model(s) from %s", len(bundle.models), path)
        return bundle


__all__ = [
    "DEFAULT_HORIZONS",
    "MIN_TRAINING_DAYS",
    "ForecastResult",
    "InsufficientHistory",
    "ProphetBundle",
    "ProphetConfig",
    "ProphetDemandForecaster",
    "SeriesReport",
    "holiday_frame",
]
