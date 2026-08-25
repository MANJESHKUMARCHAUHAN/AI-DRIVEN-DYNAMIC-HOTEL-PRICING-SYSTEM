"""Model monitoring: is the model still the model we evaluated?

Three questions, in increasing order of how long they take to answer:

**Has the input distribution moved?** Population Stability Index, feature by
feature, comparing recent rows against the training window. This is the earliest
signal available -- it fires before accuracy degrades, because it fires when the
*world* changes rather than when the consequences show up.

**Have the predictions moved?** Cheaper still, and it catches things PSI misses:
a model that has started predicting 0.9 for everything has not necessarily seen
drifted inputs, it may simply be broken.

**Has accuracy degraded?** The question everyone asks first and the one that
answers last, because it needs the outcome -- which for a hotel arrives after
the stay date. By the time this metric moves, the prices were already wrong.

PSI thresholds are the industry-conventional ones and are stated here rather
than buried: below 0.10 is no meaningful shift, 0.10-0.25 is worth watching,
above 0.25 warrants retraining. They are conventions, not laws, which is why the
threshold is configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config import Settings, get_settings
from database.models import DemandFeature, Prediction, PricingDecision
from models.metrics import RegressionMetrics, evaluate
from monitoring.data_monitor import CheckResult, Severity
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

#: Conventional PSI bands.
PSI_NO_SHIFT = 0.10
PSI_MODERATE = 0.25

#: Bins used for the PSI histogram. Ten is the convention; more bins make the
#: index jumpy on small samples.
PSI_BINS = 10

#: Fewer rows than this and any drift number is noise.
MIN_SAMPLE = 50


def population_stability_index(
    reference: Sequence[float], current: Sequence[float], *, bins: int = PSI_BINS
) -> float:
    """PSI between a reference and a current sample.

    Bin edges come from the *reference* distribution's quantiles, which is the
    whole point: the question is "where do today's values fall relative to what
    the model was trained on", not "how do two arbitrary histograms compare".

    Empty bins are floored rather than dropped -- a category that has vanished
    entirely is the strongest possible drift signal, and dropping it would make
    the index *smaller*.
    """
    reference_array = np.asarray(reference, dtype=float)
    current_array = np.asarray(current, dtype=float)
    reference_array = reference_array[np.isfinite(reference_array)]
    current_array = current_array[np.isfinite(current_array)]

    if reference_array.size < MIN_SAMPLE or current_array.size < MIN_SAMPLE:
        return float("nan")

    quantiles = np.linspace(0, 100, bins + 1)
    edges = np.unique(np.percentile(reference_array, quantiles))
    if edges.size < 3:
        # A near-constant reference: PSI is undefined rather than zero.
        return float("nan")

    edges[0], edges[-1] = -np.inf, np.inf

    reference_share = np.histogram(reference_array, bins=edges)[0] / reference_array.size
    current_share = np.histogram(current_array, bins=edges)[0] / current_array.size

    floor = 1e-4
    reference_share = np.clip(reference_share, floor, None)
    current_share = np.clip(current_share, floor, None)

    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def psi_severity(value: float, threshold: float = PSI_MODERATE) -> Severity:
    """Map a PSI value onto a severity."""
    if not np.isfinite(value):
        return Severity.OK
    if value >= threshold:
        return Severity.CRITICAL
    if value >= PSI_NO_SHIFT:
        return Severity.WARNING
    return Severity.OK


@dataclass
class DriftResult:
    """PSI for one feature."""

    feature: str
    psi: float
    severity: Severity
    reference_mean: float
    current_mean: float
    reference_n: int
    current_n: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature,
            "psi": None if not np.isfinite(self.psi) else round(self.psi, 4),
            "severity": self.severity.value,
            "reference_mean": round(self.reference_mean, 4),
            "current_mean": round(self.current_mean, 4),
            "reference_n": self.reference_n,
            "current_n": self.current_n,
        }


@dataclass
class ModelHealthReport:
    """Drift, prediction behaviour, guardrail pressure and accuracy."""

    drift: List[DriftResult] = field(default_factory=list)
    checks: List[CheckResult] = field(default_factory=list)
    prediction_stats: Dict[str, Any] = field(default_factory=dict)
    price_stats: Dict[str, Any] = field(default_factory=dict)
    guardrail_counts: Dict[str, int] = field(default_factory=dict)
    accuracy: Optional[RegressionMetrics] = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def severity(self) -> Severity:
        severities = [c.severity for c in self.checks] + [d.severity for d in self.drift]
        if any(s is Severity.CRITICAL for s in severities):
            return Severity.CRITICAL
        if any(s is Severity.WARNING for s in severities):
            return Severity.WARNING
        return Severity.OK

    @property
    def drifted_features(self) -> List[DriftResult]:
        return [d for d in self.drift if d.severity is not Severity.OK]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "severity": self.severity.value,
            "drift": [d.as_dict() for d in self.drift],
            "checks": [c.as_dict() for c in self.checks],
            "prediction_stats": self.prediction_stats,
            "price_stats": self.price_stats,
            "guardrail_counts": self.guardrail_counts,
            "accuracy": self.accuracy.as_dict() if self.accuracy else None,
        }

    def summary(self) -> str:
        return (
            f"severity {self.severity.value}, "
            f"{len(self.drifted_features)}/{len(self.drift)} feature(s) drifting, "
            f"{sum(1 for c in self.checks if not c.passed)} check(s) failing"
        )


class ModelMonitor:
    """Compares recent behaviour against the training window.

    Example::

        with session_scope() as session:
            report = ModelMonitor().run(session)
    """

    #: Features worth watching. Not all thirty: PSI on a one-hot season flag is
    #: noise, and a report with thirty rows is a report nobody reads.
    WATCHED_FEATURES = (
        "occupancy_rate",
        "competitor_rate",
        "search_demand",
        "historical_demand",
        "current_room_price",
        "lead_time",
        "booking_count",
    )

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    # -- drift ----------------------------------------------------------------- #

    @staticmethod
    def seasonality_caveat(session: Session, window_days: int) -> Optional[CheckResult]:
        """Warn when the drift comparison cannot separate drift from season.

        This is the single most important caveat in the whole monitoring layer,
        and it is stated as a check rather than a comment because it changes how
        every PSI number below should be read.

        Comparing a recent 30-day window against a reference that spans a whole
        year will *always* show drift in a seasonal business: August in Goa
        genuinely does not look like January in Goa, and PSI cannot tell "the
        world changed" from "it is a different month". With at least two years
        of history the comparison can be made season-on-season; with less, the
        numbers are still useful as a relative ranking but a high PSI is not by
        itself grounds to retrain.
        """
        span = session.execute(
            select(
                func.min(DemandFeature.stay_date), func.max(DemandFeature.stay_date)
            )
        ).one()

        if span[0] is None or span[1] is None:
            return None

        history_days = (span[1] - span[0]).days
        if history_days >= 730:
            return None

        return CheckResult(
            "drift_seasonality_caveat",
            Severity.WARNING,
            f"only {history_days} days of history, so the {window_days}-day drift "
            f"window is compared against a reference covering different seasons. "
            f"In a seasonal business that shows as drift even when nothing has "
            f"changed -- read the PSI values below as a ranking, not as a "
            f"retraining trigger.",
            value=float(history_days),
            threshold=730,
        )

    def feature_drift(
        self, session: Session, *, window_days: Optional[int] = None
    ) -> List[DriftResult]:
        """PSI for each watched feature: recent rows against everything older.

        See :meth:`seasonality_caveat` for why these numbers need reading with
        care on a dataset shorter than two years.
        """
        window_days = window_days or self.settings.monitoring.metrics_window_days
        cutoff = date.today() - timedelta(days=window_days)
        threshold = self.settings.monitoring.drift_psi_threshold

        columns = [getattr(DemandFeature, name) for name in self.WATCHED_FEATURES]

        def _frame(recent: bool) -> pd.DataFrame:
            condition = (
                DemandFeature.stay_date >= cutoff
                if recent
                else DemandFeature.stay_date < cutoff
            )
            rows = session.execute(
                select(*columns).where(
                    condition, DemandFeature.feature_version.is_not(None)
                )
            ).all()
            return pd.DataFrame(rows, columns=list(self.WATCHED_FEATURES))

        reference = _frame(recent=False)
        current = _frame(recent=True)

        if reference.empty or current.empty:
            logger.info(
                "Not enough history to measure drift (reference=%d, current=%d)",
                len(reference),
                len(current),
            )
            return []

        results: List[DriftResult] = []
        for name in self.WATCHED_FEATURES:
            reference_values = pd.to_numeric(reference[name], errors="coerce").dropna()
            current_values = pd.to_numeric(current[name], errors="coerce").dropna()

            psi = population_stability_index(reference_values, current_values)
            severity = psi_severity(psi, threshold)

            result = DriftResult(
                feature=name,
                psi=psi,
                severity=severity,
                reference_mean=float(reference_values.mean()) if len(reference_values) else float("nan"),
                current_mean=float(current_values.mean()) if len(current_values) else float("nan"),
                reference_n=len(reference_values),
                current_n=len(current_values),
            )
            if severity is not Severity.OK:
                logger.warning(
                    "DRIFT %s: PSI %.3f (%s) -- mean moved %.3f -> %.3f",
                    name,
                    psi,
                    severity.value,
                    result.reference_mean,
                    result.current_mean,
                )
            results.append(result)

        return results

    # -- prediction behaviour --------------------------------------------------- #

    def prediction_health(self, session: Session) -> tuple[Dict[str, Any], List[CheckResult]]:
        """Distribution of served predictions, and whether it looks sane."""
        rows = session.execute(
            select(
                Prediction.blended_demand, Prediction.confidence, Prediction.latency_ms
            ).order_by(Prediction.created_at.desc()).limit(5_000)
        ).all()

        if not rows:
            return {}, [
                CheckResult(
                    "prediction_volume",
                    Severity.WARNING,
                    "no predictions have been served yet",
                )
            ]

        frame = pd.DataFrame(rows, columns=["demand", "confidence", "latency_ms"])
        demand = pd.to_numeric(frame["demand"], errors="coerce").dropna()

        stats = {
            "n": int(len(frame)),
            "demand_mean": round(float(demand.mean()), 4),
            "demand_std": round(float(demand.std()), 4),
            "demand_min": round(float(demand.min()), 4),
            "demand_max": round(float(demand.max()), 4),
            "confidence_mean": round(float(pd.to_numeric(frame["confidence"]).mean()), 4),
            "latency_p95_ms": round(
                float(pd.to_numeric(frame["latency_ms"], errors="coerce").dropna().quantile(0.95)),
                2,
            )
            if frame["latency_ms"].notna().any()
            else None,
            "values": demand.tolist()[:2_000],
        }

        checks: List[CheckResult] = []

        # Below a usable sample, every distribution statistic is noise. Two
        # predictions from a smoke test have a standard deviation near zero,
        # which is not evidence that the model returns a constant -- and a
        # monitor that cries wolf on an idle service gets muted.
        if len(demand) < MIN_SAMPLE:
            checks.append(
                CheckResult(
                    "prediction_volume",
                    Severity.OK,
                    f"only {len(demand)} prediction(s) served; too few to judge the "
                    f"distribution (need {MIN_SAMPLE})",
                    value=float(len(demand)),
                    threshold=MIN_SAMPLE,
                )
            )
            return stats, checks

        # A collapsed distribution is the classic sign of a model serving a
        # constant -- a broken artifact, a missing feature, an all-null input.
        if stats["demand_std"] < 0.02:
            checks.append(
                CheckResult(
                    "prediction_variance",
                    Severity.CRITICAL,
                    f"predicted demand has almost no spread (std {stats['demand_std']:.4f}); "
                    f"the model may be returning a constant",
                    value=stats["demand_std"],
                    threshold=0.02,
                )
            )
        else:
            checks.append(
                CheckResult(
                    "prediction_variance",
                    Severity.OK,
                    f"predicted demand spread is {stats['demand_std']:.3f}",
                    value=stats["demand_std"],
                )
            )

        sigma = self.settings.monitoring.prediction_sigma_threshold
        outliers = demand[(demand - demand.mean()).abs() > sigma * max(demand.std(), 1e-9)]
        share = len(outliers) / len(demand)
        checks.append(
            CheckResult(
                "prediction_outliers",
                Severity.WARNING if share > 0.05 else Severity.OK,
                f"{share:.1%} of predictions are beyond {sigma:g} sigma",
                value=round(share, 4),
                threshold=0.05,
            )
        )

        if stats["confidence_mean"] < 0.4:
            checks.append(
                CheckResult(
                    "prediction_confidence",
                    Severity.WARNING,
                    f"mean confidence is {stats['confidence_mean']:.0%}; the models "
                    f"are unsure or disagreeing, so prices are barely moving off base",
                    value=stats["confidence_mean"],
                    threshold=0.4,
                )
            )

        return stats, checks

    # -- pricing behaviour ------------------------------------------------------ #

    def pricing_health(
        self, session: Session
    ) -> tuple[Dict[str, Any], Dict[str, int], List[CheckResult]]:
        """Price distribution and guardrail pressure.

        A guardrail firing occasionally is the system working. A guardrail
        firing on most decisions means the model wants prices the business will
        not allow, which is a retuning signal rather than a success -- and it is
        only visible if somebody counts.
        """
        rows = session.execute(
            select(
                PricingDecision.final_recommended_price,
                PricingDecision.raw_recommended_price,
                PricingDecision.price_change_percent,
                PricingDecision.guardrails_applied,
            ).order_by(PricingDecision.created_at.desc()).limit(5_000)
        ).all()

        if not rows:
            return {}, {}, [
                CheckResult(
                    "pricing_volume",
                    Severity.WARNING,
                    "no pricing decisions have been recorded yet",
                )
            ]

        frame = pd.DataFrame(
            rows, columns=["final", "raw", "change_percent", "guardrails"]
        )
        final = pd.to_numeric(frame["final"], errors="coerce").dropna()

        counts: Dict[str, int] = {}
        for applied in frame["guardrails"]:
            for rule in applied or []:
                counts[rule] = counts.get(rule, 0) + 1

        clamped_share = float(frame["guardrails"].apply(bool).mean())

        stats = {
            "n": int(len(frame)),
            "price_mean": round(float(final.mean()), 2),
            "price_min": round(float(final.min()), 2),
            "price_max": round(float(final.max()), 2),
            "clamped_share": round(clamped_share, 4),
            "mean_change_percent": round(
                float(pd.to_numeric(frame["change_percent"], errors="coerce").mean()), 2
            ),
            "values": final.tolist()[:2_000],
        }

        checks: List[CheckResult] = []
        if clamped_share > 0.50:
            checks.append(
                CheckResult(
                    "guardrail_pressure",
                    Severity.WARNING,
                    f"guardrails changed {clamped_share:.0%} of prices; the model is "
                    f"consistently asking for prices the business will not allow",
                    value=round(clamped_share, 4),
                    threshold=0.50,
                    detail=counts,
                )
            )
        else:
            checks.append(
                CheckResult(
                    "guardrail_pressure",
                    Severity.OK,
                    f"guardrails changed {clamped_share:.0%} of prices",
                    value=round(clamped_share, 4),
                    detail=counts,
                )
            )

        limits = self.settings.pricing
        at_limit = float(
            ((final <= limits.min_price) | (final >= limits.max_price)).mean()
        )
        if at_limit > 0.10:
            checks.append(
                CheckResult(
                    "absolute_limits",
                    Severity.WARNING,
                    f"{at_limit:.0%} of prices are pinned to MIN_PRICE or MAX_PRICE; "
                    f"the limits may be miscalibrated for this estate",
                    value=round(at_limit, 4),
                    threshold=0.10,
                )
            )

        return stats, counts, checks

    # -- accuracy ---------------------------------------------------------------- #

    def realised_accuracy(self, session: Session) -> Optional[RegressionMetrics]:
        """Compare served predictions against what actually happened.

        The truest measure and the slowest: a prediction for next month cannot
        be scored until next month. Only nights that have completed count.
        """
        rows = session.execute(
            select(Prediction.blended_demand, DemandFeature.target_demand)
            .join(
                DemandFeature,
                (Prediction.hotel_id == DemandFeature.hotel_id)
                & (Prediction.room_type == DemandFeature.room_type)
                & (Prediction.check_in_date == DemandFeature.stay_date),
            )
            .where(
                DemandFeature.target_demand.is_not(None),
                Prediction.check_in_date < date.today(),
            )
        ).all()

        if len(rows) < MIN_SAMPLE:
            logger.info(
                "Only %d completed prediction(s); too few to score realised accuracy",
                len(rows),
            )
            return None

        predicted = [float(r[0]) for r in rows]
        actual = [float(r[1]) for r in rows]
        metrics = evaluate(actual, predicted)
        logger.info("Realised accuracy on %d completed night(s): %s", len(rows), metrics.summary())
        return metrics

    # -- orchestration ------------------------------------------------------------ #

    def run(self, session: Session) -> ModelHealthReport:
        """Run every model check and return the report."""
        window_days = self.settings.monitoring.metrics_window_days
        drift = self.feature_drift(session, window_days=window_days)
        prediction_stats, prediction_checks = self.prediction_health(session)
        price_stats, guardrail_counts, price_checks = self.pricing_health(session)
        accuracy = self.realised_accuracy(session)

        checks = prediction_checks + price_checks
        caveat = self.seasonality_caveat(session, window_days)
        if caveat is not None and drift:
            caveat.log()
            checks.insert(0, caveat)

        report = ModelHealthReport(
            drift=drift,
            checks=checks,
            prediction_stats=prediction_stats,
            price_stats=price_stats,
            guardrail_counts=guardrail_counts,
            accuracy=accuracy,
        )
        logger.info("Model health: %s", report.summary())
        return report


__all__ = [
    "MIN_SAMPLE",
    "PSI_BINS",
    "PSI_MODERATE",
    "PSI_NO_SHIFT",
    "DriftResult",
    "ModelHealthReport",
    "ModelMonitor",
    "population_stability_index",
    "psi_severity",
]
