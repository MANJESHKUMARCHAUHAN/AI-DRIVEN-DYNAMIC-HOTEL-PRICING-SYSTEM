"""Regression metrics, defined once and shared by both models.

Prophet and the Gradient Boosting regressor are evaluated with the same
functions so their numbers are comparable, and so "MAPE" means one thing across
the whole project rather than whatever each training script happened to
implement.

MAPE deserves the attention it gets here. The naive formula divides by the
actual value, which explodes when demand is near zero -- and a hotel with two
rooms sold on a wet Tuesday in the monsoon is exactly the kind of row this
dataset contains. Three responses, all offered:

* :func:`mape` masks out actuals below a floor and reports how many rows it
  dropped, so a headline MAPE can never quietly be computed over 60% of the data;
* :func:`smape` is symmetric and bounded at 200%, so it degrades instead of
  exploding;
* :func:`weighted_mape` weights by actual volume, which is the metric a revenue
  manager actually cares about -- being wrong about a full hotel matters more
  than being wrong about an empty one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np

#: Actual values below this are excluded from MAPE. Demand is a fraction of
#: inventory, so 0.02 is "one room in fifty" -- genuinely near zero.
MAPE_FLOOR = 0.02


def _as_arrays(
    actual: Sequence[float], predicted: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Coerce to aligned float arrays, dropping pairs with a missing value."""
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if a.shape != p.shape:
        raise ValueError(
            f"actual and predicted have different shapes: {a.shape} vs {p.shape}"
        )
    if a.size == 0:
        raise ValueError("cannot compute metrics over an empty series")

    finite = np.isfinite(a) & np.isfinite(p)
    if not finite.any():
        raise ValueError("no finite (actual, predicted) pairs to compare")
    return a[finite], p[finite]


def mae(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean absolute error, in the units of the target."""
    a, p = _as_arrays(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Root mean squared error. Punishes large misses harder than :func:`mae`."""
    a, p = _as_arrays(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(
    actual: Sequence[float], predicted: Sequence[float], *, floor: float = MAPE_FLOOR
) -> float:
    """Mean absolute percentage error over rows whose actual exceeds ``floor``.

    Returns ``nan`` when no row clears the floor, rather than a number computed
    from nothing.
    """
    a, p = _as_arrays(actual, predicted)
    usable = np.abs(a) >= floor
    if not usable.any():
        return float("nan")
    return float(np.mean(np.abs((a[usable] - p[usable]) / a[usable])) * 100.0)


def mape_coverage(
    actual: Sequence[float], *, floor: float = MAPE_FLOOR
) -> float:
    """Fraction of rows :func:`mape` was actually computed over."""
    a = np.asarray(actual, dtype=float).ravel()
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(np.abs(finite) >= floor))


def smape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Symmetric MAPE, bounded at 200%. Safe when actuals approach zero."""
    a, p = _as_arrays(actual, predicted)
    denominator = (np.abs(a) + np.abs(p)) / 2.0
    usable = denominator > 0
    if not usable.any():
        return 0.0
    return float(
        np.mean(np.abs(a[usable] - p[usable]) / denominator[usable]) * 100.0
    )


def weighted_mape(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Volume-weighted absolute percentage error.

    Equivalent to ``sum|a-p| / sum|a|``. This is the business metric: it asks
    "across all the demand there was, what fraction did we miss by", so a bad
    call on a sold-out night counts for more than one on an empty night.
    """
    a, p = _as_arrays(actual, predicted)
    total = np.sum(np.abs(a))
    if total == 0:
        return float("nan")
    return float(np.sum(np.abs(a - p)) / total * 100.0)


def r2(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Coefficient of determination.

    Negative means the model is worse than predicting the mean -- which is worth
    seeing, so it is not clipped.
    """
    a, p = _as_arrays(actual, predicted)
    total = np.sum((a - np.mean(a)) ** 2)
    if total == 0:
        return float("nan")
    return float(1.0 - np.sum((a - p) ** 2) / total)


def bias(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """Mean signed error. Positive means the model over-predicts.

    A model with excellent MAE and a large bias is systematically wrong in one
    direction, which for pricing means consistently over- or under-charging.
    """
    a, p = _as_arrays(actual, predicted)
    return float(np.mean(p - a))


def interval_coverage(
    actual: Sequence[float], lower: Sequence[float], upper: Sequence[float]
) -> float:
    """Fraction of actuals inside a prediction interval.

    A nominal 80% interval that covers 40% of outcomes is not an uncertainty
    estimate, it is decoration -- and the pricing engine derives its confidence
    score from interval width, so this number matters.
    """
    a = np.asarray(actual, dtype=float).ravel()
    lo = np.asarray(lower, dtype=float).ravel()
    hi = np.asarray(upper, dtype=float).ravel()
    finite = np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
    if not finite.any():
        return float("nan")
    return float(np.mean((a[finite] >= lo[finite]) & (a[finite] <= hi[finite])))


@dataclass(frozen=True)
class RegressionMetrics:
    """Every headline number for one evaluation, in one object."""

    mae: float
    rmse: float
    mape: float
    smape: float
    weighted_mape: float
    r2: float
    bias: float
    n: int
    mape_coverage: float
    interval_coverage: Optional[float] = None

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def summary(self) -> str:
        """One line, for a log or a training report."""
        return (
            f"n={self.n} MAE={self.mae:.4f} RMSE={self.rmse:.4f} "
            f"MAPE={self.mape:.2f}% wMAPE={self.weighted_mape:.2f}% "
            f"R2={self.r2:.3f} bias={self.bias:+.4f}"
        )


def evaluate(
    actual: Sequence[float],
    predicted: Sequence[float],
    *,
    lower: Optional[Sequence[float]] = None,
    upper: Optional[Sequence[float]] = None,
) -> RegressionMetrics:
    """Compute every metric at once.

    Args:
        lower: Optional interval lower bound; enables ``interval_coverage``.
        upper: Optional interval upper bound.
    """
    a, p = _as_arrays(actual, predicted)
    return RegressionMetrics(
        mae=mae(a, p),
        rmse=rmse(a, p),
        mape=mape(a, p),
        smape=smape(a, p),
        weighted_mape=weighted_mape(a, p),
        r2=r2(a, p),
        bias=bias(a, p),
        n=int(a.size),
        mape_coverage=mape_coverage(a),
        interval_coverage=(
            interval_coverage(actual, lower, upper)
            if lower is not None and upper is not None
            else None
        ),
    )


def baseline_metrics(
    actual: Sequence[float], *, naive_value: Optional[float] = None
) -> RegressionMetrics:
    """Metrics for the "predict the mean" baseline.

    Every reported model number should be read against this. A model that beats
    nothing is not a model, and printing the baseline next to it makes that
    impossible to miss.
    """
    a = np.asarray(actual, dtype=float).ravel()
    a = a[np.isfinite(a)]
    value = float(np.mean(a)) if naive_value is None else naive_value
    return evaluate(a, np.full_like(a, value))


__all__ = [
    "MAPE_FLOOR",
    "RegressionMetrics",
    "baseline_metrics",
    "bias",
    "evaluate",
    "interval_coverage",
    "mae",
    "mape",
    "mape_coverage",
    "r2",
    "rmse",
    "smape",
    "weighted_mape",
]
