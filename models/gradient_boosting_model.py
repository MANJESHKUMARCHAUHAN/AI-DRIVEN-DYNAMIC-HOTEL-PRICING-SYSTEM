"""Demand prediction with Gradient Boosting.

Where Prophet extrapolates the *calendar* -- what a Tuesday in November usually
looks like -- this model reads the *situation*: how full the hotel already is,
how fast rooms are moving, where the competitive set has priced, how far out we
are. It is the model that knows the difference between a night at 80% occupancy
thirty days out (about to sell out, raise the price) and 80% on the day itself
(this is as good as it gets, hold).

Three decisions define it:

**The split is chronological, never random.** This is a time-series panel.
``train_test_split(shuffle=True)`` would put next Tuesday in the training set
and last Tuesday in the test set, and report a score the model can never achieve
in production. :func:`time_based_split` holds out the most recent window, which
is the only split that answers the question we actually have -- "trained on the
past, how does it do on the future?"

**Importances are reported two ways.** scikit-learn's built-in importances are
impurity-based, which systematically favours continuous, high-cardinality
features over binary ones: ``competitor_rate`` will always look more important
than ``is_weekend`` partly because it has more places to split. Permutation
importance, measured on the *holdout*, asks a better question -- how much worse
does the model get if this feature is scrambled -- so both are computed and the
permutation one is what the docs quote.

**Uncertainty comes from the residuals.** A point estimate with no error bar is
not usable by a pricing engine that has to decide how much to trust the number.
The holdout residual standard deviation is stored with the artifact and served
alongside every prediction.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from features.feature_engineering import FEATURE_COLUMNS, FEATURE_VERSION, TARGET_COLUMN
from features.feature_store import FeatureVersionMismatch
from models.metrics import RegressionMetrics, baseline_metrics, evaluate
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

#: Days held out for evaluation. Two months covers a full weekly cycle many
#: times over while leaving ten months to train on.
DEFAULT_TEST_DAYS = 60

#: Artifact format tag, checked on load so a Prophet bundle cannot be loaded
#: here and produce nonsense.
ARTIFACT_FORMAT = "gbr-model-v1"


@dataclass
class GradientBoostingConfig:
    """Hyperparameters.

    The defaults were selected on a chronological validation split, not copied
    from a tutorial -- see ``docs/ml_pipeline.md`` for the grid. The three that
    matter:

    Attributes:
        n_estimators: With ``learning_rate=0.05`` this is the main capacity
            knob. Early stopping means overshooting costs time, not accuracy.
        max_depth: 4, not the default 3. The signal here is interactional --
            occupancy *given* lead time, price *relative to* the market -- and
            depth 3 cannot express a three-way interaction.
        subsample: 0.8 makes this stochastic gradient boosting. Each tree sees a
            different 80% of rows, which decorrelates them and is worth more
            than the small variance increase.
        n_iter_no_change: Early stopping on an internal validation slice. Note
            scikit-learn takes that slice *randomly*, which is acceptable only
            because it is used to decide when to stop rather than to report a
            score -- the reported score comes from the chronological holdout.
    """

    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 4
    min_samples_leaf: int = 20
    min_samples_split: int = 20
    subsample: float = 0.8
    max_features: Optional[str] = "sqrt"
    loss: str = "squared_error"
    random_state: int = 42
    validation_fraction: float = 0.15
    n_iter_no_change: Optional[int] = 25
    tol: float = 1e-4

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_sklearn(self) -> Dict[str, Any]:
        """Keyword arguments for ``GradientBoostingRegressor``."""
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split,
            "subsample": self.subsample,
            "max_features": self.max_features,
            "loss": self.loss,
            "random_state": self.random_state,
            "validation_fraction": self.validation_fraction,
            "n_iter_no_change": self.n_iter_no_change,
            "tol": self.tol,
        }


@dataclass
class TrainingSplit:
    """A chronological train/test split, and the cutoff that produced it."""

    train: pd.DataFrame
    test: pd.DataFrame
    cutoff: date

    def describe(self) -> str:
        return (
            f"train {len(self.train):,} rows up to {self.cutoff}, "
            f"test {len(self.test):,} rows after"
        )


def time_based_split(
    frame: pd.DataFrame,
    *,
    test_days: int = DEFAULT_TEST_DAYS,
    date_column: str = "stay_date",
) -> TrainingSplit:
    """Hold out the most recent ``test_days`` of stay dates.

    Raises:
        ValueError: If either side of the split would be empty, which usually
            means the history is shorter than the requested holdout.
    """
    if date_column not in frame.columns:
        raise ValueError(f"frame has no {date_column!r} column to split on")

    dates = pd.to_datetime(frame[date_column])
    cutoff = dates.max() - pd.Timedelta(days=test_days)

    train = frame[dates <= cutoff]
    test = frame[dates > cutoff]

    if train.empty or test.empty:
        raise ValueError(
            f"a {test_days}-day holdout leaves {len(train)} training and "
            f"{len(test)} test row(s); the history spans only "
            f"{(dates.max() - dates.min()).days} days"
        )

    split = TrainingSplit(
        train.reset_index(drop=True), test.reset_index(drop=True), cutoff.date()
    )
    logger.info("Chronological split: %s", split.describe())
    return split


def dataset_hash(features: pd.DataFrame, target: pd.Series) -> str:
    """Stable hash of the exact training data.

    Recorded with every model so "same data, same hyperparameters" can be
    verified to give the same model, rather than assumed.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(features.to_numpy(dtype=float)).tobytes())
    digest.update(np.ascontiguousarray(target.to_numpy(dtype=float)).tobytes())
    digest.update(",".join(features.columns).encode())
    return digest.hexdigest()


@dataclass
class ModelMetadata:
    """Everything needed to explain, audit or reproduce a trained model."""

    version: str
    feature_version: str
    features: List[str]
    hyperparameters: Dict[str, Any]
    trained_at: str
    n_train: int
    n_test: int
    train_start: Optional[str] = None
    train_end: Optional[str] = None
    test_start: Optional[str] = None
    test_end: Optional[str] = None
    dataset_hash: Optional[str] = None
    #: Standard deviation of holdout residuals -- the served uncertainty.
    residual_std: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    baseline_metrics: Dict[str, Any] = field(default_factory=dict)
    n_estimators_used: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GradientBoostingDemandModel:
    """Predicts realised demand for a night from its feature vector.

    Example::

        model = GradientBoostingDemandModel()
        model.fit(split.train)
        metrics = model.evaluate(split.test)
        model.save(Path("models/artifacts/gbr_v1.joblib"))
    """

    def __init__(
        self,
        config: Optional[GradientBoostingConfig] = None,
        *,
        version: str = "v1",
        features: Sequence[str] = FEATURE_COLUMNS,
    ) -> None:
        self.config = config or GradientBoostingConfig()
        self.version = version
        self.features: List[str] = list(features)
        self.model: Any = None
        self.metadata: Optional[ModelMetadata] = None

    # -- training ----------------------------------------------------------- #

    def fit(
        self,
        train: pd.DataFrame,
        *,
        test: Optional[pd.DataFrame] = None,
    ) -> "GradientBoostingDemandModel":
        """Fit on a feature matrix.

        Args:
            train: Rows containing :data:`FEATURE_COLUMNS` and the target.
            test: Optional holdout. When supplied, its metrics and residual
                spread are recorded in the metadata -- which is what makes the
                saved artifact self-describing rather than just a set of weights.
        """
        from sklearn.ensemble import GradientBoostingRegressor

        X, y = self._split_xy(train)

        model = GradientBoostingRegressor(**self.config.to_sklearn())
        model.fit(X, y)
        self.model = model

        used = int(getattr(model, "n_estimators_", self.config.n_estimators))
        if used < self.config.n_estimators:
            logger.info(
                "Early stopping used %d of %d trees", used, self.config.n_estimators
            )

        self.metadata = ModelMetadata(
            version=self.version,
            feature_version=FEATURE_VERSION,
            features=list(self.features),
            hyperparameters=self.config.as_dict(),
            trained_at=datetime.now(timezone.utc).isoformat(),
            n_train=len(X),
            n_test=len(test) if test is not None else 0,
            dataset_hash=dataset_hash(X, y),
            n_estimators_used=used,
        )
        self._record_window(train, "train")

        if test is not None and not test.empty:
            self._record_window(test, "test")
            metrics = self.evaluate(test)
            self.metadata.metrics = metrics.as_dict()
            self.metadata.baseline_metrics = baseline_metrics(
                test[TARGET_COLUMN]
            ).as_dict()

            residuals = test[TARGET_COLUMN].to_numpy(dtype=float) - self.predict(test)
            self.metadata.residual_std = float(np.std(residuals))

            logger.info("GBR holdout | %s", metrics.summary())

        return self

    def _record_window(self, frame: pd.DataFrame, prefix: str) -> None:
        if self.metadata is None or "stay_date" not in frame.columns:
            return
        dates = pd.to_datetime(frame["stay_date"])
        setattr(self.metadata, f"{prefix}_start", str(dates.min().date()))
        setattr(self.metadata, f"{prefix}_end", str(dates.max().date()))

    def _split_xy(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract the feature block and the target, checking the contract."""
        missing = [c for c in self.features if c not in frame.columns]
        if missing:
            raise ValueError(f"frame is missing feature column(s): {missing}")
        if TARGET_COLUMN not in frame.columns:
            raise ValueError(f"frame has no {TARGET_COLUMN!r} column")

        usable = frame[frame[TARGET_COLUMN].notna()]
        dropped = len(frame) - len(usable)
        if dropped:
            logger.info("Dropped %d row(s) with no target", dropped)
        if usable.empty:
            raise ValueError("no rows with a target to train on")

        X = usable[self.features].astype(float)
        y = usable[TARGET_COLUMN].astype(float)

        if not np.isfinite(X.to_numpy()).all():
            raise ValueError(
                "feature matrix contains non-finite values; the feature pipeline "
                "should have rejected them"
            )
        return X, y

    # -- inference ---------------------------------------------------------- #

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict demand for a feature matrix.

        Output is clipped at zero: a negative demand is an arithmetic artefact
        of the ensemble, not a possible outcome, and a downstream price
        adjustment computed from it would move the wrong way.
        """
        self._require_fitted()
        self.validate_features(frame.columns)
        X = frame[self.features].astype(float)
        return np.clip(self.model.predict(X), 0.0, None)

    def predict_one(self, row: pd.DataFrame) -> Dict[str, float]:
        """Predict a single row, with the uncertainty band from training.

        Returns:
            ``demand``, ``lower``, ``upper`` and ``confidence``. The band is one
            residual standard deviation either side; ``confidence`` maps that
            spread onto 0-1 so the pricing engine has a single number to weigh.
        """
        prediction = float(self.predict(row)[0])
        spread = float(self.metadata.residual_std or 0.0) if self.metadata else 0.0
        return {
            "demand": prediction,
            "lower": max(prediction - spread, 0.0),
            "upper": prediction + spread,
            "confidence": self.confidence_from_spread(spread, prediction),
        }

    @staticmethod
    def confidence_from_spread(spread: float, prediction: float) -> float:
        """Map a residual spread onto a 0-1 confidence.

        A spread of zero is total confidence; a spread as large as the
        prediction itself is none. Deliberately simple and monotone -- an
        elaborate calibration here would imply a precision the residuals do not
        support.
        """
        if prediction <= 0:
            return 0.0
        return float(np.clip(1.0 - spread / max(prediction, 1e-6), 0.0, 1.0))

    def validate_features(self, columns: Sequence[str]) -> None:
        """Assert the caller's frame carries what the model was trained on.

        Raises:
            FeatureVersionMismatch: If a trained-on feature is absent. Extra
                columns are fine -- the model selects by name -- but a *missing*
                one means the pipeline and the artifact have diverged.
        """
        missing = [c for c in self.features if c not in set(columns)]
        if missing:
            raise FeatureVersionMismatch(
                f"model {self.version} was trained on {len(self.features)} features "
                f"and {len(missing)} are absent from the input: {missing[:5]}"
                f"{'...' if len(missing) > 5 else ''}. Rebuild features or retrain."
            )

    # -- evaluation --------------------------------------------------------- #

    def evaluate(self, frame: pd.DataFrame) -> RegressionMetrics:
        """Score the model on a labelled frame."""
        self._require_fitted()
        X, y = self._split_xy(frame)
        return evaluate(y, np.clip(self.model.predict(X), 0.0, None))

    def evaluate_by(
        self, frame: pd.DataFrame, column: str
    ) -> pd.DataFrame:
        """Score per group -- per hotel, per room type, per horizon.

        A good headline number can hide a model that is excellent on the four
        big hotels and useless on the four small ones, and it is the small ones
        whose prices will look absurd.
        """
        self._require_fitted()
        rows: List[Dict[str, Any]] = []
        for value, group in frame.groupby(column):
            if group[TARGET_COLUMN].notna().sum() < 5:
                continue
            metrics = self.evaluate(group)
            rows.append(
                {
                    column: value,
                    "n": metrics.n,
                    "mae": round(metrics.mae, 4),
                    "rmse": round(metrics.rmse, 4),
                    "mape": round(metrics.mape, 2),
                    "r2": round(metrics.r2, 3),
                    "bias": round(metrics.bias, 4),
                    "baseline_mae": round(baseline_metrics(group[TARGET_COLUMN]).mae, 4),
                }
            )
        return pd.DataFrame(rows).sort_values("mae", ascending=False)

    # -- interpretation ----------------------------------------------------- #

    def feature_importance(self) -> pd.DataFrame:
        """Impurity-based importances, as scikit-learn computes them.

        Biased towards continuous, high-cardinality features: a column with a
        hundred distinct values simply offers more places to split than a binary
        flag. Use :meth:`permutation_importance` for the honest ranking.
        """
        self._require_fitted()
        return (
            pd.DataFrame(
                {
                    "feature": self.features,
                    "importance": self.model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def permutation_importance(
        self, frame: pd.DataFrame, *, n_repeats: int = 10, random_state: int = 42
    ) -> pd.DataFrame:
        """Holdout permutation importance: the ranking worth quoting.

        Scrambles one feature at a time and measures how much worse the model
        gets. Unbiased with respect to cardinality, measured on data the model
        has never seen, and directly interpretable as "how much do we lose
        without this".
        """
        from sklearn.inspection import permutation_importance as sk_permutation

        self._require_fitted()
        X, y = self._split_xy(frame)
        result = sk_permutation(
            self.model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="neg_mean_absolute_error",
        )
        return (
            pd.DataFrame(
                {
                    "feature": self.features,
                    "importance": result.importances_mean,
                    "std": result.importances_std,
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # -- persistence -------------------------------------------------------- #

    def save(self, path: Path) -> Path:
        """Write the model, its feature list and its metadata as one artifact."""
        import joblib

        self._require_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format": ARTIFACT_FORMAT,
                "version": self.version,
                "model": self.model,
                "features": self.features,
                "config": self.config.as_dict(),
                "metadata": self.metadata.as_dict() if self.metadata else {},
            },
            path,
            compress=3,
        )
        logger.info(
            "Saved GBR %s to %s (%.2f MB)",
            self.version,
            path,
            path.stat().st_size / 1_048_576,
        )
        return path

    @classmethod
    def load(cls, path: Path) -> "GradientBoostingDemandModel":
        """Read a model back, checking the artifact really is one of ours."""
        import joblib

        if not path.is_file():
            raise FileNotFoundError(
                f"{path} is missing. Train first: python scripts/train_models.py"
            )

        payload = joblib.load(path)
        if payload.get("format") != ARTIFACT_FORMAT:
            raise ValueError(
                f"{path} is not a Gradient Boosting artifact "
                f"(format={payload.get('format')!r})"
            )

        known = set(GradientBoostingConfig.__dataclass_fields__)
        model = cls(
            GradientBoostingConfig(
                **{k: v for k, v in payload["config"].items() if k in known}
            ),
            version=payload.get("version", "v1"),
            features=payload["features"],
        )
        model.model = payload["model"]
        if payload.get("metadata"):
            model.metadata = ModelMetadata(**payload["metadata"])

        logger.info(
            "Loaded GBR %s (%d features) from %s",
            model.version,
            len(model.features),
            path,
        )
        return model

    def _require_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError(
                f"GBR {self.version} has not been fitted; call fit() first"
            )


def train_gradient_boosting(
    features: pd.DataFrame,
    *,
    config: Optional[GradientBoostingConfig] = None,
    test_days: int = DEFAULT_TEST_DAYS,
    version: str = "v1",
) -> Tuple[GradientBoostingDemandModel, TrainingSplit, RegressionMetrics]:
    """Split chronologically, fit, and score in one call.

    Returns:
        The fitted model, the split it used, and the holdout metrics.
    """
    split = time_based_split(features, test_days=test_days)
    model = GradientBoostingDemandModel(config, version=version).fit(
        split.train, test=split.test
    )
    metrics = model.evaluate(split.test)

    baseline = baseline_metrics(split.test[TARGET_COLUMN])
    improvement = (1.0 - metrics.mae / baseline.mae) * 100.0 if baseline.mae else 0.0
    logger.info(
        "GBR %s: MAE %.4f vs baseline %.4f (%.0f%% better)",
        version,
        metrics.mae,
        baseline.mae,
        improvement,
    )
    return model, split, metrics


__all__ = [
    "ARTIFACT_FORMAT",
    "DEFAULT_TEST_DAYS",
    "GradientBoostingConfig",
    "GradientBoostingDemandModel",
    "ModelMetadata",
    "TrainingSplit",
    "dataset_hash",
    "time_based_split",
    "train_gradient_boosting",
]
