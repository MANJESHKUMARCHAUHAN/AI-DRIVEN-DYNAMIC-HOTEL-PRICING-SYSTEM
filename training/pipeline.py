"""The training pipeline: feature store in, versioned model artifacts out.

One entry point, :meth:`TrainingPipeline.run`, executing the sequence from
docs/architecture.md §7 Flow B::

    feature store -> validate -> chronological split -> Prophet -> GBR
                  -> evaluate -> artifacts -> report

Three properties this pipeline is built around:

**Reproducible.** Every run records the dataset hash, the feature version, the
hyperparameters and the exact train/test windows. Two runs over the same data
with the same configuration produce the same model, and the artifact carries
enough to prove it.

**No future information.** The split is chronological and the feature matrix was
built with snapshot discipline (see :mod:`features.feature_engineering`).
Nothing here shuffles, and nothing re-derives a feature from the test window.

**Partially successful is a real outcome.** Prophet failing must not lose a
perfectly good Gradient Boosting model, and vice versa. Each model is trained in
its own guarded step and the report says plainly what succeeded. A pipeline that
is all-or-nothing turns one flaky Stan fit into "no models today".
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy.orm import Session

from config import Settings, get_settings
from features.feature_engineering import FEATURE_COLUMNS, FEATURE_VERSION, TARGET_COLUMN
from features.feature_store import FeatureStore, save_feature_list
from models.gradient_boosting_model import (
    DEFAULT_TEST_DAYS,
    GradientBoostingConfig,
    GradientBoostingDemandModel,
    time_based_split,
)
from models.metrics import baseline_metrics
from models.prophet_model import ProphetBundle, ProphetConfig
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

#: Minimum labelled rows before training is worth attempting.
MIN_TRAINING_ROWS = 500


class TrainingFailed(RuntimeError):
    """Raised when the pipeline cannot produce any usable model."""


@dataclass
class StepOutcome:
    """What one model's training step did."""

    name: str
    succeeded: bool
    duration_seconds: float = 0.0
    artifact: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    baseline: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingResult:
    """The full record of a training run.

    Serialised to ``training_report_<version>.json`` beside the artifacts, and
    consumed by the model registry in Phase 10.
    """

    version: str
    feature_version: str
    started_at: str
    finished_at: str
    duration_seconds: float
    n_rows: int
    n_train: int
    n_test: int
    train_window: List[str]
    test_window: List[str]
    dataset_hash: Optional[str]
    steps: List[StepOutcome] = field(default_factory=list)
    feature_importance: List[Dict[str, Any]] = field(default_factory=list)
    per_horizon: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """True if at least one model was trained and saved."""
        return any(step.succeeded for step in self.steps)

    def step(self, name: str) -> Optional[StepOutcome]:
        return next((s for s in self.steps if s.name == name), None)

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["succeeded"] = self.succeeded
        return payload

    def summary(self) -> str:
        lines = [
            f"version         {self.version}",
            f"feature version {self.feature_version}",
            f"rows            {self.n_rows:,} "
            f"({self.n_train:,} train / {self.n_test:,} test)",
            f"train window    {self.train_window[0]} .. {self.train_window[1]}",
            f"test window     {self.test_window[0]} .. {self.test_window[1]}",
            f"duration        {self.duration_seconds:.1f}s",
        ]
        for step in self.steps:
            if step.succeeded:
                mae = step.metrics.get("mae")
                base = step.baseline.get("mae")
                gain = (
                    f" ({(1 - mae / base) * 100:.0f}% better than baseline)"
                    if mae and base
                    else ""
                )
                lines.append(
                    f"{step.name:<15} ok  MAE={mae:.4f}{gain}"
                    if mae
                    else f"{step.name:<15} ok"
                )
            else:
                lines.append(f"{step.name:<15} FAILED  {step.error}")
        return "\n".join(lines)


class TrainingPipeline:
    """Trains both demand models and writes versioned artifacts.

    Example::

        with session_scope() as session:
            result = TrainingPipeline().run(session)
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        gbr_config: Optional[GradientBoostingConfig] = None,
        prophet_config: Optional[ProphetConfig] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.gbr_config = gbr_config or GradientBoostingConfig()
        self.prophet_config = prophet_config or ProphetConfig()
        self.artifact_dir: Path = self.settings.model.model_dir

    # -- versioning --------------------------------------------------------- #

    def next_version(self) -> str:
        """``v1``, ``v2``, ... derived from what is already on disk.

        Deliberately not a timestamp: a human comparing two models wants to know
        which came second, and ``v3`` answers that at a glance where
        ``20260824-162143`` does not.
        """
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        existing = [
            int(match.group(1))
            for path in self.artifact_dir.glob("training_report_v*.json")
            if (match := re.fullmatch(r"training_report_v(\d+)\.json", path.name))
        ]
        return f"v{max(existing, default=0) + 1}"

    # -- data ---------------------------------------------------------------- #

    def load_features(self, session: Session) -> pd.DataFrame:
        """Read the feature store and expand it into the model matrix.

        Raises:
            TrainingFailed: If there is not enough labelled data to train on.
        """
        matrix = FeatureStore.load_model_matrix(session)

        labelled = matrix[matrix[TARGET_COLUMN].notna()]
        if len(labelled) < MIN_TRAINING_ROWS:
            raise TrainingFailed(
                f"only {len(labelled)} labelled row(s) in the feature store, need "
                f"at least {MIN_TRAINING_ROWS}. Run scripts/seed_database.py and "
                f"scripts/build_features.py first."
            )

        logger.info(
            "Training on %d labelled row(s), %d feature(s), version %s",
            len(labelled),
            len(FEATURE_COLUMNS),
            FEATURE_VERSION,
        )
        return labelled.reset_index(drop=True)

    # -- steps --------------------------------------------------------------- #

    def _train_gradient_boosting(
        self, split, version: str, result: TrainingResult
    ) -> Optional[GradientBoostingDemandModel]:
        """Fit, evaluate and save the Gradient Boosting model."""
        started = time.perf_counter()
        try:
            model = GradientBoostingDemandModel(
                self.gbr_config, version=version
            ).fit(split.train, test=split.test)

            metrics = model.evaluate(split.test)
            baseline = baseline_metrics(split.test[TARGET_COLUMN])
            path = model.save(self.artifact_dir / f"gbr_{version}.joblib")

            result.dataset_hash = model.metadata.dataset_hash if model.metadata else None
            result.feature_importance = (
                model.permutation_importance(split.test, n_repeats=5)
                .head(20)
                .round(6)
                .to_dict("records")
            )
            result.per_horizon = model.evaluate_by(
                split.test, "days_to_checkin"
            ).to_dict("records")
            result.artifacts["gradient_boosting"] = str(path)

            result.steps.append(
                StepOutcome(
                    name="gradient_boosting",
                    succeeded=True,
                    duration_seconds=time.perf_counter() - started,
                    artifact=str(path),
                    metrics=metrics.as_dict(),
                    baseline=baseline.as_dict(),
                    detail={
                        "n_estimators_used": model.metadata.n_estimators_used,
                        "residual_std": model.metadata.residual_std,
                    },
                )
            )
            return model
        except Exception as exc:
            logger.error(
                "Gradient Boosting training failed: %s: %s", type(exc).__name__, exc
            )
            result.steps.append(
                StepOutcome(
                    name="gradient_boosting",
                    succeeded=False,
                    duration_seconds=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            return None

    def _train_prophet(
        self,
        split,
        version: str,
        result: TrainingResult,
        *,
        backtest_folds: int,
        backtest_horizon: int,
    ) -> Optional[ProphetBundle]:
        """Fit on the training window, score on the same holdout as the GBR."""
        started = time.perf_counter()
        try:
            daily = ProphetBundle.daily_demand(split.train)
            bundle = ProphetBundle(self.prophet_config).fit_all(
                daily,
                backtest_folds=backtest_folds,
                backtest_horizon=backtest_horizon,
            )
            if not bundle.models:
                raise TrainingFailed("no Prophet series could be fitted")

            # Scored on the chronological holdout, exactly like the GBR, so the
            # two numbers are comparable and the blend weight has a basis.
            metrics = bundle.evaluate_on(split.test)
            baseline = baseline_metrics(split.test[TARGET_COLUMN])
            path = bundle.save(self.artifact_dir / f"prophet_{version}.joblib")
            result.artifacts["prophet"] = str(path)

            report = bundle.report_frame()
            backtest = bundle.aggregate_metrics()
            result.steps.append(
                StepOutcome(
                    name="prophet",
                    succeeded=True,
                    duration_seconds=time.perf_counter() - started,
                    artifact=str(path),
                    metrics=metrics.as_dict() if metrics else {},
                    baseline=baseline.as_dict(),
                    detail={
                        "series_fitted": int(report["fitted"].sum()),
                        "series_total": int(len(report)),
                        "backtest_folds": backtest_folds,
                        # Secondary number: rolling-origin inside the training
                        # window. Useful for spotting instability over time,
                        # not comparable to the holdout score above.
                        "backtest_metrics": backtest.as_dict() if backtest else None,
                    },
                )
            )
            return bundle
        except Exception as exc:
            logger.error("Prophet training failed: %s: %s", type(exc).__name__, exc)
            result.steps.append(
                StepOutcome(
                    name="prophet",
                    succeeded=False,
                    duration_seconds=time.perf_counter() - started,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            return None

    # -- orchestration ------------------------------------------------------- #

    def run(
        self,
        session: Session,
        *,
        version: Optional[str] = None,
        test_days: int = DEFAULT_TEST_DAYS,
        backtest_folds: int = 2,
        backtest_horizon: int = 30,
        train_prophet: bool = True,
        train_gbr: bool = True,
    ) -> TrainingResult:
        """Run the pipeline end to end.

        Args:
            version: Artifact version. Defaults to the next unused ``vN``.
            test_days: Length of the chronological holdout.
            backtest_folds: Rolling-origin folds per Prophet series. Zero skips
                Prophet backtesting, which halves the run time.

        Returns:
            The training record, whether or not every step succeeded.

        Raises:
            TrainingFailed: If there is too little data, or if no model could be
                trained at all.
        """
        version = version or self.next_version()
        started_at = datetime.now(timezone.utc)
        clock = time.perf_counter()

        features = self.load_features(session)
        split = time_based_split(features, test_days=test_days)

        result = TrainingResult(
            version=version,
            feature_version=FEATURE_VERSION,
            started_at=started_at.isoformat(),
            finished_at="",
            duration_seconds=0.0,
            n_rows=len(features),
            n_train=len(split.train),
            n_test=len(split.test),
            train_window=[
                str(split.train["stay_date"].min().date()),
                str(split.train["stay_date"].max().date()),
            ],
            test_window=[
                str(split.test["stay_date"].min().date()),
                str(split.test["stay_date"].max().date()),
            ],
            dataset_hash=None,
        )

        if train_gbr:
            self._train_gradient_boosting(split, version, result)

        if train_prophet:
            # Fitted on the training window only: the holdout must stay unseen
            # by both models, or their scores are not comparable.
            self._train_prophet(
                split,
                version,
                result,
                backtest_folds=backtest_folds,
                backtest_horizon=backtest_horizon,
            )

        if not result.succeeded:
            errors = "; ".join(s.error or "" for s in result.steps)
            raise TrainingFailed(f"no model could be trained. {errors}")

        # The feature list is written once per run, next to the artifacts, and
        # validated at load time. This is the train/serve contract on disk.
        feature_list = save_feature_list(self.artifact_dir)
        result.artifacts["feature_list"] = str(feature_list)

        result.finished_at = datetime.now(timezone.utc).isoformat()
        result.duration_seconds = time.perf_counter() - clock

        report_path = self.artifact_dir / f"training_report_{version}.json"
        report_path.write_text(
            json.dumps(result.as_dict(), indent=2, default=str), encoding="utf-8"
        )
        result.artifacts["report"] = str(report_path)

        logger.info("Training run %s complete in %.1fs", version, result.duration_seconds)
        return result


def latest_report(artifact_dir: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """The most recent training report, or ``None`` if nothing has been trained."""
    artifact_dir = artifact_dir or get_settings().model.model_dir
    reports = sorted(
        artifact_dir.glob("training_report_v*.json"),
        key=lambda p: int(re.findall(r"v(\d+)", p.name)[0]),
    )
    if not reports:
        return None
    return json.loads(reports[-1].read_text(encoding="utf-8"))


__all__ = [
    "MIN_TRAINING_ROWS",
    "StepOutcome",
    "TrainingFailed",
    "TrainingPipeline",
    "TrainingResult",
    "latest_report",
]
