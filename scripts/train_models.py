"""Train both demand models and write versioned artifacts.

Reads the feature store, splits it chronologically, trains the Gradient Boosting
regressor and the Prophet bundle, evaluates both against a predict-the-mean
baseline, and saves the artifacts with a JSON training report.

Usage::

    python scripts/train_models.py                    # train both
    python scripts/train_models.py --gbr-only         # skip Prophet (fast)
    python scripts/train_models.py --no-backtest      # skip Prophet backtesting
    python scripts/train_models.py --test-days 90
    python scripts/train_models.py --version v7       # overwrite a version

Every reported number is printed next to the baseline it must beat. A model that
does not beat predicting the mean is not a model, and the output makes that
impossible to miss.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Allow execution as a plain script from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from config import get_settings  # noqa: E402
from database.connection import session_scope, wait_for_database  # noqa: E402
from models.gradient_boosting_model import DEFAULT_TEST_DAYS  # noqa: E402
from monitoring.logging_config import configure_logging, get_logger  # noqa: E402
from training.pipeline import TrainingFailed, TrainingPipeline, TrainingResult  # noqa: E402

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/train_models.py",
        description="Train the demand models and save versioned artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--version", type=str, default=None,
        help="Artifact version. Defaults to the next unused vN.",
    )
    parser.add_argument(
        "--test-days", type=int, default=DEFAULT_TEST_DAYS,
        help="Length of the chronological holdout in days.",
    )
    parser.add_argument(
        "--gbr-only", action="store_true", help="Train only the Gradient Boosting model."
    )
    parser.add_argument(
        "--prophet-only", action="store_true", help="Train only the Prophet bundle."
    )
    parser.add_argument(
        "--no-backtest", action="store_true",
        help="Skip Prophet's rolling-origin backtest. Roughly halves the run time, "
        "at the cost of having no Prophet metrics to report.",
    )
    parser.add_argument(
        "--backtest-folds", type=int, default=2,
        help="Rolling-origin folds per Prophet series.",
    )
    return parser


def _print_report(result: TrainingResult) -> None:
    print()
    print("=" * 78)
    print("TRAINING COMPLETE")
    print("=" * 78)
    print(result.summary())

    gbr = result.step("gradient_boosting")
    if gbr and gbr.succeeded:
        print("\n" + "-" * 78)
        print("GRADIENT BOOSTING -- holdout")
        print("-" * 78)
        for name in ("mae", "rmse", "mape", "weighted_mape", "r2", "bias"):
            model_value = gbr.metrics.get(name)
            base_value = gbr.baseline.get(name)
            if model_value is None:
                continue
            print(
                f"  {name:<16} {model_value:>10.4f}"
                + (f"   baseline {base_value:>9.4f}" if base_value is not None else "")
            )

        if result.feature_importance:
            print("\n  permutation importance (holdout, top 10):")
            for row in result.feature_importance[:10]:
                print(f"    {row['feature']:<24} {row['importance']:.5f}")

        if result.per_horizon:
            print("\n  accuracy by days to check-in:")
            frame = pd.DataFrame(result.per_horizon).sort_values("days_to_checkin")
            for row in frame.itertuples():
                print(
                    f"    {int(row.days_to_checkin):>3}d  n={int(row.n):>4}  "
                    f"MAE={row.mae:.4f}  R2={row.r2:+.3f}  "
                    f"(baseline {row.baseline_mae:.4f})"
                )

    prophet = result.step("prophet")
    if prophet and prophet.succeeded:
        print("\n" + "-" * 78)
        print("PROPHET -- same holdout as the Gradient Boosting model")
        print("-" * 78)
        print(
            f"  series fitted    {prophet.detail.get('series_fitted')}"
            f"/{prophet.detail.get('series_total')}"
        )
        for name in ("mae", "rmse", "mape", "r2", "interval_coverage"):
            value = prophet.metrics.get(name)
            base_value = prophet.baseline.get(name)
            if value is None:
                continue
            print(
                f"  {name:<16} {value:>10.4f}"
                + (f"   baseline {base_value:>9.4f}" if base_value is not None else "")
            )

        backtest = prophet.detail.get("backtest_metrics")
        if backtest:
            print(
                f"\n  rolling-origin backtest inside the training window: "
                f"MAE={backtest['mae']:.4f} "
                f"(a stability check, not comparable to the holdout above)"
            )

    print("\n" + "-" * 78)
    print("ARTIFACTS")
    print("-" * 78)
    for name, path in result.artifacts.items():
        print(f"  {name:<18} {path}")
    print("=" * 78)


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns a process exit code."""
    settings = get_settings()
    configure_logging(settings)
    args = _build_parser().parse_args(argv)

    if args.gbr_only and args.prophet_only:
        logger.error("--gbr-only and --prophet-only are mutually exclusive")
        return 2

    settings.ensure_directories()
    pipeline = TrainingPipeline(settings)

    try:
        wait_for_database()
        with session_scope() as session:
            result = pipeline.run(
                session,
                version=args.version,
                test_days=args.test_days,
                backtest_folds=0 if args.no_backtest else args.backtest_folds,
                train_prophet=not args.gbr_only,
                train_gbr=not args.prophet_only,
            )
    except TrainingFailed as exc:
        logger.error("Training failed: %s", exc)
        return 1
    except Exception as exc:
        logger.error("Training failed: %s: %s", type(exc).__name__, exc)
        return 1

    _print_report(result)

    failed = [s.name for s in result.steps if not s.succeeded]
    if failed:
        print(f"\nWARNING: {', '.join(failed)} did not train. See the log above.")
        print("Next: python scripts/train_models.py  (after fixing the cause)")
        return 1

    print("\nNext: uvicorn api.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
