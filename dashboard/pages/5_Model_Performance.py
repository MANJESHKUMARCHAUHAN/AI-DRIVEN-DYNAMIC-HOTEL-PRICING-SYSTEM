"""Model Performance: what is trained, how good it is, and retrain.

Every metric is shown next to the predict-the-mean baseline it has to beat. A
model number with no baseline beside it is unreadable -- 0.064 MAE means nothing
until you know the naive answer is 0.143.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json  # noqa: E402

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import api_client  # noqa: E402
from dashboard.components.charts import (  # noqa: E402
    feature_importance_chart,
    metric_by_horizon_chart,
)
from dashboard.components.ui import (  # noqa: E402
    PROJECT_ROOT as ROOT,
    bootstrap,
    empty_state,
    kpi_row,
    sidebar_status,
)

bootstrap("Model Performance")
sidebar_status()

result = api_client.models()
if not result.ok:
    empty_state("Could not read the model registry.", result.error)
    st.stop()

body = result.data
versions = body.get("versions", [])

# --------------------------------------------------------------------------- #
# Status
# --------------------------------------------------------------------------- #

kpi_row(
    [
        ("Active version", body.get("active_version") or "none", None),
        ("Serving", ", ".join(body.get("available", [])) or "nothing", None),
        ("Feature version", body.get("feature_version", "?"), None),
        ("Versions on disk", str(len(versions)), None),
    ]
)

if body.get("errors"):
    for name, message in body["errors"].items():
        if name == "feature_list" and "contract" in message.lower():
            st.error(
                f"**Feature contract mismatch** ({name}): {message}\n\n"
                "The running code no longer produces the features this model was "
                "trained on. Serving it would produce silently wrong prices, so "
                "it is not being served. Retrain."
            )
        else:
            st.warning(f"{name}: {message}")

if not versions:
    empty_state(
        "Nothing has been trained yet.",
        "Run `python scripts/train_models.py`, or use the retrain control below.",
    )

# --------------------------------------------------------------------------- #
# Per-version metrics
# --------------------------------------------------------------------------- #

if versions:
    st.divider()
    st.subheader("Trained versions")

    rows = []
    for version in versions:
        gbr = version["metrics"].get("gradient_boosting") or {}
        prophet = version["metrics"].get("prophet") or {}
        rows.append(
            {
                "Version": version["version"],
                "Active": "yes" if version["is_active"] else "",
                "Trained": (version.get("trained_at") or "")[:19].replace("T", " "),
                "Train rows": version.get("n_train"),
                "Test rows": version.get("n_test"),
                "GBR MAE": gbr.get("mae"),
                "GBR R2": gbr.get("r2"),
                "Prophet MAE": prophet.get("mae"),
                "Prophet R2": prophet.get("r2"),
                "Features": version.get("feature_version"),
            }
        )

    st.dataframe(
        pd.DataFrame(rows).style.format(
            {
                "GBR MAE": "{:.4f}",
                "GBR R2": "{:.3f}",
                "Prophet MAE": "{:.4f}",
                "Prophet R2": "{:.3f}",
                "Train rows": "{:,.0f}",
                "Test rows": "{:,.0f}",
            },
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Both models are scored on the *same* chronological holdout, so the two "
        "MAEs are directly comparable -- which is what gives the blend weight "
        "between them a basis."
    )

# --------------------------------------------------------------------------- #
# Detail from the training report
# --------------------------------------------------------------------------- #
# The report is written beside the artifacts by the training pipeline. Reading
# it from disk keeps a fair amount of diagnostic detail out of the API surface,
# at the cost of the dashboard needing to share a volume with the API -- which
# it already does in Compose.

report_path = None
if body.get("active_version"):
    candidate = ROOT / "models" / "artifacts" / f"training_report_{body['active_version']}.json"
    if candidate.is_file():
        report_path = candidate

if report_path:
    report = json.loads(report_path.read_text(encoding="utf-8"))

    st.divider()
    st.subheader(f"Training run {report['version']}")

    columns = st.columns(4)
    columns[0].metric("Duration", f"{report['duration_seconds']:.0f}s")
    columns[1].metric("Rows", f"{report['n_rows']:,}")
    columns[2].metric("Train window", " to ".join(report["train_window"]))
    columns[3].metric("Test window", " to ".join(report["test_window"]))

    for step in report.get("steps", []):
        metrics = step.get("metrics") or {}
        baseline = step.get("baseline") or {}
        if not step["succeeded"]:
            st.error(f"{step['name']}: {step['error']}")
            continue
        if not metrics:
            continue

        st.markdown(f"**{step['name'].replace('_', ' ').title()}**")
        cards = []
        for key, label in (
            ("mae", "MAE"), ("rmse", "RMSE"), ("mape", "MAPE %"),
            ("r2", "R2"), ("interval_coverage", "Interval coverage"),
        ):
            if metrics.get(key) is None:
                continue
            delta = None
            if baseline.get(key) is not None and key in {"mae", "rmse", "mape"}:
                improvement = (1 - metrics[key] / baseline[key]) * 100
                delta = f"{improvement:.0f}% better than baseline"
            cards.append((label, f"{metrics[key]:.4f}", delta))
        kpi_row(cards)

    if report.get("feature_importance"):
        st.divider()
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                feature_importance_chart(report["feature_importance"]),
                use_container_width=True,
            )
            st.caption(
                "Permutation importance on the holdout: how much worse the model "
                "gets when a feature is scrambled. Unbiased with respect to "
                "cardinality, unlike scikit-learn's built-in impurity measure."
            )
        with right:
            if report.get("per_horizon"):
                st.plotly_chart(
                    metric_by_horizon_chart(report["per_horizon"]),
                    use_container_width=True,
                )
                st.caption(
                    "Error should be lowest near check-in, where the booking "
                    "curve has already resolved most of the answer. A flat line "
                    "would mean the model is not using the curve."
                )

# --------------------------------------------------------------------------- #
# Retrain
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("Retrain")
st.caption(
    "Runs the full pipeline synchronously: chronological split, Gradient "
    "Boosting, Prophet, evaluation, artifacts. **This takes tens of seconds.**"
)

with st.form("retrain"):
    columns = st.columns(4)
    test_days = columns[0].number_input("Holdout days", 7, 365, 60, step=1)
    with_gbr = columns[1].checkbox("Gradient Boosting", value=True)
    with_prophet = columns[2].checkbox("Prophet", value=True)
    folds = columns[3].number_input(
        "Prophet backtest folds", 0, 5, 0, step=1,
        help="Above zero roughly doubles the run time.",
    )

    if st.form_submit_button("Train now", type="primary"):
        if not (with_gbr or with_prophet):
            st.error("Select at least one model.")
        else:
            with st.spinner("Training. This will take a while..."):
                response = api_client.train(
                    test_days=int(test_days),
                    train_prophet=with_prophet,
                    train_gradient_boosting=with_gbr,
                    backtest_folds=int(folds),
                )
            if response.ok:
                st.success(f"Trained {response.data['version']} in "
                           f"{response.data['duration_seconds']:.0f}s")
                st.code(response.data["summary"], language="text")
                st.cache_data.clear()
            elif response.status_code == 409:
                st.error(
                    f"{response.error}\n\nSeed and build features first: "
                    "`python scripts/seed_database.py && python scripts/build_features.py`"
                )
            else:
                st.error(response.error)
