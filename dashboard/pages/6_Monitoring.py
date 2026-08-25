"""Monitoring: data quality, drift, prediction behaviour, guardrail pressure.

Reads two things: the live API (for service and model status) and the JSON
report written by ``scripts/monitor.py``. The report is a file rather than an
endpoint on purpose -- monitoring runs on a schedule and holds a lot of
diagnostic detail that does not belong on the pricing service's public surface.

The dashboard and the API share the artifact volume in Compose, which is what
makes the file readable from here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import api_client  # noqa: E402
from dashboard.components.charts import distribution_chart  # noqa: E402
from dashboard.components.ui import (  # noqa: E402
    bootstrap,
    empty_state,
    kpi_row,
    money,
    percent,
    sidebar_status,
)

bootstrap("Monitoring")
sidebar_status()

SEVERITY_STYLE = {"ok": st.success, "warning": st.warning, "critical": st.error}

# --------------------------------------------------------------------------- #
# Live service status
# --------------------------------------------------------------------------- #

st.subheader("Service")

health = api_client.health()
if not health.ok:
    st.error(f"The API is unreachable: {health.error}")
else:
    body = health.data
    dependencies = body.get("dependencies", [])
    kpi_row(
        [("Status", body["status"].upper(), None)]
        + [
            (d["name"], d["state"], f"{d['latency_ms']:.0f} ms" if d.get("latency_ms") else None)
            for d in dependencies
        ]
    )

st.divider()

# --------------------------------------------------------------------------- #
# The monitoring report
# --------------------------------------------------------------------------- #

data_dir = Path(__file__).resolve().parents[2] / "data"
report_path = data_dir / "monitoring_report.json"

if not report_path.is_file():
    empty_state(
        "No monitoring report has been generated yet.",
        "Run `python scripts/monitor.py` to produce one.",
    )
    st.stop()

report = json.loads(report_path.read_text(encoding="utf-8"))

banner = SEVERITY_STYLE.get(report["severity"], st.info)
banner(f"Overall severity: **{report['severity'].upper()}**")
st.caption(f"Report generated {report['generated_at']}")

# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #

data_quality = report.get("data_quality")
if data_quality:
    st.subheader(f"Data quality -- {data_quality['passed']}/{data_quality['total']} passed")
    st.caption(
        "Most production ML failures are data failures wearing a model's "
        "clothes: a feed that quietly stops, a pipeline a day behind, a column "
        "that starts arriving null. None of those raise an exception."
    )

    for check in data_quality["checks"]:
        severity = check["severity"]
        icon = {"ok": ":white_check_mark:", "warning": ":warning:", "critical": ":x:"}[severity]
        with st.expander(
            f"{icon} {check['name'].replace('_', ' ')} -- {check['message']}",
            expanded=severity != "ok",
        ):
            columns = st.columns(3)
            columns[0].metric("Severity", severity.upper())
            if check.get("value") is not None:
                columns[1].metric("Value", f"{check['value']:,.4g}")
            if check.get("threshold") is not None:
                columns[2].metric("Threshold", f"{check['threshold']:,.4g}")
            if check.get("detail"):
                st.json(check["detail"])

st.divider()

# --------------------------------------------------------------------------- #
# Model health
# --------------------------------------------------------------------------- #

model_health = report.get("model_health")
if not model_health:
    st.stop()

st.subheader("Model health")

for check in model_health.get("checks", []):
    severity = check["severity"]
    if severity == "critical":
        st.error(f"**{check['name']}** -- {check['message']}")
    elif severity == "warning":
        st.warning(f"**{check['name']}** -- {check['message']}")

# --- drift ------------------------------------------------------------------ #

drift = model_health.get("drift", [])
if drift:
    st.markdown("#### Feature drift")
    frame = pd.DataFrame(drift).sort_values("psi", ascending=False, na_position="last")

    st.dataframe(
        frame.rename(
            columns={
                "feature": "Feature",
                "psi": "PSI",
                "severity": "Severity",
                "reference_mean": "Reference mean",
                "current_mean": "Current mean",
                "reference_n": "Ref n",
                "current_n": "Cur n",
            }
        ).style.format(
            {
                "PSI": "{:.3f}",
                "Reference mean": "{:,.3f}",
                "Current mean": "{:,.3f}",
                "Ref n": "{:,.0f}",
                "Cur n": "{:,.0f}",
            },
            na_rep="n/a",
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "PSI bands are the industry conventions: below 0.10 no meaningful "
        "shift, 0.10-0.25 worth watching, above 0.25 worth retraining. They are "
        "conventions rather than laws -- and on less than two years of history "
        "they cannot separate drift from seasonality, which is why the caveat "
        "above matters more than any individual number."
    )

# --- distributions ---------------------------------------------------------- #

st.markdown("#### Served distributions")
st.caption(
    "The shape moving is the earliest visible sign of a problem, well before "
    "any accuracy metric reacts -- accuracy needs the outcome, and for a hotel "
    "that arrives after the stay date."
)

left, right = st.columns(2)

predictions = model_health.get("prediction_stats") or {}
prices = model_health.get("price_stats") or {}

with left:
    if predictions.get("values"):
        st.plotly_chart(
            distribution_chart(
                predictions["values"],
                title="Predicted demand",
                x_title="Demand (share of inventory)",
                reference=predictions.get("demand_mean"),
            ),
            use_container_width=True,
        )
        kpi_row(
            [
                ("Predictions", f"{predictions['n']:,}", None),
                ("Mean", percent(predictions["demand_mean"]), None),
                ("Spread", f"{predictions['demand_std']:.3f}", None),
                ("Confidence", percent(predictions["confidence_mean"], 0), None),
            ]
        )
        if predictions.get("latency_p95_ms"):
            st.caption(f"p95 serving latency: {predictions['latency_p95_ms']:.0f} ms")
    else:
        empty_state("No predictions have been served yet.")

with right:
    if prices.get("values"):
        st.plotly_chart(
            distribution_chart(
                prices["values"],
                title="Recommended prices",
                x_title="Rate (INR)",
                reference=prices.get("price_mean"),
            ),
            use_container_width=True,
        )
        kpi_row(
            [
                ("Decisions", f"{prices['n']:,}", None),
                ("Mean rate", money(prices["price_mean"]), None),
                ("Clamped", percent(prices["clamped_share"], 0), None),
                ("Mean change", f"{prices['mean_change_percent']:+.1f}%", None),
            ]
        )
    else:
        empty_state("No pricing decisions have been recorded yet.")

# --- guardrails -------------------------------------------------------------- #

counts = model_health.get("guardrail_counts") or {}
if counts:
    st.markdown("#### Guardrails fired")
    import plotly.express as px

    frame = pd.DataFrame(
        sorted(counts.items(), key=lambda kv: kv[1], reverse=True),
        columns=["rule", "count"],
    )
    figure = px.bar(frame, x="count", y="rule", orientation="h", labels={"rule": "", "count": "Times fired"})
    figure.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, use_container_width=True)

    st.caption(
        "A guardrail firing occasionally is the system working. One firing on "
        "most decisions means the model wants prices the business will not "
        "allow -- a retuning signal, not a success."
    )

# --- realised accuracy -------------------------------------------------------- #

accuracy = model_health.get("accuracy")
if accuracy:
    st.markdown("#### Realised accuracy")
    st.caption(
        "Served predictions scored against what actually happened. The truest "
        "measure and the slowest -- by the time it moves, the prices were "
        "already wrong."
    )
    kpi_row(
        [
            ("Nights scored", f"{accuracy['n']:,}", None),
            ("MAE", f"{accuracy['mae']:.4f}", None),
            ("RMSE", f"{accuracy['rmse']:.4f}", None),
            ("R2", f"{accuracy['r2']:.3f}", None),
            ("Bias", f"{accuracy['bias']:+.4f}", None),
        ]
    )
else:
    st.info(
        "Not enough completed nights to score realised accuracy yet. A "
        "prediction for next month cannot be scored until next month."
    )
