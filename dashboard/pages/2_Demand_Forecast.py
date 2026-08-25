"""Demand Forecast: Prophet's view of the coming weeks.

Shows the forecast with its 80% interval, because the interval is the point. A
forecast line alone invites a reader to treat 0.82 as a fact; the band shows it
is 0.74 to 0.90, and that pricing decisions should respect the difference.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import api_client  # noqa: E402
from dashboard.components.charts import forecast_chart  # noqa: E402
from dashboard.components.ui import (  # noqa: E402
    bootstrap,
    empty_state,
    horizon_selector,
    hotel_selector,
    kpi_row,
    percent,
    room_type_selector,
    sidebar_status,
)

bootstrap("Demand Forecast")
sidebar_status()

hotel = hotel_selector(key="fc_hotel")
if hotel is None:
    st.stop()

room_type = room_type_selector(key="fc_room")
horizon = horizon_selector(key="fc_horizon")

result = api_client.forecast(
    hotel["hotel_id"], room_type=room_type, horizon_days=horizon
)

if not result.ok:
    if result.status_code == 503:
        empty_state(
            "No forecasting model has been trained yet.",
            "Run `python scripts/train_models.py`, or use the Model Performance page.",
        )
    else:
        empty_state("Could not load the forecast.", result.error)
    st.stop()

body = result.data
points = body["points"]

if not points:
    empty_state("The model returned no forecast points.")
    st.stop()

frame = pd.DataFrame(points)
frame["date"] = pd.to_datetime(frame["date"])
frame["weekday"] = frame["date"].dt.day_name()
frame["width"] = frame["upper"] - frame["lower"]

st.caption(
    f"{hotel['hotel_name']} | {room_type} | {horizon} days | "
    f"model {body.get('model_version') or 'unknown'}"
)

kpi_row(
    [
        ("Mean demand", percent(frame["forecast"].mean()), None),
        (
            "Peak",
            percent(frame["forecast"].max()),
            frame.loc[frame["forecast"].idxmax(), "date"].strftime("%d %b"),
        ),
        (
            "Trough",
            percent(frame["forecast"].min()),
            frame.loc[frame["forecast"].idxmin(), "date"].strftime("%d %b"),
        ),
        ("Mean interval width", percent(frame["width"].mean()), None),
    ]
)

st.plotly_chart(
    forecast_chart(points, title=f"{hotel['hotel_id']} / {room_type}"),
    use_container_width=True,
)

st.divider()

# --------------------------------------------------------------------------- #
# Weekly shape
# --------------------------------------------------------------------------- #

left, right = st.columns(2)

with left:
    st.subheader("Weekly shape")
    by_weekday = (
        frame.groupby("weekday", sort=False)["forecast"]
        .mean()
        .reindex(
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        )
        .dropna()
    )
    import plotly.express as px

    figure = px.bar(
        by_weekday.reset_index(),
        x="weekday",
        y="forecast",
        labels={"weekday": "", "forecast": "Mean demand"},
    )
    figure.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)"
    )
    figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(figure, use_container_width=True)

    st.caption(
        "Business hotels empty at the weekend and leisure hotels fill. A model "
        "that had learnt a single global weekend effect would be wrong for half "
        "the estate, so each property gets its own fitted series."
    )

with right:
    st.subheader("Uncertainty grows with distance")
    figure = px.line(
        frame,
        x="date",
        y="width",
        labels={"date": "", "width": "Interval width"},
    )
    figure.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)"
    )
    figure.update_yaxes(tickformat=".0%", rangemode="tozero")
    st.plotly_chart(figure, use_container_width=True)

    st.caption(
        "The band should widen as the horizon lengthens. If it does not, the "
        "model is claiming a confidence about next month that it cannot have."
    )

st.divider()

with st.expander("Forecast table"):
    display = frame[["date", "weekday", "forecast", "lower", "upper", "trend"]].rename(
        columns={
            "date": "Date",
            "weekday": "Day",
            "forecast": "Forecast",
            "lower": "Lower",
            "upper": "Upper",
            "trend": "Trend",
        }
    )
    st.dataframe(
        display.style.format(
            {"Forecast": "{:.1%}", "Lower": "{:.1%}", "Upper": "{:.1%}", "Trend": "{:.1%}"}
        ),
        use_container_width=True,
        hide_index=True,
    )
