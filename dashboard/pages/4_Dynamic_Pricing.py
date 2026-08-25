"""Dynamic Pricing: ask for a rate and see exactly how it was built.

The most important page in the dashboard, and deliberately the most verbose. A
price a revenue manager cannot interrogate is a price they will override, and an
overridden pricing system is a switched-off pricing system.

What-if queries default to `persist=False` so that exploring on this page does
not fill the audit trail with prices nobody ever charged.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import api_client  # noqa: E402
from dashboard.components.charts import adjustment_waterfall  # noqa: E402
from dashboard.components.ui import (  # noqa: E402
    bootstrap,
    empty_state,
    hotel_selector,
    kpi_row,
    money,
    percent,
    require,
    room_type_selector,
    sidebar_status,
)

bootstrap("Dynamic Pricing")
sidebar_status()

hotel = hotel_selector(key="price_hotel")
if hotel is None:
    st.stop()

detail = require(api_client.get_hotel(hotel["hotel_id"]), "Could not load the hotel")
rooms = {r["room_type"]: r for r in detail.get("rooms", [])}
if not rooms:
    empty_state("This hotel has no rooms configured.")
    st.stop()

room_type = room_type_selector(key="price_room", options=sorted(rooms))
room = rooms[room_type]

# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #

st.subheader("Situation")
st.caption(
    "Leave a field blank and the API fills it from the feature store. Only the "
    "hotel, room type and date are actually required."
)

columns = st.columns(4)
check_in = columns[0].date_input("Check-in", value=date.today() + timedelta(days=21))
current_price = columns[1].number_input(
    "Current rate", min_value=0.0, max_value=1_000_000.0,
    value=float(room["base_price"]), step=100.0,
    help="Today's rate. Enables the day-over-day change cap.",
)
occupancy = columns[2].slider(
    "Occupancy", min_value=0.0, max_value=1.0, value=0.70, step=0.01,
    help="Rooms already on the books, as a share of inventory.",
)
competitor_rate = columns[3].number_input(
    "Competitor rate", min_value=0.0, max_value=1_000_000.0, value=0.0, step=100.0,
    help="Zero means 'look it up from observed rates'.",
)

persist = st.checkbox(
    "Record this decision in the audit trail",
    value=False,
    help="Off by default: exploring here should not fill the history with "
    "prices nobody ever charged.",
)

if not st.button("Recommend a price", type="primary"):
    st.stop()

result = api_client.predict_price(
    hotel_id=hotel["hotel_id"],
    room_type=room_type,
    check_in_date=check_in,
    current_price=current_price or None,
    occupancy_rate=occupancy,
    available_rooms=int(round(room["room_count"] * (1 - occupancy))),
    competitor_rate=competitor_rate or None,
    persist=persist,
)

if not result.ok:
    empty_state("Could not price this night.", result.error)
    st.stop()

body = result.data

# --------------------------------------------------------------------------- #
# The answer
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("Recommendation")

kpi_row(
    [
        ("Base rate", money(body["base_price"]), None),
        ("Raw price", money(body["raw_recommended_price"]), None),
        (
            "Final price",
            money(body["final_recommended_price"]),
            f"{body['price_change_percent']:+.1f}% vs current",
        ),
        ("Confidence", percent(body["confidence"], 0), None),
        ("Model", body["model_version"], None),
    ]
)

if body["guardrails_applied"]:
    st.warning(
        "Guardrails changed this price: " + ", ".join(body["guardrails_applied"])
    )
else:
    st.success("No guardrails were triggered -- the model's price was served as-is.")

if body["demand"]["degraded"]:
    st.info(
        "Running degraded: " + "; ".join(body["demand"]["notes"] or ["a model was unavailable"])
    )

# --------------------------------------------------------------------------- #
# The build-up
# --------------------------------------------------------------------------- #

st.divider()
left, right = st.columns([3, 2])

with left:
    st.subheader("How the price was built")
    st.plotly_chart(
        adjustment_waterfall(body["adjustments"], body["base_price"]),
        use_container_width=True,
    )

with right:
    st.subheader("Why")
    for adjustment in body["adjustments"]:
        marker = "clamped" if adjustment["clamped"] else ""
        st.markdown(
            f"**{adjustment['name'].title()}** &nbsp; `{adjustment['percent']:+.1f}%` "
            f"{'`' + marker + '`' if marker else ''}  \n{adjustment['reason']}"
        )

    st.divider()
    st.markdown(
        f"**Demand** &nbsp; `{body['blended_demand']:.1%}`  \n"
        f"Prophet {body['forecasted_demand']:.1%} | "
        f"GBR {body['predicted_demand']:.1%}"
        if body["forecasted_demand"] is not None and body["predicted_demand"] is not None
        else f"**Demand** &nbsp; `{body['blended_demand']:.1%}`"
    )

# --------------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------------- #

if body["guardrail_detail"]:
    st.divider()
    st.subheader("Guardrails that fired")
    guardrails = pd.DataFrame(body["guardrail_detail"])
    st.dataframe(
        guardrails.rename(
            columns={
                "rule": "Rule",
                "before": "Before",
                "after": "After",
                "delta": "Change",
                "reason": "Reason",
            }
        ).style.format({"Before": "{:,.0f}", "After": "{:,.0f}", "Change": "{:+,.0f}"}),
        use_container_width=True,
        hide_index=True,
    )

# --------------------------------------------------------------------------- #
# Sensitivity
# --------------------------------------------------------------------------- #

st.divider()
st.subheader("Sensitivity to occupancy")
st.caption(
    "The same night priced across the occupancy range. The shape shows the "
    "occupancy x lead-time interaction: a hotel that is behind pace is only "
    "discounted once time has run short."
)

with st.spinner("Pricing across the occupancy range..."):
    rows = []
    for candidate in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98]:
        sweep = api_client.predict_price(
            hotel_id=hotel["hotel_id"],
            room_type=room_type,
            check_in_date=check_in,
            current_price=current_price or None,
            occupancy_rate=candidate,
            competitor_rate=competitor_rate or None,
            persist=False,
        )
        if sweep.ok:
            rows.append(
                {
                    "occupancy": candidate,
                    "final": sweep.data["final_recommended_price"],
                    "raw": sweep.data["raw_recommended_price"],
                    "guardrails": len(sweep.data["guardrails_applied"]),
                }
            )

if rows:
    import plotly.graph_objects as go

    from dashboard.components.ui import COLOURS

    sweep_frame = pd.DataFrame(rows)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=sweep_frame["occupancy"], y=sweep_frame["raw"], name="Raw",
            line=dict(color=COLOURS["muted"], dash="dot"),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=sweep_frame["occupancy"], y=sweep_frame["final"], name="Final",
            line=dict(color=COLOURS["primary"], width=3), mode="lines+markers",
        )
    )
    figure.update_layout(
        height=360,
        xaxis=dict(title="Occupancy", tickformat=".0%"),
        yaxis=dict(title="Recommended rate (INR)"),
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=1.05),
    )
    st.plotly_chart(figure, use_container_width=True)

# --------------------------------------------------------------------------- #
# The full explanation
# --------------------------------------------------------------------------- #

with st.expander("Full explanation (paste this into a ticket)"):
    st.code(body["explanation"], language="text")
    st.caption(f"prediction_id: {body['prediction_id']} | {body['latency_ms']:.0f} ms")
