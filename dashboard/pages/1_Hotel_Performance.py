"""Hotel Performance: how one property is actually trading.

Occupancy and ADR are drawn on twin axes because they trade off against each
other -- the easiest way to lift occupancy is to drop ADR, and either number
alone can be made to look excellent while RevPAR falls.
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
from dashboard.components.ui import (  # noqa: E402
    bootstrap,
    empty_state,
    hotel_selector,
    kpi_row,
    money,
    percent,
    require,
    sidebar_status,
)

bootstrap("Hotel Performance")
sidebar_status()

hotel = hotel_selector(key="perf_hotel")
if hotel is None:
    st.stop()

detail = require(api_client.get_hotel(hotel["hotel_id"]), "Could not load the hotel")

st.subheader(f"{detail['hotel_name']} -- {detail['city']}")
st.caption(
    f"{detail['star_rating']}-star | {detail['segment']} | "
    f"{detail['total_rooms']} rooms | {detail['currency']}"
)

kpi_row(
    [
        ("Occupancy (30d)", percent(detail.get("occupancy_last_30_days")), None),
        ("ADR (30d)", money(detail.get("adr_last_30_days")), None),
        ("RevPAR (30d)", money(detail.get("revpar_last_30_days")), None),
        ("Room types", str(len(detail.get("rooms", []))), None),
    ]
)

st.divider()

# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #

st.subheader("Inventory and rate structure")

rooms = pd.DataFrame(detail.get("rooms", []))
if rooms.empty:
    empty_state("This hotel has no rooms configured.")
    st.stop()

left, right = st.columns([3, 2])

with left:
    display = rooms[
        ["room_type", "capacity", "room_count", "base_price", "floor_price", "ceiling_price"]
    ].rename(
        columns={
            "room_type": "Room type",
            "capacity": "Sleeps",
            "room_count": "Rooms",
            "base_price": "Base rate",
            "floor_price": "Floor",
            "ceiling_price": "Ceiling",
        }
    )
    st.dataframe(
        display.style.format(
            {"Base rate": "{:,.0f}", "Floor": "{:,.0f}", "Ceiling": "{:,.0f}"},
            na_rep="n/a",
        ),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Floor and ceiling are per-room guardrails. They sit *inside* the global "
        "MIN_PRICE and MAX_PRICE limits -- their job is to stop one room type "
        "being sold at another's price, not to replace the absolute bounds."
    )

with right:
    import plotly.express as px

    figure = px.bar(
        rooms,
        x="room_type",
        y="room_count",
        color="base_price",
        color_continuous_scale="Blues",
        labels={"room_type": "", "room_count": "Rooms", "base_price": "Base rate"},
    )
    figure.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(figure, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------- #
# Recent pricing activity
# --------------------------------------------------------------------------- #

st.subheader("Recent pricing activity")

history = api_client.pricing_history(hotel["hotel_id"], limit=200)
if not history.ok:
    empty_state("Could not load pricing history.", history.error)
elif not history.data.get("items"):
    empty_state(
        "No prices have been recommended for this hotel yet.",
        "Use the Dynamic Pricing page to generate one.",
    )
else:
    items = pd.DataFrame(history.data["items"])
    items["clamped"] = items["guardrails_applied"].apply(bool)

    kpi_row(
        [
            ("Decisions", str(len(items)), None),
            ("Mean final rate", money(items["final_recommended_price"].mean()), None),
            (
                "Mean change",
                f"{items['price_change_percent'].mean():+.1f}%",
                None,
            ),
            (
                "Guardrails fired",
                f"{items['clamped'].mean():.0%}",
                None,
            ),
        ]
    )

    from dashboard.components.charts import price_history_chart

    st.plotly_chart(price_history_chart(history.data["items"]), use_container_width=True)

    if items["clamped"].mean() > 0.5:
        st.warning(
            "Guardrails are firing on more than half of decisions. That is the "
            "model asking for prices the business will not allow -- a retuning "
            "signal, not a success."
        )
