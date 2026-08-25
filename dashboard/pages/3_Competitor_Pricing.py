"""Competitor Pricing: where the market is, and where we sit in it.

The spread between the cheapest and dearest competitor is as informative as the
average. A tight band means the market has price discipline and moving out of it
is conspicuous; a wide one means there is room to move without anybody noticing.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import api_client  # noqa: E402
from dashboard.components.charts import price_vs_market_chart  # noqa: E402
from dashboard.components.ui import (  # noqa: E402
    COMPETITORS,
    bootstrap,
    date_range_selector,
    empty_state,
    hotel_selector,
    kpi_row,
    money,
    room_type_selector,
    sidebar_status,
)

bootstrap("Competitor Pricing")
sidebar_status()

hotel = hotel_selector(key="comp_hotel")
if hotel is None:
    st.stop()

room_type = room_type_selector(key="comp_room")
start, end = date_range_selector(key="comp_dates", days_back=0, days_forward=45)

result = api_client.competitors(
    hotel["hotel_id"], room_type=room_type, start_date=start, end_date=end
)
if not result.ok:
    empty_state("Could not load competitor rates.", result.error)
    st.stop()

body = result.data
summaries = body.get("summaries", [])
observations = body.get("observations", [])

st.caption(
    f"{hotel['hotel_name']} | {room_type} | {start} to {end} | "
    f"{body['count']} observation(s)"
)

if not summaries:
    empty_state(
        "No competitor rates in this window.",
        "Run the producer and consumer, or submit one below.",
    )
else:
    frame = pd.DataFrame(summaries)

    kpi_row(
        [
            ("Nights covered", str(len(frame)), None),
            ("Market average", money(frame["competitor_rate"].mean()), None),
            ("Cheapest seen", money(frame["competitor_min_rate"].min()), None),
            ("Dearest seen", money(frame["competitor_max_rate"].max()), None),
            ("Mean spread", f"{frame['spread_percent'].mean():.1f}%", None),
        ]
    )

    # Our own base rate for reference -- the market means nothing without it.
    detail = api_client.get_hotel(hotel["hotel_id"])
    our_rate = None
    if detail.ok:
        rooms = {r["room_type"]: r for r in detail.data.get("rooms", [])}
        if room_type in rooms:
            our_rate = rooms[room_type]["base_price"]

    st.plotly_chart(
        price_vs_market_chart(summaries, our_price=our_rate), use_container_width=True
    )

    if our_rate is not None:
        above = (frame["competitor_rate"] < our_rate).mean()
        st.caption(
            f"Our base rate of {money(our_rate)} sits above the market average on "
            f"{above:.0%} of these nights."
        )

st.divider()

# --------------------------------------------------------------------------- #
# Per-source detail
# --------------------------------------------------------------------------- #

if observations:
    st.subheader("By source")
    obs = pd.DataFrame(observations)
    obs["check_in_date"] = pd.to_datetime(obs["check_in_date"])

    left, right = st.columns([2, 3])

    with left:
        by_source = (
            obs.groupby("competitor")
            .agg(rate=("price", "mean"), observations=("price", "size"),
                 available=("is_available", "mean"))
            .reset_index()
            .sort_values("rate")
        )
        st.dataframe(
            by_source.rename(
                columns={
                    "competitor": "Source",
                    "rate": "Mean rate",
                    "observations": "Obs",
                    "available": "Available",
                }
            ).style.format({"Mean rate": "{:,.0f}", "Available": "{:.0%}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "A source that is systematically cheaper is a structural discount, "
            "not noise -- which is why `competitor_min_rate` and "
            "`competitor_max_rate` are separate model features."
        )

    with right:
        figure = px.box(
            obs,
            x="competitor",
            y="price",
            color="competitor",
            labels={"competitor": "", "price": "Rate (INR)"},
        )
        figure.update_layout(
            height=340,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(figure, use_container_width=True)

st.divider()

# --------------------------------------------------------------------------- #
# Manual submission
# --------------------------------------------------------------------------- #

st.subheader("Submit an observed rate")
st.caption(
    "Goes through the same validation and the same persistence handler as the "
    "scrapers and the Kafka consumer. A rate submitted here is indistinguishable "
    "downstream from one collected automatically -- which is the point."
)

with st.form("competitor_event"):
    columns = st.columns(4)
    competitor = columns[0].selectbox("Source", COMPETITORS)
    event_room = columns[1].selectbox("Room type", [room_type])
    check_in = columns[2].date_input("Check-in", value=date.today() + timedelta(days=14))
    price = columns[3].number_input("Rate", min_value=100.0, max_value=1_000_000.0,
                                    value=6500.0, step=100.0)

    if st.form_submit_button("Submit"):
        response = api_client.submit_competitor_event(
            hotel_id=hotel["hotel_id"],
            competitor=competitor,
            room_type=event_room,
            check_in_date=check_in,
            price=float(price),
        )
        if response.ok:
            payload = response.data
            st.success(f"Accepted ({payload['detail']}) -- event {payload['event_id'][:8]}")
            if not payload["published_to_kafka"]:
                st.info(
                    "Kafka was unavailable, so the rate was stored but not "
                    "streamed. The feature pipeline will still pick it up."
                )
            st.cache_data.clear()
        else:
            st.error(response.error)
