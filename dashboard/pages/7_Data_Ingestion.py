"""Data Ingestion: where competitor rates come from, and proof they arrive.

The rest of the dashboard shows what the system decided. This page shows what it
knows and how it found out, because a pricing engine reasoning about a market it
can no longer see is the failure that looks most like normal operation.

Everything here goes through the API. The page never scrapes anything itself --
a rate you see on this screen is the same row that landed in the table, because
there is only one implementation of collecting one.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import api_client  # noqa: E402
from dashboard.components.ui import (  # noqa: E402
    ROOM_TYPES,
    bootstrap,
    empty_state,
    kpi_row,
    money,
    require,
    sidebar_status,
)

bootstrap("Data Ingestion")
sidebar_status()

status = require(api_client.ingestion_status(), "Could not read the ingestion status")

source = status["source"]
robots = status["robots"]

# --------------------------------------------------------------------------- #
# What the feed is configured to do
# --------------------------------------------------------------------------- #

st.subheader("Source")

kpi_row(
    [
        ("Source", source["source"], None),
        ("Mode", "Scraper" if source["is_scraper"] else "Synthetic", None),
        ("Observations", f"{status['observations_total']:,}", None),
        ("Last hour", f"{status['observations_last_hour']:,}", None),
        ("Rate limit", f"{source['rate_limit_seconds']:.0f}s", None),
    ]
)

left, right = st.columns([3, 2])

with left:
    if source["is_scraper"]:
        st.markdown(
            f"Scraping **{source['base_url']}** as `{source['user_agent']}`, one "
            f"request every {source['rate_limit_seconds']:.0f}s."
        )
        if source["source"] == "demo_ota":
            st.caption(
                "The demo OTA ships with this project. It is a real web server "
                "returning real HTML over real HTTP, and the scraper reads its "
                "robots.txt and parses it with CSS selectors like any other site "
                "-- the only thing that is not third-party is who runs it."
            )
        elif source["is_third_party"]:
            st.warning(
                "This source sends traffic to a third-party website. Whether you "
                "are permitted to do that is your decision and your legal "
                "exposure. Both Booking.com and Expedia disallow the search "
                "paths this scraper needs, so expect robots.txt to refuse."
            )
    else:
        st.markdown(
            "The **synthetic** generator is producing competitor rates offline. "
            "No network requests are made and no site is being crawled."
        )
        st.caption(
            "Set `INGESTION_SOURCE=demo_ota` to switch to the real scraping path "
            "against the bundled demo site."
        )

with right:
    # robots.txt is surfaced, not hidden, because a scraper that quietly stopped
    # honouring it is a failure with no other symptom.
    if not robots["checked"]:
        st.info("**robots.txt** -- not applicable")
    elif robots["allowed"] is True:
        st.success("**robots.txt** -- crawling permitted")
    elif robots["allowed"] is False:
        st.error("**robots.txt** -- crawling refused")
    else:
        st.warning("**robots.txt** -- could not determine")
    st.caption(robots["detail"])

if robots.get("allowed") is False:
    st.info(
        "A refusal here is the system working correctly, not a fault to clear. "
        "The scraper asked the site for permission and was told no, so it will "
        "not fetch. Use `INGESTION_SOURCE=demo_ota` for a target that grants it."
    )

st.divider()

# --------------------------------------------------------------------------- #
# Run a pass
# --------------------------------------------------------------------------- #

st.subheader("Collect now")
st.caption(
    "Runs one pass through the configured source and, unless you untick publish, "
    "streams each rate to Kafka and persists it through the same handler the "
    "consumer uses -- so a rate collected here and the same rate arriving off the "
    "topic are idempotent."
)

hotels = api_client.list_hotels().unwrap([]) or []
hotel_labels = {f"{h['hotel_id']} - {h['hotel_name']}": h["hotel_id"] for h in hotels}

with st.form("run_ingestion"):
    columns = st.columns([3, 2, 2, 2])

    chosen_hotels = columns[0].multiselect(
        "Hotels", list(hotel_labels), default=list(hotel_labels)[:2],
        help="Leave empty for every hotel. Each one multiplies the request count.",
    )
    chosen_rooms = columns[1].multiselect("Room types", ROOM_TYPES, default=["deluxe"])
    horizon_text = columns[2].text_input(
        "Horizons (days)", value="1,7,30", help="Comma-separated days ahead."
    )
    publish = columns[3].checkbox(
        "Publish", value=True, help="Untick to preview without changing any data."
    )

    submitted = st.form_submit_button("Run collection pass", type="primary")

if submitted:
    try:
        horizons = [int(part.strip()) for part in horizon_text.split(",") if part.strip()]
    except ValueError:
        horizons = []
        st.error("Horizons must be whole numbers, e.g. `1,7,30`.")

    if horizons:
        planned = max(len(chosen_hotels), 1) * max(len(chosen_rooms), 1) * len(horizons)
        with st.spinner(
            f"Collecting -- {planned} request(s) at "
            f"{source['rate_limit_seconds']:.0f}s apart, so about "
            f"{planned * source['rate_limit_seconds']:.0f}s."
        ):
            result = api_client.run_ingestion(
                hotel_ids=[hotel_labels[label] for label in chosen_hotels],
                room_types=chosen_rooms,
                horizons=horizons,
                publish=publish,
            )

        if not result.ok:
            st.error(result.error)
        else:
            run = result.data
            st.session_state["last_ingestion_run"] = run
            if run["succeeded"]:
                st.success(run["detail"])
            elif run["blocked"]:
                st.error(run["detail"])
            else:
                st.warning(run["detail"])
            st.cache_data.clear()

run = st.session_state.get("last_ingestion_run")

if run:
    st.markdown("##### Last run")
    kpi_row(
        [
            ("Requests", str(run["requests_made"]), None),
            ("Rates", str(run["rates_collected"]), None),
            ("Persisted", str(run["persisted"]), None),
            ("Duplicates", str(run["duplicates"]), None),
            ("Empty", str(run["failed"]), None),
            ("Took", f"{run['duration_seconds']:.1f}s", None),
        ]
    )

    if run["duplicates"]:
        st.caption(
            f"{run['duplicates']} rate(s) were already known. That is the "
            "idempotency key doing its job, not wasted work -- the same "
            "observation arriving twice must not move the competitor average."
        )

    if run["errors"]:
        with st.expander(f"{len(run['errors'])} error(s)"):
            for error in run["errors"]:
                st.code(error, language=None)

    rates = run.get("rates") or []
    if rates:
        frame = pd.DataFrame(rates)
        frame["check_in_date"] = pd.to_datetime(frame["check_in_date"])

        st.markdown("##### Rates scraped")
        table, chart = st.columns([3, 2])

        with table:
            st.dataframe(
                frame[
                    ["hotel_id", "competitor", "room_type", "check_in_date", "price"]
                ]
                .rename(
                    columns={
                        "hotel_id": "Hotel",
                        "competitor": "Source",
                        "room_type": "Room",
                        "check_in_date": "Stay date",
                        "price": "Rate",
                    }
                )
                .style.format({"Rate": "{:,.0f}", "Stay date": lambda d: d.date().isoformat()}),
                use_container_width=True,
                hide_index=True,
                height=320,
            )

        with chart:
            figure = px.box(
                frame,
                x="competitor",
                y="price",
                color="competitor",
                labels={"competitor": "", "price": "Rate (INR)"},
            )
            figure.update_layout(
                height=320,
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(figure, use_container_width=True)

        st.caption(
            f"Spread across this pass: {money(frame['price'].min())} to "
            f"{money(frame['price'].max())}. The spread is what makes "
            "`competitor_min_rate` and `competitor_max_rate` separate features."
        )

st.divider()

# --------------------------------------------------------------------------- #
# What has landed
# --------------------------------------------------------------------------- #

st.subheader("What has landed")

by_source = status.get("by_source") or {}
if not by_source:
    empty_state(
        "No competitor observations yet.",
        "Run a collection pass above, or start the producer service.",
    )
else:
    columns = st.columns([2, 3])

    with columns[0]:
        breakdown = (
            pd.DataFrame(
                [{"Source": key, "Rows": value} for key, value in by_source.items()]
            )
            .sort_values("Rows", ascending=False)
        )
        st.dataframe(
            breakdown.style.format({"Rows": "{:,}"}),
            use_container_width=True,
            hide_index=True,
        )
        if status.get("latest_observed_at"):
            st.caption(f"Most recent observation: {status['latest_observed_at']}")

    with columns[1]:
        st.markdown(
            "The `source` column records which implementation produced each row, "
            "so a table containing both synthetic history and scraped rates stays "
            "auditable. Rows from different sources are otherwise identical "
            "downstream -- the feature pipeline does not know or care which is "
            "which, which is exactly the property that makes swapping in a "
            "licensed rate feed a one-class change."
        )
        if status["observations_last_hour"] == 0 and status["observations_total"] > 0:
            st.warning(
                "Nothing has arrived in the last hour. If the producer should be "
                "running, check `docker compose logs producer` -- a dead feed "
                "explains a stale competitor band better than any story about "
                "the market."
            )
