"""Streamlit analytics dashboard -- entry point and Overview page.

The dashboard is a **read-only consumer of the API**. It never opens a database
connection of its own and it contains no pricing logic; every number it renders
comes from an HTTP call. That is not laziness -- it means there is exactly one
implementation of "what is the right price", and the dashboard doubles as an
integration test of the API. If a page renders, the endpoint behind it works.

The other eight pages live in ``dashboard/pages/`` and Streamlit discovers them
automatically, in filename order. ``8_AI_Agent`` is the one exception to the
"every page is an integration test of the API" rule: it needs the optional
``anthropic`` SDK, and renders installation instructions rather than failing when
it is absent.

Run it with::

    streamlit run dashboard/app.py

``API_BASE_URL`` points it at the API; inside Compose that is ``http://api:8000``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit executes this file as a top-level script, so the project root is not
# importable until we say so. Must happen before any project import.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from dashboard import api_client  # noqa: E402
from dashboard.components.ui import (  # noqa: E402
    bootstrap,
    empty_state,
    kpi_row,
    money,
    percent,
    require,
    sidebar_status,
)

bootstrap("Overview")

st.caption(
    "Dynamic pricing across the estate. Every figure on this page is served by "
    f"the API at `{api_client.API_BASE_URL}` -- the dashboard holds no business "
    "logic of its own."
)

sidebar_status()

# --------------------------------------------------------------------------- #
# Estate summary
# --------------------------------------------------------------------------- #

# One call, not one per hotel. `include_performance` makes the list endpoint
# compute occupancy, ADR and RevPAR for the whole catalogue in two grouped
# queries; the previous version fetched each hotel's detail separately, which
# was 9 HTTP calls and 16 database round trips to render one page.
hotels = require(
    api_client.list_hotels(include_performance=True), "Could not load hotels"
)

if not hotels:
    empty_state(
        "No hotels found.",
        "Seed the database first: `python scripts/seed_database.py`",
    )
    st.stop()

frame = pd.DataFrame(hotels)

# An older API that does not know include_performance simply omits these
# columns. Rendering "no trading data" is the correct response to that; raising
# a KeyError is not, and this module's whole premise is that failure is rendered
# rather than raised. Caught here because a version skew between a deployed API
# and a redeployed dashboard is normal, not exceptional.
TRADING_COLUMNS = ["occupancy_last_30_days", "adr_last_30_days", "revpar_last_30_days"]
for column in TRADING_COLUMNS:
    if column not in frame.columns:
        frame[column] = None

occupied = frame.dropna(subset=["occupancy_last_30_days"])

kpi_row(
    [
        ("Hotels", str(len(frame)), None),
        ("Rooms", f"{int(frame['total_rooms'].sum()):,}", None),
        ("Cities", str(frame["city"].nunique()), None),
        (
            "Occupancy (30d)",
            percent(occupied["occupancy_last_30_days"].mean()) if not occupied.empty else "n/a",
            None,
        ),
        (
            "ADR (30d)",
            money(occupied["adr_last_30_days"].mean()) if not occupied.empty else "n/a",
            None,
        ),
        (
            "RevPAR (30d)",
            money(occupied["revpar_last_30_days"].mean()) if not occupied.empty else "n/a",
            None,
        ),
    ]
)

st.divider()

# --------------------------------------------------------------------------- #
# Estate table
# --------------------------------------------------------------------------- #

left, right = st.columns([3, 2])

with left:
    st.subheader("Properties")
    display = frame[
        [
            "hotel_id", "hotel_name", "city", "star_rating", "total_rooms",
            "segment", "occupancy_last_30_days", "adr_last_30_days",
            "revpar_last_30_days",
        ]
    ].rename(
        columns={
            "hotel_id": "ID",
            "hotel_name": "Hotel",
            "city": "City",
            "star_rating": "Stars",
            "total_rooms": "Rooms",
            "segment": "Segment",
            "occupancy_last_30_days": "Occupancy",
            "adr_last_30_days": "ADR",
            "revpar_last_30_days": "RevPAR",
        }
    )
    st.dataframe(
        display.style.format(
            {"Occupancy": "{:.1%}", "ADR": "{:,.0f}", "RevPAR": "{:,.0f}"},
            na_rep="n/a",
        ),
        use_container_width=True,
        hide_index=True,
    )

with right:
    st.subheader("RevPAR by property")
    if occupied.empty:
        empty_state("No trading data yet.")
    else:
        ranked = occupied.sort_values("revpar_last_30_days")
        figure = px.bar(
            ranked,
            x="revpar_last_30_days",
            y="hotel_id",
            orientation="h",
            color="segment",
            hover_data=["hotel_name", "city"],
            labels={"revpar_last_30_days": "RevPAR (INR)", "hotel_id": ""},
        )
        figure.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", y=1.05),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(figure, use_container_width=True)

    st.caption(
        "RevPAR -- occupancy x ADR -- is the number hotels manage against. "
        "Occupancy alone rewards giving rooms away; ADR alone rewards an empty "
        "hotel with one expensive suite sold."
    )

st.divider()

# --------------------------------------------------------------------------- #
# Model status
# --------------------------------------------------------------------------- #

st.subheader("Models in service")
models = api_client.models()

if not models.ok:
    empty_state("Could not read the model registry.", models.error)
elif not models.data.get("active_version"):
    empty_state(
        "No models have been trained yet -- pricing is running on the historical fallback.",
        "Train them with `python scripts/train_models.py`, or from the Model Performance page.",
    )
else:
    body = models.data
    kpi_row(
        [
            ("Active version", body["active_version"], None),
            ("Serving", ", ".join(body["available"]), None),
            ("Feature version", body["feature_version"], None),
            ("Versions on disk", str(len(body["versions"])), None),
        ]
    )
    if body.get("errors"):
        st.warning(
            "The registry reported problems: "
            + "; ".join(f"{k}: {v}" for k, v in body["errors"].items())
        )
