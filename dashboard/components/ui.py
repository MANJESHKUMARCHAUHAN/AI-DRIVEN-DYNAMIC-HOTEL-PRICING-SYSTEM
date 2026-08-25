"""Shared dashboard building blocks: bootstrap, filters, KPI cards, error states.

Streamlit re-runs the entire script on every interaction, so anything shared
between pages has to be a function rather than a module-level object. These are
the pieces every page needs and none of them should re-implement.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Streamlit runs each page as its own top-level script, so the project root is
# not on sys.path. Every page calls bootstrap() first.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CURRENCY = "INR"
ROOM_TYPES = ["standard", "deluxe", "premium", "suite"]
COMPETITORS = ["booking", "expedia", "agoda", "makemytrip"]

#: One palette, used everywhere, so "our price" is the same colour on every
#: chart. A dashboard where the same series changes colour between pages is a
#: dashboard people misread.
COLOURS = {
    "primary": "#2563eb",
    "accent": "#f59e0b",
    "success": "#16a34a",
    "danger": "#dc2626",
    "muted": "#94a3b8",
    "forecast": "#7c3aed",
    "band": "rgba(124, 58, 237, 0.18)",
}


def bootstrap(title: str, icon: str = "H") -> None:
    """Put the project on ``sys.path`` and configure the page. Call this first."""
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    st.set_page_config(page_title=f"{title} | Hotel Pricing", page_icon=icon, layout="wide")
    st.title(title)


def api_error(result, context: str = "") -> None:
    """Render a failed call as an explanation rather than a traceback."""
    from dashboard.api_client import API_BASE_URL

    prefix = f"{context}: " if context else ""
    st.error(f"{prefix}{result.error}")
    if result.status_code is None:
        st.caption(
            f"The dashboard is configured to call **{API_BASE_URL}**. "
            "Start the API with `uvicorn api.main:app --reload`, or set "
            "`API_BASE_URL` if it runs elsewhere."
        )


def require(result, context: str = "") -> Any:
    """Return the data, or render the error and stop the page.

    ``st.stop()`` rather than an exception: a half-rendered page with a
    traceback at the bottom is harder to read than a page that says what is
    wrong and nothing else.
    """
    if not result.ok:
        api_error(result, context)
        st.stop()
    return result.data


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=60, show_spinner=False)
def _cached_hotels() -> List[Dict[str, Any]]:
    from dashboard import api_client

    result = api_client.list_hotels()
    return result.unwrap([]) or []


def hotel_selector(*, key: str = "hotel", label: str = "Hotel") -> Optional[Dict[str, Any]]:
    """Sidebar hotel picker. Returns the selected hotel record."""
    hotels = _cached_hotels()
    if not hotels:
        st.sidebar.warning("No hotels available. Is the API running and the database seeded?")
        return None

    labels = {f"{h['hotel_id']} - {h['hotel_name']}": h for h in hotels}
    chosen = st.sidebar.selectbox(label, list(labels), key=key)
    return labels[chosen]


def room_type_selector(
    *, key: str = "room_type", options: Optional[List[str]] = None, default: str = "deluxe"
) -> str:
    choices = options or ROOM_TYPES
    index = choices.index(default) if default in choices else 0
    return st.sidebar.selectbox("Room type", choices, index=index, key=key)


def date_range_selector(
    *, key: str = "dates", days_back: int = 30, days_forward: int = 30
) -> Tuple[date, date]:
    """Sidebar date range, defaulting to a window around today."""
    today = date.today()
    default = (today - timedelta(days=days_back), today + timedelta(days=days_forward))
    chosen = st.sidebar.date_input("Date range", value=default, key=key)

    if isinstance(chosen, (list, tuple)) and len(chosen) == 2:
        return chosen[0], chosen[1]
    # Streamlit returns a single date mid-edit, before the second is picked.
    single = chosen if isinstance(chosen, date) else today
    return single, single


def horizon_selector(*, key: str = "horizon", default: int = 30) -> int:
    """The specification's headline horizons, plus a couple either side."""
    options = [7, 14, 30, 60, 90]
    index = options.index(default) if default in options else 2
    return st.sidebar.selectbox("Forecast horizon (days)", options, index=index, key=key)


def sidebar_status() -> None:
    """A compact health badge, on every page.

    Worth the space: almost every confusing dashboard state ("why is this
    empty?") is actually "the API is down" or "nothing has been trained yet".
    """
    from dashboard import api_client

    result = api_client.health()
    st.sidebar.divider()

    if not result.ok:
        st.sidebar.error("API unreachable")
        st.sidebar.caption(api_client.API_BASE_URL)
        return

    body = result.data
    states = {d["name"]: d["state"] for d in body.get("dependencies", [])}
    icon = {"ok": "OK", "degraded": "DEGRADED", "error": "ERROR"}.get(body["status"], "?")

    if body["status"] == "ok":
        st.sidebar.success(f"API {icon}")
    else:
        st.sidebar.warning(f"API {icon}")

    st.sidebar.caption(
        " | ".join(f"{name}: {state}" for name, state in states.items())
    )
    active = (body.get("models") or {}).get("active_version")
    st.sidebar.caption(f"model: {active or 'none trained'}")


# --------------------------------------------------------------------------- #
# KPI cards
# --------------------------------------------------------------------------- #


def money(value: Optional[float], currency: str = CURRENCY) -> str:
    return f"{currency} {value:,.0f}" if value is not None else "n/a"


def percent(value: Optional[float], digits: int = 1) -> str:
    return f"{value:.{digits}%}" if value is not None else "n/a"


def kpi_row(cards: List[Tuple[str, str, Optional[str]]]) -> None:
    """A row of metric cards: ``(label, value, delta)``."""
    columns = st.columns(len(cards))
    for column, (label, value, delta) in zip(columns, cards):
        column.metric(label, value, delta)


def empty_state(message: str, hint: str = "") -> None:
    """What to show when a query succeeded but returned nothing."""
    st.info(message)
    if hint:
        st.caption(hint)


__all__ = [
    "COLOURS",
    "COMPETITORS",
    "CURRENCY",
    "PROJECT_ROOT",
    "ROOM_TYPES",
    "api_error",
    "bootstrap",
    "date_range_selector",
    "empty_state",
    "horizon_selector",
    "hotel_selector",
    "kpi_row",
    "money",
    "percent",
    "require",
    "room_type_selector",
    "sidebar_status",
]
