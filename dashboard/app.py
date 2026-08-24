"""Streamlit analytics dashboard -- entry point.

The dashboard is a **read-only consumer of the API**. It never opens a database
connection of its own; every number it renders comes from an HTTP call. That
keeps one implementation of the business rules instead of two.

Phase 1 renders configuration and live API health so the container is verifiable
end to end. The seven analytics pages arrive in Phase 9.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

from config import get_settings

settings = get_settings()

#: Inside Compose this is ``http://api:8000`` -- a service name, never localhost.
API_BASE_URL = os.getenv("API_BASE_URL", f"http://localhost:{settings.api.port}")

st.set_page_config(
    page_title="Dynamic Hotel Pricing",
    page_icon="*",
    layout="wide",
)

st.title("AI-Driven Dynamic Hotel Pricing")
st.caption(
    f"{settings.app.app_name} v{settings.app.app_version} "
    f"| environment: {settings.app.environment.value}"
)

st.info(
    "**Phase 1 — foundation.** Configuration, logging, packaging and container "
    "wiring are in place. Analytics pages arrive in Phase 9.",
    icon=None,
)

left, right = st.columns(2)

with left:
    st.subheader("API connectivity")
    st.write(f"Target: `{API_BASE_URL}`")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "ok":
            st.success(f"API healthy — status `{payload['status']}`")
        else:
            st.warning(f"API reachable but degraded — status `{payload['status']}`")
        st.json(payload)
    except requests.RequestException as exc:
        st.error(f"API unreachable: {exc}")
        st.caption(
            "Expected until the API container is running. "
            "Start it with `docker compose up`."
        )

with right:
    st.subheader("Effective configuration")
    st.caption("Credentials are redacted by `Settings.summary()`.")
    st.json(settings.summary())

st.divider()
st.caption(
    "Guardrails in force: "
    f"{settings.pricing.currency} {settings.pricing.min_price:,.0f} – "
    f"{settings.pricing.max_price:,.0f}, "
    f"max daily change ±{settings.pricing.max_daily_change_percent:.0%}"
)
