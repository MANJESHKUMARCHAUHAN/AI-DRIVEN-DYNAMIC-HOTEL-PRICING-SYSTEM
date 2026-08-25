"""HTTP client the dashboard uses to reach the API.

The dashboard holds **no business logic and no database connection**. Every
number it renders comes from an HTTP call, which means there is one
implementation of "what is the right price" rather than two that drift apart.
It also means the dashboard is a genuine integration test of the API: if a page
renders, the endpoint behind it works.

Two things this module is careful about:

**Failure is rendered, not raised.** A dashboard that shows a red Python
traceback when the API restarts is worse than one that shows "the API is not
reachable, here is the URL it tried". Every call returns a result object the
caller can render either way.

**Responses are cached.** Streamlit re-runs the whole script on every widget
interaction, so an uncached client would re-fetch every hotel each time a
dropdown moved. The TTLs are short enough that the data still looks live.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Optional

import requests

#: Inside Compose this is ``http://api:8000`` -- a service name, never
#: localhost, which inside a container means the container itself.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")

#: Prefix every business endpoint sits behind.
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")

#: The dashboard drives every endpoint including retraining, so it carries the
#: WRITE-scoped key. The AI agent deliberately does not -- see ai_agent/tools.py,
#: where the read key is what makes "the agent cannot write" a network fact.
API_KEY = os.getenv("API_WRITE_KEY") or os.getenv("SECURITY_WRITE_KEY", "")

#: Generous enough for a synchronous training call, short enough that a hung
#: API does not freeze the dashboard forever.
DEFAULT_TIMEOUT = 30.0
TRAINING_TIMEOUT = 900.0

#: A collection pass is rate-limited on purpose, so it is slow on purpose.
INGESTION_TIMEOUT = 600.0


@dataclass
class ApiResult:
    """The outcome of one call: data, or a reason there is none."""

    ok: bool
    data: Any = None
    error: Optional[str] = None
    status_code: Optional[int] = None

    def unwrap(self, default: Any = None) -> Any:
        """The data, or ``default`` when the call failed."""
        return self.data if self.ok else default

    @property
    def is_empty(self) -> bool:
        return self.ok and not self.data


def _url(path: str) -> str:
    if path.startswith("/health"):
        return f"{API_BASE_URL}{path}"
    return f"{API_BASE_URL}{API_PREFIX}{path}"


def _request(
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json: Optional[Dict[str, Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> ApiResult:
    """Make one call and turn every failure mode into an :class:`ApiResult`."""
    url = _url(path)
    # Sent unconditionally. With SECURITY_ENABLED=false the API ignores it, so
    # there is no branch here that could be wrong in one of the two modes.
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    try:
        response = requests.request(
            method, url, params=params, json=json, timeout=timeout, headers=headers
        )
    except requests.exceptions.ConnectionError:
        return ApiResult(
            ok=False,
            error=f"Could not reach the API at {API_BASE_URL}. Is it running?",
        )
    except requests.exceptions.Timeout:
        return ApiResult(ok=False, error=f"The API did not respond within {timeout:.0f}s.")
    except requests.exceptions.RequestException as exc:
        return ApiResult(ok=False, error=f"{type(exc).__name__}: {exc}")

    if response.status_code >= 400:
        # The API's error envelope carries a human-readable detail; use it in
        # preference to the raw body, which is noise to a dashboard user.
        try:
            body = response.json()
            detail = body.get("detail") or body.get("error") or response.text
        except ValueError:
            detail = response.text[:300]
        return ApiResult(ok=False, error=str(detail), status_code=response.status_code)

    try:
        return ApiResult(ok=True, data=response.json(), status_code=response.status_code)
    except ValueError:
        return ApiResult(
            ok=False, error="The API returned a non-JSON body.", status_code=response.status_code
        )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


def health() -> ApiResult:
    """Service and dependency status."""
    return _request("GET", "/health", timeout=5.0)


def list_hotels(
    city: Optional[str] = None, *, include_performance: bool = False
) -> ApiResult:
    """The hotel catalogue.

    ``include_performance`` adds 30-day occupancy, ADR and RevPAR to the same
    response. The Overview page uses it to render the whole estate in one call;
    without it that page made one request per hotel.
    """
    params: Dict[str, Any] = {}
    if city:
        params["city"] = city
    if include_performance:
        params["include_performance"] = "true"
    return _request("GET", "/hotels", params=params or None)


def get_hotel(hotel_id: str) -> ApiResult:
    return _request("GET", f"/hotels/{hotel_id}")


def predict_price(
    *,
    hotel_id: str,
    room_type: str,
    check_in_date: date,
    current_price: Optional[float] = None,
    occupancy_rate: Optional[float] = None,
    available_rooms: Optional[int] = None,
    competitor_rate: Optional[float] = None,
    persist: bool = True,
) -> ApiResult:
    """Ask for a price. ``persist=False`` keeps what-ifs out of the audit trail."""
    payload: Dict[str, Any] = {
        "hotel_id": hotel_id,
        "room_type": room_type,
        "check_in_date": check_in_date.isoformat(),
        "persist": persist,
    }
    for key, value in (
        ("current_price", current_price),
        ("occupancy_rate", occupancy_rate),
        ("available_rooms", available_rooms),
        ("competitor_rate", competitor_rate),
    ):
        if value is not None:
            payload[key] = value
    return _request("POST", "/pricing/predict", json=payload)


def pricing_history(
    hotel_id: str, *, room_type: Optional[str] = None, limit: int = 100
) -> ApiResult:
    params: Dict[str, Any] = {"limit": limit}
    if room_type:
        params["room_type"] = room_type
    return _request("GET", f"/pricing/{hotel_id}", params=params)


def forecast(
    hotel_id: str,
    *,
    room_type: str = "deluxe",
    horizon_days: int = 30,
    start_date: Optional[date] = None,
) -> ApiResult:
    params: Dict[str, Any] = {"room_type": room_type, "horizon_days": horizon_days}
    if start_date:
        params["start_date"] = start_date.isoformat()
    return _request("GET", f"/forecast/{hotel_id}", params=params)


def competitors(
    hotel_id: str,
    *,
    room_type: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    limit: int = 500,
) -> ApiResult:
    params: Dict[str, Any] = {"limit": limit}
    if room_type:
        params["room_type"] = room_type
    if start_date:
        params["start_date"] = start_date.isoformat()
    if end_date:
        params["end_date"] = end_date.isoformat()
    return _request("GET", f"/competitors/{hotel_id}", params=params)


def submit_competitor_event(
    *,
    hotel_id: str,
    competitor: str,
    room_type: str,
    check_in_date: date,
    price: float,
) -> ApiResult:
    return _request(
        "POST",
        "/competitors/events",
        json={
            "hotel_id": hotel_id,
            "competitor": competitor,
            "room_type": room_type,
            "check_in_date": check_in_date.isoformat(),
            "price": price,
            "source": "dashboard",
        },
    )


def ingestion_status() -> ApiResult:
    """Competitor feed configuration, crawl permission and row counts."""
    return _request("GET", "/ingestion/status")


def run_ingestion(
    *,
    hotel_ids: Optional[list] = None,
    room_types: Optional[list] = None,
    horizons: Optional[list] = None,
    publish: bool = True,
) -> ApiResult:
    """Run one competitor collection pass.

    Generous timeout: the pass is paced by ``INGESTION_RATE_LIMIT_SECONDS``, so
    even a narrow one takes seconds per request by design. Politeness costs
    wall-clock; the alternative is a scraper that hammers its target.
    """
    body: Dict[str, Any] = {"publish": publish}
    for key, value in (
        ("hotel_ids", hotel_ids),
        ("room_types", room_types),
        ("horizons", horizons),
    ):
        if value:
            body[key] = value
    return _request("POST", "/ingestion/run", json=body, timeout=INGESTION_TIMEOUT)


def models() -> ApiResult:
    return _request("GET", "/models")


def train(
    *,
    test_days: int = 60,
    train_prophet: bool = True,
    train_gradient_boosting: bool = True,
    backtest_folds: int = 0,
) -> ApiResult:
    """Trigger a retrain. Synchronous and slow -- see the API docstring."""
    return _request(
        "POST",
        "/models/train",
        json={
            "test_days": test_days,
            "train_prophet": train_prophet,
            "train_gradient_boosting": train_gradient_boosting,
            "backtest_folds": backtest_folds,
        },
        timeout=TRAINING_TIMEOUT,
    )


__all__ = [
    "API_BASE_URL",
    "API_PREFIX",
    "ApiResult",
    "competitors",
    "forecast",
    "get_hotel",
    "health",
    "ingestion_status",
    "list_hotels",
    "models",
    "predict_price",
    "pricing_history",
    "run_ingestion",
    "submit_competitor_event",
    "train",
]