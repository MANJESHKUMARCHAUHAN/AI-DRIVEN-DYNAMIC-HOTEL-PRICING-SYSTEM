"""The agent's entire capability surface: four read-only tools over the REST API.

WHY THE ALLOWLIST EXISTS
------------------------
Every tool below is a read or a simulation. That is easy to state and easy to
erode -- a later tool added "just to submit a corrected rate" would quietly turn
a component that cannot alter data into one that can, and the only thing standing
in the way would be a sentence in a system prompt.

So the restriction is enforced one layer below the tools:
:data:`_ALLOWED_CALLS` is a fixed set of ``(method, path-prefix)`` pairs and
:func:`_request` refuses anything else with :class:`ToolCallDenied`. Adding a
write tool does not "just work"; it fails until somebody deliberately edits the
allowlist, which is a diff a reviewer will see.

This is the same reasoning as ``pricing/guardrails.py``'s module-private
construction token. Both times the question was "how do we guarantee X?", and
both times the answer was to make the violating state unrepresentable rather than
merely discouraged. A constraint you can only state in a prompt is not a
constraint -- prompts are advice to a probabilistic system, and prompt injection
is a real path into an agent that reads scraped text.

**In production, add the network layer too.** The allowlist is in-process, so it
binds this code and not a different process holding the same API URL. The
complete control is an API credential with no write scope, which this project
does not have because it ships no auth layer at all. That is a documented gap,
not a solved problem: see ``docs/ai_agent_design.md``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

#: Where the pricing API lives. A Compose service name inside a container.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")

#: The READ-scoped key, never the write one.
#:
#: This is what turns the allowlist below from the only control into defence in
#: depth. With SECURITY_ENABLED=true the API itself refuses this key on every
#: write endpoint and on persist=true -- so a prompt injection, a bug in the
#: allowlist, or an entirely different process holding this key still cannot
#: write a price. That was the documented gap in docs/ai_agent_design.md.
API_KEY = os.getenv("API_READ_KEY") or os.getenv("SECURITY_READ_KEY", "")

#: Long enough for a cold model load, short enough that a wedged API does not
#: hold a chat turn open indefinitely.
REQUEST_TIMEOUT = 20.0

#: (method, path prefix) pairs this package may call. Everything else raises.
#: ``/pricing/predict`` is here because a *simulation* is a legitimate tool --
#: :func:`simulate_price` hardcodes ``persist=False`` and there is no code path
#: through this module that can set it true.
_ALLOWED_CALLS: Tuple[Tuple[str, str], ...] = (
    ("GET", "/hotels"),
    ("GET", "/pricing/"),
    ("GET", "/competitors/"),
    ("GET", "/forecast/"),
    ("GET", "/ingestion/status"),
    ("POST", "/pricing/predict"),
)


class AgentUnavailable(RuntimeError):
    """The agent cannot run: no SDK installed, or no API key configured."""


class ToolCallDenied(RuntimeError):
    """A tool tried to make a call outside :data:`_ALLOWED_CALLS`."""


def _is_allowed(method: str, path: str) -> bool:
    return any(
        method == allowed_method and path.startswith(allowed_path)
        for allowed_method, allowed_path in _ALLOWED_CALLS
    )


def _request(
    method: str,
    path: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Dict[str, Any]] = None,
) -> str:
    """Make one allowlisted call and return its body as text for the model.

    Returns a readable error string rather than raising on HTTP failure: a tool
    result the model can reason about ("that hotel does not exist") produces a
    better answer than an exception that ends the turn.

    Raises:
        ToolCallDenied: If ``(method, path)`` is not in the allowlist. This one
            *does* raise -- it is a programming error in this module, not a
            condition the model should be asked to work around.
    """
    if not _is_allowed(method, path):
        raise ToolCallDenied(
            f"{method} {path} is not in the agent's allowlist. This package is "
            f"read-only by construction; see ai_agent/tools.py."
        )

    url = f"{API_BASE_URL}{API_PREFIX}{path}"
    try:
        response = httpx.request(
            method,
            url,
            params={k: v for k, v in (params or {}).items() if v is not None},
            json=json_body,
            timeout=REQUEST_TIMEOUT,
            headers={"X-API-Key": API_KEY} if API_KEY else {},
        )
    except httpx.ConnectError:
        return f"ERROR: could not reach the pricing API at {API_BASE_URL}."
    except httpx.TimeoutException:
        return f"ERROR: the pricing API did not respond within {REQUEST_TIMEOUT:.0f}s."
    except httpx.HTTPError as exc:
        return f"ERROR: {type(exc).__name__} calling the pricing API: {exc}"

    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text[:400])
        except ValueError:
            detail = response.text[:400]
        return f"ERROR {response.status_code}: {detail}"

    try:
        # Compact JSON: the model reads this, and pretty-printing several hundred
        # competitor rows is a meaningful share of a context window.
        return json.dumps(response.json(), separators=(",", ":"), default=str)
    except ValueError:
        return response.text[:4000]


# --------------------------------------------------------------------------- #
# Tool implementations
#
# These are plain functions so they can be tested and called without the SDK
# installed. `AGENT_TOOLS` wraps them with @beta_tool lazily, at the bottom.
# --------------------------------------------------------------------------- #


def list_hotels(city: Optional[str] = None) -> str:
    """List the hotels in the portfolio, optionally filtered to one city.

    Call this first when the user names a hotel by description ("the Goa
    property") rather than by id, or when you need to know which hotel ids and
    cities exist. Every other tool takes a hotel_id.

    Args:
        city: Restrict to one city, e.g. Goa. Omit for all hotels.
    """
    return _request("GET", "/hotels", params={"city": city})


def get_pricing_history(
    hotel_id: str, room_type: Optional[str] = None, limit: int = 50
) -> str:
    """Past pricing decisions with the full arithmetic breakdown for each.

    This is the tool for any "why was the price X" question. Each decision
    records the base rate, every adjustment and its size, the demand score and
    confidence, and `guardrails_applied` -- the identifiers of the rules that
    actually changed the number. Quote those identifiers verbatim; do not
    paraphrase them into a cause of your own.

    Args:
        hotel_id: Hotel identifier, e.g. H004.
        room_type: One of standard, deluxe, premium, suite. Omit for all.
        limit: How many recent decisions to return, 1-500.
    """
    return _request(
        "GET",
        f"/pricing/{hotel_id}",
        params={"room_type": room_type, "limit": max(1, min(limit, 500))},
    )


def get_competitor_rates(
    hotel_id: str,
    room_type: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Competitor rates observed for a hotel over a range of stay dates.

    Call this whenever a decision's guardrails mention a competitor band, or when
    the user asks about market position. Check `collected_at` on the observations:
    a rate observed thirty hours before the decision is stale, and stale
    competitor data is a common and easily-missed explanation for a surprising
    price.

    Args:
        hotel_id: Hotel identifier.
        room_type: One of standard, deluxe, premium, suite. Omit for all.
        start_date: First stay date, ISO format (YYYY-MM-DD).
        end_date: Last stay date, inclusive, ISO format.
    """
    return _request(
        "GET",
        f"/competitors/{hotel_id}",
        params={
            "room_type": room_type,
            "start_date": start_date,
            "end_date": end_date,
            "limit": 200,
        },
    )


def get_demand_forecast(
    hotel_id: str, room_type: str = "deluxe", horizon_days: int = 14
) -> str:
    """Demand forecast for the next N nights, with prediction intervals.

    Call this to check whether a price reflected genuinely soft demand or
    something else. Read the interval width, not just the point forecast: low
    confidence scales the whole pricing multiplier back towards the base rate
    (floor 0.5), so a wide interval is frequently the entire explanation for a
    price that looks unresponsive to obvious demand.

    Args:
        hotel_id: Hotel identifier.
        room_type: One of standard, deluxe, premium, suite.
        horizon_days: Nights ahead to forecast, 1-90.
    """
    return _request(
        "GET",
        f"/forecast/{hotel_id}",
        params={
            "room_type": room_type,
            "horizon_days": max(1, min(horizon_days, 90)),
        },
    )


def simulate_price(
    hotel_id: str,
    room_type: str,
    check_in_date: str,
    occupancy_rate: Optional[float] = None,
    competitor_rate: Optional[float] = None,
) -> str:
    """Price a room-night WITHOUT recording it. Use for what-if questions.

    This always sends persist=false, so it never enters the audit trail, never
    affects a published rate, and is safe to call repeatedly. Override
    occupancy_rate or competitor_rate to answer "what would we charge if...".

    You cannot change a real price with this or any other tool. If the user wants
    a rate changed, show them what the simulation says and tell them the change
    is theirs to make.

    Args:
        hotel_id: Hotel identifier.
        room_type: One of standard, deluxe, premium, suite.
        check_in_date: The stay date to price, ISO format (YYYY-MM-DD).
        occupancy_rate: Override on-the-books occupancy, 0.0-1.0.
        competitor_rate: Override the competitor reference rate, in INR.
    """
    body: Dict[str, Any] = {
        "hotel_id": hotel_id,
        "room_type": room_type,
        "check_in_date": check_in_date,
        # Not a parameter. If the model could set this, the allowlist above
        # would be the only thing between an explanation and a written price.
        "persist": False,
    }
    if occupancy_rate is not None:
        body["occupancy_rate"] = occupancy_rate
    if competitor_rate is not None:
        body["competitor_rate"] = competitor_rate
    return _request("POST", "/pricing/predict", json_body=body)


def get_ingestion_status() -> str:
    """Where competitor data is coming from and whether it is still arriving.

    Call this when competitor rates look stale, sparse or missing, before
    concluding anything about the market. It reports the configured source, what
    the target's robots.txt permits, and how many observations landed in the last
    hour. Zero recent observations means the feed is down -- which explains a
    stale competitor band far better than any story about competitor behaviour.
    """
    return _request("GET", "/ingestion/status")


#: The plain callables, for tests and for callers without the SDK.
RAW_TOOLS = (
    list_hotels,
    get_pricing_history,
    get_competitor_rates,
    get_demand_forecast,
    simulate_price,
    get_ingestion_status,
)


# --------------------------------------------------------------------------- #
# SDK wiring
# --------------------------------------------------------------------------- #


def _load_anthropic():
    """Import the SDK, or explain precisely what is missing.

    Raises:
        AgentUnavailable: If ``anthropic`` is not installed.
    """
    try:
        import anthropic  # noqa: F401
    except ImportError as exc:
        raise AgentUnavailable(
            "The anthropic SDK is not installed. The pricing system does not "
            "need it; only the AI agent does.\n\n"
            '    pip install -e ".[agent]"'
        ) from exc
    return anthropic


def agent_status() -> Dict[str, Any]:
    """Whether the agent can run, and if not, exactly why.

    Used by the dashboard to render an explanation instead of a traceback, and
    by tests to skip when the environment has no key.
    """
    problems: List[str] = []

    try:
        _load_anthropic()
        sdk_installed = True
    except AgentUnavailable as exc:
        sdk_installed = False
        problems.append(str(exc))

    has_key = bool(os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    if not has_key:
        problems.append(
            "No ANTHROPIC_API_KEY in the environment. Add it to .env "
            "(it is gitignored) or export it before starting the dashboard."
        )

    return {
        "available": sdk_installed and has_key,
        "sdk_installed": sdk_installed,
        "api_key_present": has_key,
        "api_base_url": API_BASE_URL,
        "tool_count": len(RAW_TOOLS),
        "problems": problems,
    }


def build_tools() -> list:
    """Wrap the raw callables as SDK tools.

    Deferred rather than applied at import time so that importing this module --
    which the dashboard does on every page load -- does not require the SDK.

    Raises:
        AgentUnavailable: If ``anthropic`` is not installed.
    """
    anthropic = _load_anthropic()
    return [anthropic.beta_tool(fn) for fn in RAW_TOOLS]


class _LazyToolList:
    """``AGENT_TOOLS`` without importing the SDK until it is indexed.

    Lets ``from ai_agent import AGENT_TOOLS`` succeed in an environment with no
    SDK, so a dashboard page can import the package and *then* render a helpful
    message about what to install.
    """

    def __iter__(self):
        return iter(build_tools())

    def __len__(self) -> int:
        return len(RAW_TOOLS)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<AGENT_TOOLS: {len(RAW_TOOLS)} tools, SDK loaded on use>"


AGENT_TOOLS = _LazyToolList()


__all__ = [
    "AGENT_TOOLS",
    "API_BASE_URL",
    "RAW_TOOLS",
    "AgentUnavailable",
    "ToolCallDenied",
    "agent_status",
    "build_tools",
    "get_competitor_rates",
    "get_demand_forecast",
    "get_ingestion_status",
    "get_pricing_history",
    "list_hotels",
    "simulate_price",
]
