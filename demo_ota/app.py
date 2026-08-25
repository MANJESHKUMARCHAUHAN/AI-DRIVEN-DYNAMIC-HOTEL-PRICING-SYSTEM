"""The demo OTA web server.

Serves search-results pages that look like an OTA's: a list of property cards,
each with a name, a room description and a price, wrapped in the sort of nested
markup with ``data-testid`` hooks that real sites use. The scraper parses these
with CSS selectors, exactly as it would parse a real site.

Run it standalone::

    python -m uvicorn demo_ota.app:app --port 8900

Endpoints:

``GET /``                     Landing page, links to a sample search.
``GET /robots.txt``           Crawl policy. **Allows** ``/search``; disallows
                              ``/admin``. This is the real thing the scraper's
                              robots check reads.
``GET /search``               Search results as HTML.
``GET /healthz``              Liveness probe for Compose.

THE ``layout`` PARAMETER
------------------------
``/search?layout=v2`` serves the same rates under completely different markup,
as though the site had been redesigned overnight. Nothing else changes. This is
not a gimmick: "the site redesigned and our selectors now match nothing" is the
single most common way a scraper fails in production, and it is the failure this
system is designed to make loud rather than silent. Being able to trigger it on
demand means the ``ScraperParseError`` path is demonstrable and testable instead
of theoretical.
"""

from __future__ import annotations

import html
from datetime import date, timedelta
from typing import List

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, PlainTextResponse

from demo_ota.rates import ROOM_TYPES, RateQuote, known_cities, quote_rates

app = FastAPI(
    title="Demo OTA",
    description=(
        "A stand-in online travel agency, served locally so the competitor "
        "scraper has a real site to scrape without touching anyone's terms of "
        "service. Rates are generated, not real."
    ),
    version="1.0.0",
    docs_url="/docs",
)


# --------------------------------------------------------------------------- #
# robots.txt
# --------------------------------------------------------------------------- #

#: The crawl policy the scraper actually reads and obeys.
#:
#: ``/search`` is allowed, which is the entire point of this service: the
#: scraper's robots check passes because permission was genuinely granted, not
#: because the check was skipped. ``/admin`` is disallowed so there is a path
#: that demonstrably gets refused -- a test asserts the scraper honours it.
#:
#: ``Crawl-delay`` is advisory in the standard and ignored by
#: ``urllib.robotparser``; ``INGESTION_RATE_LIMIT_SECONDS`` is what actually
#: paces requests. It is stated here because a real site would state it.
#:
#: **Rule order is load-bearing.** ``urllib.robotparser`` implements the original
#: 1994 standard, where the *first* matching rule wins -- not the
#: longest-match-wins behaviour Google documents. An earlier version of this file
#: listed ``Allow: /`` above ``Disallow: /admin``, so ``/admin`` matched the
#: blanket allow first and the disallow was dead text. The scraper dutifully
#: fetched a path it had been told not to. Specific disallows must come first,
#: and there is no blanket ``Allow: /`` at all: anything unmatched is permitted
#: by default, so it bought nothing and silently broke the rule under it.
ROBOTS_TXT = """\
User-agent: *
Disallow: /admin
Allow: /search
Crawl-delay: 1

# This is a demonstration server that ships with the dynamic hotel pricing
# project. Automated access to /search is expressly permitted.
"""


@app.get("/robots.txt", response_class=PlainTextResponse, include_in_schema=False)
def robots() -> str:
    """Crawl policy. Fetched by the scraper before its first request."""
    return ROBOTS_TXT


@app.get("/healthz", response_class=PlainTextResponse, include_in_schema=False)
def healthz() -> str:
    """Liveness probe."""
    return "ok"


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

_STYLE = """\
body { font-family: system-ui, sans-serif; margin: 0; background: #f5f6f8; color: #1a1a2e; }
header { background: #003580; color: #fff; padding: 18px 28px; }
header h1 { margin: 0; font-size: 20px; letter-spacing: .3px; }
header p { margin: 4px 0 0; font-size: 13px; opacity: .85; }
main { padding: 22px 28px; max-width: 980px; }
.summary { font-size: 14px; color: #555; margin-bottom: 18px; }
.property-card { background: #fff; border: 1px solid #e3e5ea; border-radius: 8px;
                 padding: 16px 18px; margin-bottom: 12px; display: flex;
                 justify-content: space-between; align-items: center; }
.property-name { font-size: 16px; font-weight: 600; margin: 0 0 4px; }
.unit-name { font-size: 13px; color: #666; margin: 0; }
.rate-block { text-align: right; }
.rate { font-size: 20px; font-weight: 700; color: #003580; }
.per-night { font-size: 11px; color: #888; display: block; }
.sold-out { font-size: 14px; color: #b00020; font-weight: 600; }
.note { margin-top: 24px; font-size: 12px; color: #777; border-top: 1px solid #e3e5ea;
        padding-top: 14px; }
"""

#: Human-facing unit names, so the scraper's room classifier has realistic text
#: to work on rather than the bare category string it is trying to recover.
_UNIT_LABEL = {
    "standard": "Standard Double Room",
    "deluxe": "Deluxe King Room with City View",
    "premium": "Premium Executive Room, Club Access",
    "suite": "Junior Suite with Balcony",
}


def _card_v1(quote: RateQuote) -> str:
    """A property card in the current markup."""
    name = html.escape(quote.property_name)
    unit = html.escape(_UNIT_LABEL.get(quote.room_type, quote.room_type))

    if quote.price is None:
        rate_block = '<div class="rate-block"><span class="sold-out">Sold out</span></div>'
    else:
        # Thousands separators and a currency symbol, because that is what a
        # real page shows and what the scraper's price regex must survive.
        rate_block = (
            '<div class="rate-block">'
            f'<span class="rate" data-testid="price-and-discounted-price">'
            f"&#8377;&nbsp;{quote.price:,.0f}</span>"
            '<span class="per-night">per night, incl. taxes</span>'
            "</div>"
        )

    return (
        '<div class="property-card" data-testid="property-card">'
        '<div class="property-info">'
        f'<h3 class="property-name" data-testid="title">{name}</h3>'
        f'<p class="unit-name" data-testid="recommended-units">{unit}</p>'
        "</div>"
        f"{rate_block}"
        "</div>"
    )


def _card_v2(quote: RateQuote) -> str:
    """The same card after a hypothetical redesign.

    Every class name, element and ``data-testid`` differs. Nothing the v1
    selectors look for is present, so the scraper raises ``ScraperParseError``
    rather than quietly reporting an empty market.
    """
    name = html.escape(quote.property_name)
    unit = html.escape(_UNIT_LABEL.get(quote.room_type, quote.room_type))

    if quote.price is None:
        rate_block = '<span class="listing__unavailable">No availability</span>'
    else:
        rate_block = (
            f'<span class="listing__cost" data-qa="nightly-cost">'
            f"&#8377;&nbsp;{quote.price:,.0f}</span>"
        )

    return (
        '<article class="listing" data-qa="listing-tile">'
        f'<h4 class="listing__title">{name}</h4>'
        f'<span class="listing__unit">{unit}</span>'
        f"{rate_block}"
        "</article>"
    )


def _page(title: str, summary: str, body: str) -> str:
    return (
        "<!DOCTYPE html>"
        '<html lang="en"><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        f"<style>{_STYLE}</style></head><body>"
        "<header><h1>Demo OTA</h1>"
        "<p>Generated rates, served locally for the pricing system's scraper.</p>"
        "</header><main>"
        f'<div class="summary">{summary}</div>'
        f"{body}"
        '<div class="note">These rates are generated by '
        "<code>demo_ota.rates</code>. They are realistic in shape and are not "
        "real market data.</div>"
        "</main></body></html>"
    )


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> str:
    """Landing page with a worked example, so the service is explorable by hand."""
    tomorrow = (date.today() + timedelta(days=1)).isoformat()
    links = "".join(
        f'<li><a href="/search?city={city.replace(" ", "+")}'
        f'&checkin={tomorrow}&room=deluxe">{html.escape(city)}</a></li>'
        for city in known_cities()
    )
    return _page(
        "Demo OTA",
        "Pick a city to see tomorrow's deluxe rates.",
        f"<ul>{links}</ul>"
        '<p style="font-size:13px;color:#666">'
        "Add <code>&amp;layout=v2</code> to any search to serve the same rates "
        "under redesigned markup, which is how the scraper's parse-error path "
        "is demonstrated.</p>",
    )


@app.get("/search", response_class=HTMLResponse)
def search(
    city: str = Query(..., description="City to search, e.g. Goa."),
    checkin: date = Query(..., description="Check-in date, ISO format."),
    room: str = Query("deluxe", description=f"One of {', '.join(ROOM_TYPES)}."),
    layout: str = Query(
        "v1",
        description="Markup version. 'v2' simulates a site redesign that breaks "
        "the scraper's selectors.",
    ),
) -> HTMLResponse:
    """Search results for one city, night and room category.

    Returns 200 with an empty result list for an unknown room type rather than
    422: a real OTA does not fail a search because a filter matched nothing, and
    the scraper needs to distinguish "no rates" from "page broke".
    """
    if room not in ROOM_TYPES:
        return HTMLResponse(
            _page(
                f"No results -- {city}",
                f"No room category matching {html.escape(room)}.",
                '<div class="summary">0 properties found.</div>',
            )
        )

    quotes: List[RateQuote] = quote_rates(city, checkin, room)
    render = _card_v2 if layout == "v2" else _card_v1
    available = sum(1 for q in quotes if q.is_available)

    summary = (
        f"<strong>{len(quotes)}</strong> properties in "
        f"<strong>{html.escape(city)}</strong> for "
        f"<strong>{checkin.isoformat()}</strong>, "
        f"{html.escape(_UNIT_LABEL.get(room, room))} &mdash; "
        f"{available} with availability."
    )
    body = "".join(render(q) for q in quotes)

    return HTMLResponse(_page(f"{city} -- {checkin.isoformat()}", summary, body))


@app.get("/admin", response_class=PlainTextResponse, include_in_schema=False)
def admin() -> str:
    """A path ``robots.txt`` disallows.

    Reachable by a browser, refused by any crawler that honours robots -- which
    is the assertion ``test_scraper_refuses_disallowed_path`` makes.
    """
    return "Demo OTA admin. robots.txt disallows this path."


__all__ = ["ROBOTS_TXT", "app"]
