"""A small online travel agency, served locally, for the scraper to scrape.

WHY THIS EXISTS
---------------
The project needs to demonstrate a *working* web-scraping ingestion path. It
cannot do that against Booking.com or Expedia: both disallow the search paths
these scrapers would need in their ``robots.txt``, and
:class:`~ingestion.scraper_base.HttpCompetitorScraper` honours robots.txt, so
those scrapers correctly refuse to fetch anything. Removing that check to make a
demo work would trade a real guarantee for a screenshot.

So this package serves a real website instead. It is a genuine HTTP server
returning genuine HTML with the kind of markup an OTA actually uses -- nested
containers, ``data-testid`` attributes, prices with currency symbols and
thousands separators, sold-out cards with no price at all. The scraper fetches
it over the network, reads its ``robots.txt``, parses it with CSS selectors, and
raises real parse errors when the markup does not match.

Everything in the scraping stack is exercised for real. The only thing that
changes is *whose* server is on the other end, and this one is ours.

WHAT IT IS NOT
--------------
Not a mock, and not a fixture. Nothing in ``ingestion/`` knows this package
exists; the scraper reaches it over HTTP like any other site, and pointing that
scraper at a different host is a base-URL change.

It is also not a source of ground truth. The rates it serves are generated (see
:mod:`demo_ota.rates`) -- realistic in shape, deterministic by design, and no
more "real market data" than the synthetic generator is. The honest claim this
package supports is "the ingestion path works end to end", not "these are the
prices in Goa".
"""

from demo_ota.rates import RateQuote, quote_rates

__all__ = ["RateQuote", "quote_rates"]
