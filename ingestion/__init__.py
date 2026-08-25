"""Competitor data acquisition, synthetic data generation and boundary validation.

Two distinct jobs live here, and they are easy to confuse:

:mod:`ingestion.synthetic_dataset`
    Builds the *historical* dataset -- months of hotels, rooms, bookings and
    competitor rates -- that the models are trained on. Batch, offline, run once
    by ``scripts/generate_data.py``.

:mod:`ingestion.synthetic_generator`
    Emits *live* competitor price events onto Kafka, one at a time, forever.
    This is the default implementation of the ``CompetitorScraper`` interface;
    ``BookingScraper`` and ``ExpediaScraper`` are the opt-in alternatives and
    are disabled unless explicitly enabled (ADR-004 in docs/architecture.md).

Both share one demand model, :func:`~ingestion.synthetic_dataset.demand_index_for`,
so the streamed events and the trained-on history come from the same
distribution.

Nothing in this package writes to PostgreSQL. Events are validated at the door
(:mod:`ingestion.validator`) and handed to :mod:`streaming`, which owns
persistence.
"""

from ingestion.demo_ota_scraper import DemoOTAScraper
from ingestion.scraper_base import (
    CompetitorScraper,
    HttpCompetitorScraper,
    ScrapeRequest,
    ScraperBlocked,
    ScraperDisabled,
    ScraperError,
    ScraperParseError,
    build_requests,
    get_scraper,
)
from ingestion.synthetic_dataset import (
    HOTEL_CATALOG,
    ROOM_MIX,
    HotelProfile,
    RoomSpec,
    SyntheticDataset,
    SyntheticDatasetGenerator,
    demand_index_for,
    generate_dataset,
    summarise,
)
from ingestion.synthetic_generator import SyntheticCompetitorGenerator
from ingestion.validator import EventRejected, EventValidator, ReferenceData

__all__ = [
    "HOTEL_CATALOG",
    "ROOM_MIX",
    "CompetitorScraper",
    "DemoOTAScraper",
    "EventRejected",
    "EventValidator",
    "HotelProfile",
    "HttpCompetitorScraper",
    "ReferenceData",
    "RoomSpec",
    "ScrapeRequest",
    "ScraperBlocked",
    "ScraperDisabled",
    "ScraperError",
    "ScraperParseError",
    "SyntheticCompetitorGenerator",
    "SyntheticDataset",
    "SyntheticDatasetGenerator",
    "build_requests",
    "demand_index_for",
    "generate_dataset",
    "get_scraper",
    "summarise",
]
