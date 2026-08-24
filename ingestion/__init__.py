"""Competitor data acquisition and boundary validation.

Defines one interface, ``CompetitorScraper``, with three implementations:
``SyntheticCompetitorGenerator`` (the default), ``BookingScraper`` and
``ExpediaScraper``. The real scrapers are disabled unless explicitly enabled --
see ADR-004 in ``docs/architecture.md``.

This package validates events at the door and hands them to :mod:`streaming`.
It never writes to PostgreSQL directly.

Phase 2 adds the synthetic generator and validator; Phase 3 wires them to Kafka.
"""

__all__: list = []
