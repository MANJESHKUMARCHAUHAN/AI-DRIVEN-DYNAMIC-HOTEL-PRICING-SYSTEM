"""Shared domain vocabulary. The innermost layer -- it imports nothing.

WHY THIS PACKAGE EXISTS
-----------------------
These enums used to live in ``database/models.py``, because that is where
SQLAlchemy needs them. The consequence was that ``pricing/`` -- the one package
that must stay free of persistence concerns -- imported the persistence module,
purely to say what a room type is.

Nothing broke. The enums are plain ``str`` subclasses with no ORM machinery, so
the import was harmless at runtime. But it made "pricing depends on the database
layer" true in the import graph, and the architecture test that asserts
otherwise could not be written without either failing or being weakened into
something that no longer meant anything.

Moving the vocabulary one layer down fixes the direction rather than the
assertion: ``database/`` and ``pricing/`` now both depend on ``domain/``, which
depends on nothing at all.

``database.models`` re-exports every name here, so existing imports keep
working -- there is one definition, in one place, reachable by both paths.
"""

from domain.enums import (
    Competitor,
    MarketSegment,
    ModelType,
    RoomType,
    RunStatus,
    Season,
)

__all__ = [
    "Competitor",
    "MarketSegment",
    "ModelType",
    "RoomType",
    "RunStatus",
    "Season",
]
