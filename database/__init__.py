"""Persistence layer: engine, session lifecycle and ORM models.

Owns *how* data is stored, never *what it means*. Business rules belong in
:mod:`pricing`; this package only knows about tables, sessions and transactions.

Typical use::

    from database import session_scope, Hotel

    with session_scope() as session:
        hotels = session.query(Hotel).all()

Importing this package builds no connections and touches no sockets -- the
engine in :mod:`database.connection` is created on first use, not at import.
:mod:`database.init_db` is deliberately *not* re-exported: schema creation is an
operation you should have to ask for by name.
"""

from database.connection import (
    DatabaseUnavailable,
    create_db_engine,
    database_info,
    dispose_engine,
    get_db,
    get_engine,
    get_sessionmaker,
    ping,
    session_scope,
    wait_for_database,
)
from database.models import (
    ALL_TABLES,
    TABLE_NAMES,
    Base,
    Booking,
    Competitor,
    CompetitorPrice,
    DemandFeature,
    Hotel,
    MarketSegment,
    ModelType,
    ModelVersion,
    Prediction,
    PricingDecision,
    Room,
    RoomType,
    RunStatus,
    Season,
    TrainingRun,
)

__all__ = [
    "ALL_TABLES",
    "TABLE_NAMES",
    "Base",
    "Booking",
    "Competitor",
    "CompetitorPrice",
    "DatabaseUnavailable",
    "DemandFeature",
    "Hotel",
    "MarketSegment",
    "ModelType",
    "ModelVersion",
    "Prediction",
    "PricingDecision",
    "Room",
    "RoomType",
    "RunStatus",
    "Season",
    "TrainingRun",
    "create_db_engine",
    "database_info",
    "dispose_engine",
    "get_db",
    "get_engine",
    "get_sessionmaker",
    "ping",
    "session_scope",
    "wait_for_database",
]
