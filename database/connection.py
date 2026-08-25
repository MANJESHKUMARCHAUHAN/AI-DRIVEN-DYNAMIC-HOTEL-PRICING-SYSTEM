"""Engine construction, session lifecycle and connection health.

One engine per process, created lazily on first use. Creating it at import time
would open sockets merely because a module was imported -- which breaks
``--help``, breaks unit tests that never touch the database, and makes the API
container die at import instead of reporting an unhealthy dependency.

Three ways to get a session, deliberately distinct:

``session_scope()``
    Context manager for scripts and background workers. Commits on success,
    rolls back on any exception, always closes.

``get_db()``
    FastAPI dependency. Yields a session and closes it; it does **not** commit,
    because a request handler decides for itself whether its work should be
    durable.

``get_sessionmaker()``
    The factory, for code that needs to manage its own lifecycle (the Kafka
    consumer commits its offset only after the database transaction succeeds).
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Dict, Iterator, Optional

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from config import Settings, get_settings
from monitoring.logging_config import get_logger

logger = get_logger(__name__)


class DatabaseUnavailable(RuntimeError):
    """Raised when the database cannot be reached after the configured retries."""


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def _engine_kwargs(url: str, settings: Settings) -> Dict[str, Any]:
    """Backend-appropriate engine options.

    SQLite ignores connection pooling and has no ``connect_timeout``; passing
    PostgreSQL's pool arguments to it raises. In-memory SQLite additionally needs
    :class:`~sqlalchemy.pool.StaticPool` so that every session in the process
    sees the *same* database rather than a fresh empty one per connection.
    """
    if _is_sqlite(url):
        kwargs: Dict[str, Any] = {
            "echo": settings.database.echo,
            "future": True,
            "connect_args": {"check_same_thread": False},
        }
        if ":memory:" in url or "mode=memory" in url:
            kwargs["poolclass"] = StaticPool
        return kwargs

    return {
        "echo": settings.database.echo,
        "future": True,
        "pool_size": settings.database.pool_size,
        "max_overflow": settings.database.max_overflow,
        "pool_timeout": settings.database.pool_timeout,
        "pool_recycle": settings.database.pool_recycle,
        # Cheap SELECT 1 before handing a pooled connection out. Without it, a
        # connection killed by a container restart surfaces as a random
        # OperationalError halfway through a request.
        "pool_pre_ping": settings.database.pool_pre_ping,
        "connect_args": {
            "connect_timeout": settings.database.connect_timeout,
            "application_name": settings.app.app_name,
        },
    }


def create_db_engine(
    url: Optional[str] = None, settings: Optional[Settings] = None
) -> Engine:
    """Build a new :class:`~sqlalchemy.Engine`.

    Args:
        url: Override DSN. Defaults to the configured PostgreSQL URL. Tests pass
            a SQLite URL here.
        settings: Configuration to read pool sizing and echo behaviour from.

    Returns:
        A configured engine. No connection is opened until it is first used.
    """
    settings = settings or get_settings()
    url = url or settings.database.url

    engine = create_engine(url, **_engine_kwargs(url, settings))

    if _is_sqlite(url):
        # SQLite ships with foreign keys switched *off*. Without this the
        # composite FKs in models.py would silently not be enforced, and the
        # model tests would pass while the real schema rejected the same data.
        @event.listens_for(engine, "connect")
        def _enable_sqlite_fk(dbapi_connection, _record):  # type: ignore[no-untyped-def]
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    logger.debug("Engine created for %s", settings.database.safe_url)
    return engine


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the process-wide engine, creating it on first call."""
    return create_db_engine()


@lru_cache(maxsize=1)
def get_sessionmaker() -> sessionmaker[Session]:
    """Return the process-wide session factory.

    ``expire_on_commit=False`` so that attributes remain readable after a commit.
    The default would emit a fresh SELECT for every attribute touched after
    commit -- and in a request handler that returns an ORM object, that is a
    surprise query per field.
    """
    return sessionmaker(
        bind=get_engine(),
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


def dispose_engine() -> None:
    """Close all pooled connections and drop the cached engine.

    Called on API shutdown, and by tests that swap the configured database.
    """
    if get_engine.cache_info().currsize:
        get_engine().dispose()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    logger.debug("Engine disposed")


# --------------------------------------------------------------------------- #
# Sessions
# --------------------------------------------------------------------------- #


@contextmanager
def session_scope(
    factory: Optional[sessionmaker[Session]] = None,
) -> Iterator[Session]:
    """Transactional scope for a unit of work.

    Example::

        with session_scope() as session:
            session.add(Hotel(...))
        # committed here, or rolled back if the block raised
    """
    factory = factory or get_sessionmaker()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session.

    Does not commit: routes that write call ``session.commit()`` explicitly, so a
    read-only handler never holds a write transaction open.
    """
    session = get_sessionmaker()()
    try:
        yield session
    finally:
        session.close()


# --------------------------------------------------------------------------- #
# Health
# --------------------------------------------------------------------------- #


def ping(engine: Optional[Engine] = None) -> bool:
    """Return ``True`` if a trivial query succeeds. Never raises."""
    engine = engine or get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as exc:
        logger.warning("Database ping failed: %s", exc.__class__.__name__)
        return False


def wait_for_database(
    engine: Optional[Engine] = None,
    *,
    retries: int = 30,
    delay_seconds: float = 2.0,
    backoff: float = 1.0,
) -> None:
    """Block until the database answers, or raise :class:`DatabaseUnavailable`.

    Compose starts ``api`` and ``postgres`` concurrently and healthchecks only
    reduce the race, they do not remove it. Every entrypoint that needs the
    database calls this first, so a cold ``docker compose up`` converges instead
    of crash-looping.

    Args:
        retries: Maximum attempts.
        delay_seconds: Initial wait between attempts.
        backoff: Multiplier applied to the delay after each failure. ``1.0``
            keeps the interval constant.
    """
    engine = engine or get_engine()
    wait = delay_seconds

    for attempt in range(1, retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Database reachable (attempt %d/%d)", attempt, retries)
            return
        except SQLAlchemyError as exc:
            if attempt == retries:
                raise DatabaseUnavailable(
                    f"Database unreachable after {retries} attempts: "
                    f"{exc.__class__.__name__}"
                ) from exc
            logger.warning(
                "Database not ready (attempt %d/%d), retrying in %.1fs",
                attempt,
                retries,
                wait,
            )
            time.sleep(wait)
            wait *= backoff


def database_info(engine: Optional[Engine] = None) -> Dict[str, Any]:
    """Server version and pool statistics, for ``/health`` and the dashboard."""
    engine = engine or get_engine()
    info: Dict[str, Any] = {
        "dialect": engine.dialect.name,
        "reachable": False,
    }
    try:
        with engine.connect() as conn:
            if engine.dialect.name == "postgresql":
                info["server_version"] = conn.execute(
                    text("SHOW server_version")
                ).scalar_one()
            else:
                info["server_version"] = ".".join(
                    str(p) for p in engine.dialect.server_version_info or ()
                )
        info["reachable"] = True
    except SQLAlchemyError as exc:
        info["error"] = exc.__class__.__name__

    pool = engine.pool
    # Only QueuePool exposes these; SQLite's StaticPool does not.
    if hasattr(pool, "size"):
        info["pool"] = {
            "size": pool.size(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
        }
    return info


__all__ = [
    "DatabaseUnavailable",
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
