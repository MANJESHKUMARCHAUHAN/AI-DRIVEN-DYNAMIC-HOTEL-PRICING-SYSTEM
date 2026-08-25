"""Shared pytest fixtures.

Tests must not depend on whatever happens to be in the developer's ``.env``.
:func:`isolated_env` gives every test a clean, deterministic environment, and
clears the ``get_settings`` cache on both entry and exit so no configuration
leaks between tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator

import pytest

if TYPE_CHECKING:  # imported for type hints only -- keeps collection fast
    from sqlalchemy import Engine
    from sqlalchemy.orm import Session

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Settings, reset_settings  # noqa: E402

#: A complete, valid environment. Values differ from the shipped defaults so a
#: test asserting on them proves the environment was actually read.
BASE_ENV: Dict[str, str] = {
    "ENVIRONMENT": "local",
    "DEBUG": "false",
    "LOG_LEVEL": "WARNING",
    "LOG_FORMAT": "console",
    "POSTGRES_HOST": "test-db",
    "POSTGRES_PORT": "5433",
    "POSTGRES_DB": "test_pricing",
    "POSTGRES_USER": "test_user",
    "POSTGRES_PASSWORD": "test_secret_value",
    "KAFKA_ENABLED": "true",
    "KAFKA_BOOTSTRAP_SERVERS": "test-kafka:9092",
    "KAFKA_TOPIC_COMPETITOR": "hotel.competitor_prices",
    "KAFKA_TOPIC_BOOKINGS": "hotel.booking_events",
    "KAFKA_TOPIC_DEMAND": "hotel.demand_events",
    "KAFKA_TOPIC_PREDICTIONS": "hotel.price_predictions",
    "API_HOST": "0.0.0.0",
    "API_PORT": "8000",
    "API_CORS_ORIGINS": "http://localhost:8501",
    "MIN_PRICE": "2500",
    "MAX_PRICE": "25000",
    "DEFAULT_BASE_PRICE": "5000",
    "MAX_DAILY_CHANGE_PERCENT": "0.15",
    "INGESTION_SOURCE": "synthetic",
    "INGESTION_ENABLE_REAL_SCRAPERS": "false",
}


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Replace the process environment with :data:`BASE_ENV` for one test.

    Also points ``ENV_FILE`` at a non-existent path so the developer's real
    ``.env`` cannot influence the result, and redirects writable directories
    into ``tmp_path``.
    """
    for key in list(os.environ):
        if key in BASE_ENV or key.startswith(
            ("POSTGRES_", "KAFKA_", "API_", "INGESTION_", "MONITORING_", "MODEL_")
        ):
            monkeypatch.delenv(key, raising=False)

    for key, value in BASE_ENV.items():
        monkeypatch.setenv(key, value)

    # get_settings() resolves ENV_FILE at call time; a non-existent path makes
    # pydantic-settings skip dotenv entirely and read only the environment.
    monkeypatch.setenv("ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "artifacts"))

    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def settings(isolated_env: None) -> Settings:
    """A freshly constructed :class:`Settings` built from :data:`BASE_ENV`."""
    from config import get_settings

    return get_settings()


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
# The schema is created on in-memory SQLite rather than against a container.
# That is only sound because database/models.py deliberately avoids PostgreSQL-
# only constructs -- portable enums, JSON with a JSONB variant, partial indexes
# both backends support. The PostgreSQL path is exercised separately by
# scripts/seed_database.py against a real server.


@pytest.fixture
def db_engine() -> Iterator["Engine"]:
    """A fresh, empty, fully migrated in-memory database per test."""
    from database.connection import create_db_engine
    from database.init_db import create_schema

    engine = create_db_engine("sqlite+pysqlite:///:memory:")
    create_schema(engine)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(db_engine: "Engine") -> Iterator["Session"]:
    """A session bound to :func:`db_engine`, rolled back on teardown."""
    from sqlalchemy.orm import sessionmaker

    factory = sessionmaker(bind=db_engine, expire_on_commit=False, future=True)
    session = factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture(scope="session")
def demo_ota_server() -> Iterator[str]:
    """A real demo OTA on a real port, for the whole test session.

    Deliberately a live HTTP server rather than a mocked transport. The thing
    under test is a *scraper*: robots.txt fetching, status-code mapping,
    connection reuse and HTML parsing are most of its behaviour, and a mocked
    ``httpx`` would skip all of it and still pass. Starting a server costs about
    a second once per session, which is cheaper than the bugs it catches.

    Yields:
        The base URL, e.g. ``http://127.0.0.1:54321``.
    """
    import socket
    import threading
    import time

    import uvicorn

    from demo_ota.app import app

    # Port 0 lets the OS pick a free one, so parallel runs and a developer's
    # already-running demo site do not collide.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 15.0
    while not server.started:
        if time.monotonic() > deadline:  # pragma: no cover - CI safety net
            server.should_exit = True
            raise RuntimeError(f"demo OTA did not start on port {port} within 15s")
        time.sleep(0.05)

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=10)


@pytest.fixture
def demo_ota_settings(
    demo_ota_server: str, monkeypatch: pytest.MonkeyPatch
) -> "Settings":
    """Settings pointed at the live demo OTA, with the rate limiter off.

    ``INGESTION_RATE_LIMIT_SECONDS=0`` because politeness to ourselves is just
    latency: a suite that scrapes fifty pages at two seconds apart takes two
    minutes to tell you nothing extra. The limiter itself is tested separately.
    """
    from config import get_settings, reset_settings

    for key, value in BASE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("INGESTION_SOURCE", "demo_ota")
    monkeypatch.setenv("INGESTION_DEMO_OTA_BASE_URL", demo_ota_server)
    monkeypatch.setenv("INGESTION_DEMO_OTA_LAYOUT", "v1")
    monkeypatch.setenv("INGESTION_RATE_LIMIT_SECONDS", "0")
    monkeypatch.setenv("INGESTION_MAX_RETRIES", "0")

    reset_settings()
    yield get_settings()
    reset_settings()


@pytest.fixture
def seeded_session(db_session: "Session") -> "Session":
    """One hotel with its four room types, committed and ready to reference."""
    from database.models import Hotel, MarketSegment, Room, RoomType

    db_session.add(
        Hotel(
            hotel_id="H001",
            hotel_name="Test Grand",
            city="Mumbai",
            star_rating=5,
            total_rooms=100,
            segment=MarketSegment.BUSINESS,
        )
    )
    db_session.flush()
    for index, (room_type, count, price) in enumerate(
        [
            (RoomType.STANDARD, 45, 5000.0),
            (RoomType.DELUXE, 30, 6400.0),
            (RoomType.PREMIUM, 17, 8250.0),
            (RoomType.SUITE, 8, 13000.0),
        ]
    ):
        db_session.add(
            Room(
                room_id=f"H001-R{index}",
                hotel_id="H001",
                room_type=room_type,
                capacity=2,
                room_count=count,
                base_price=price,
            )
        )
    db_session.commit()
    return db_session
