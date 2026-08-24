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
from typing import Dict, Iterator

import pytest

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
