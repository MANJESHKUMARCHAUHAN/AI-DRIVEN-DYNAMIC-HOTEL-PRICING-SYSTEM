"""API authentication, scopes, rate limiting and response headers.

The property these tests exist to protect is the one the AI agent's design
depends on: **a read-scoped key cannot change anything.** Every other assertion
here supports that one.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from api.dependencies import session_dependency
from api.main import create_app
from api.security import RateLimiter, reset_limiter
from database.models import Hotel, MarketSegment, Room, RoomType

PREFIX = "/api/v1"
READ_KEY = "test-read-key"
WRITE_KEY = "test-write-key"


def _seed(session) -> None:
    session.add(
        Hotel(
            hotel_id="H001",
            hotel_name="Test Grand",
            city="Mumbai",
            star_rating=5,
            total_rooms=100,
            segment=MarketSegment.BUSINESS,
        )
    )
    session.flush()
    for index, (room_type, price) in enumerate(
        [(RoomType.STANDARD, 5000.0), (RoomType.DELUXE, 6400.0)]
    ):
        session.add(
            Room(
                room_id=f"H001-R{index}",
                hotel_id="H001",
                room_type=room_type,
                capacity=2,
                room_count=50,
                base_price=price,
            )
        )
    session.commit()


@pytest.fixture
def secured(settings, db_engine, monkeypatch):
    """A client with authentication switched on and both keys known."""
    monkeypatch.setattr(settings.security, "enabled", True)
    monkeypatch.setattr(settings.security, "read_key", type(settings.security.read_key)(READ_KEY))
    monkeypatch.setattr(settings.security, "write_key", type(settings.security.write_key)(WRITE_KEY))
    monkeypatch.setattr(settings.kafka, "enabled", False)

    # Each test starts with a clean window, or an earlier test's requests count
    # against this one and the failure looks like a flake.
    reset_limiter()

    factory = sessionmaker(bind=db_engine, expire_on_commit=False, future=True)
    with factory() as seeding:
        _seed(seeding)

    def override_session():
        session = factory()
        try:
            yield session
        finally:
            session.close()

    app = create_app(settings)
    app.dependency_overrides[session_dependency] = override_session

    with TestClient(app) as client:
        client.settings = settings  # type: ignore[attr-defined]
        yield client

    reset_limiter()


def _get(client, path, key=None):
    headers = {"X-API-Key": key} if key else {}
    return client.get(f"{PREFIX}{path}", headers=headers)


def _post(client, path, body, key=None):
    headers = {"X-API-Key": key} if key else {}
    return client.post(f"{PREFIX}{path}", json=body, headers=headers)


# --------------------------------------------------------------------------- #
# Authentication
# --------------------------------------------------------------------------- #


class TestAuthentication:
    def test_no_key_is_rejected(self, secured) -> None:
        assert _get(secured, "/hotels").status_code == 401

    def test_wrong_key_is_rejected(self, secured) -> None:
        assert _get(secured, "/hotels", "not-a-real-key").status_code == 401

    def test_the_rejection_does_not_say_which_failure_it_was(self, secured) -> None:
        """"No key" and "wrong key" must read identically.

        The difference is only useful to somebody guessing.
        """
        no_key = _get(secured, "/hotels").json()["detail"]
        bad_key = _get(secured, "/hotels", "wrong").json()["detail"]
        assert no_key == bad_key

    def test_read_key_is_accepted(self, secured) -> None:
        assert _get(secured, "/hotels", READ_KEY).status_code == 200

    def test_write_key_is_accepted_for_reads(self, secured) -> None:
        assert _get(secured, "/hotels", WRITE_KEY).status_code == 200

    @pytest.mark.parametrize("path", ["/health", "/metrics"])
    def test_probes_stay_public(self, secured, path: str) -> None:
        """A healthcheck and a Prometheus scrape cannot carry a credential.

        Both are protected at the network layer in any deployment worth having;
        an API key checked into a Prometheus config is not a secret.
        """
        assert secured.get(path).status_code == 200


# --------------------------------------------------------------------------- #
# Scopes -- the property the agent's design rests on
# --------------------------------------------------------------------------- #


class TestScopes:
    BODY = {"hotel_id": "H001", "room_type": "deluxe", "check_in_date": "2026-12-24"}

    def test_read_key_may_simulate(self, secured) -> None:
        """persist=false changes nothing, so a reader may do it."""
        response = _post(secured, "/pricing/predict", {**self.BODY, "persist": False}, READ_KEY)
        assert response.status_code == 200

    def test_read_key_may_not_persist(self, secured) -> None:
        """The headline guarantee.

        This is what makes "the agent cannot write" a fact about the network
        rather than a claim about ``ai_agent/tools.py``. If this test ever goes
        green for the wrong reason, the agent's read-only guarantee is gone and
        nothing else would notice.
        """
        response = _post(secured, "/pricing/predict", {**self.BODY, "persist": True}, READ_KEY)
        assert response.status_code == 403
        assert "write-scoped" in response.json()["detail"]

    def test_write_key_may_persist(self, secured) -> None:
        response = _post(secured, "/pricing/predict", {**self.BODY, "persist": True}, WRITE_KEY)
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "path,body",
        [
            ("/ingestion/run", {"horizons": [7]}),
            ("/models/train", {}),
            (
                "/competitors/events",
                {
                    "hotel_id": "H001",
                    "competitor": "booking",
                    "room_type": "deluxe",
                    "check_in_date": "2026-12-24",
                    "price": 7000.0,
                },
            ),
        ],
    )
    def test_read_key_is_refused_on_every_write_endpoint(self, secured, path, body) -> None:
        assert _post(secured, path, body, READ_KEY).status_code == 403

    def test_every_mutating_endpoint_is_covered(self, secured) -> None:
        """A new POST must not be able to appear unguarded and unnoticed.

        Enumerates the published surface rather than trusting the list above to
        stay complete -- the failure mode this guards against is somebody adding
        an endpoint, not somebody editing this file.
        """
        paths = secured.get("/openapi.json").json()["paths"]
        mutating = {
            f"{path}"
            for path, methods in paths.items()
            for method in methods
            if method in {"post", "put", "patch", "delete"}
        }
        known = {
            f"{PREFIX}/pricing/predict",       # scope checked in-handler on persist
            f"{PREFIX}/ingestion/run",
            f"{PREFIX}/models/train",
            f"{PREFIX}/competitors/events",
        }
        assert mutating == known, (
            "a mutating endpoint was added or removed -- confirm its scope and "
            f"update this test: {mutating ^ known}"
        )


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


class TestRateLimiter:
    def test_allows_up_to_the_limit(self) -> None:
        limiter = RateLimiter(per_minute=3)
        assert [limiter.allow("caller") for _ in range(3)] == [True, True, True]

    def test_blocks_past_the_limit(self) -> None:
        limiter = RateLimiter(per_minute=3)
        for _ in range(3):
            limiter.allow("caller")
        assert limiter.allow("caller") is False

    def test_callers_are_counted_separately(self) -> None:
        """A shared NAT must not throttle everyone behind it."""
        limiter = RateLimiter(per_minute=2)
        limiter.allow("a")
        limiter.allow("a")
        assert limiter.allow("a") is False
        assert limiter.allow("b") is True

    def test_zero_disables_it(self) -> None:
        limiter = RateLimiter(per_minute=0)
        assert all(limiter.allow("caller") for _ in range(500))

    def test_retry_after_is_bounded(self) -> None:
        limiter = RateLimiter(per_minute=1)
        limiter.allow("caller")
        assert 1 <= limiter.retry_after("caller") <= 61

    def test_over_limit_returns_429_with_retry_after(self, secured) -> None:
        secured.settings.security.rate_limit_per_minute = 2  # type: ignore[attr-defined]
        reset_limiter()

        codes = [_get(secured, "/hotels", READ_KEY).status_code for _ in range(4)]
        assert codes[:2] == [200, 200]
        assert 429 in codes

        limited = next(r for r in (_get(secured, "/hotels", READ_KEY),) if r.status_code == 429)
        assert "Retry-After" in limited.headers


# --------------------------------------------------------------------------- #
# Response hardening
# --------------------------------------------------------------------------- #


class TestSecurityHeaders:
    @pytest.mark.parametrize(
        "header,value",
        [
            ("X-Content-Type-Options", "nosniff"),
            ("X-Frame-Options", "DENY"),
            ("Referrer-Policy", "no-referrer"),
        ],
    )
    def test_present_on_success(self, secured, header: str, value: str) -> None:
        response = _get(secured, "/hotels", READ_KEY)
        assert response.headers[header] == value

    def test_present_on_errors_too(self, secured) -> None:
        """The 401 path must be hardened as much as the 200 path.

        Error responses are the ones an attacker sees most.
        """
        response = _get(secured, "/hotels")
        assert response.status_code == 401
        assert response.headers["X-Content-Type-Options"] == "nosniff"

    def test_correlation_id_survives_the_hardening(self, secured) -> None:
        response = _get(secured, "/hotels", READ_KEY)
        assert response.headers.get("X-Correlation-ID")


# --------------------------------------------------------------------------- #
# Backwards compatibility
# --------------------------------------------------------------------------- #


class TestDisabledByDefault:
    def test_a_local_stack_needs_no_key(self, settings, db_engine, monkeypatch) -> None:
        """A demo that demands a header before it answers is a demo nobody runs.

        The convenient default is only safe because ``Settings._production_safety``
        refuses to boot production without auth -- tested in test_config.py.
        """
        assert settings.security.enabled is False

        monkeypatch.setattr(settings.kafka, "enabled", False)
        reset_limiter()

        factory = sessionmaker(bind=db_engine, expire_on_commit=False, future=True)
        with factory() as seeding:
            _seed(seeding)

        def override_session():
            session = factory()
            try:
                yield session
            finally:
                session.close()

        app = create_app(settings)
        app.dependency_overrides[session_dependency] = override_session

        with TestClient(app) as client:
            assert client.get(f"{PREFIX}/hotels").status_code == 200

        reset_limiter()
