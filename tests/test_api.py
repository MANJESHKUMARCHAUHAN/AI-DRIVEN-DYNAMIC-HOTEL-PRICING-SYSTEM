"""Phase 1 tests: the FastAPI application shell.

Covers app construction, the health contract, CORS restriction, correlation-id
propagation and the error envelope. The nine business endpoints arrive in Phase 8.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def client(settings, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient bound to an app built from the isolated test settings.

    The dependency hosts are moved to closed ports on loopback. ``BASE_ENV``
    uses the hostnames ``test-db`` and ``test-kafka``, which is right for the
    configuration tests -- a distinctive value proves the environment was
    actually read -- but they do not resolve, and on Windows a failed DNS
    lookup costs several seconds that ``socket``'s connect timeout does not
    bound. A refused connection to loopback fails instantly and exercises
    exactly the same "dependency is down" path.
    """
    monkeypatch.setattr(settings.database, "host", "127.0.0.1")
    monkeypatch.setattr(settings.database, "port", 5399)
    monkeypatch.setattr(settings.kafka, "bootstrap_servers", "127.0.0.1:9399")
    return TestClient(create_app(settings))


class TestAppConstruction:
    def test_app_builds(self, settings) -> None:
        app = create_app(settings)
        assert app.title == settings.api.title
        assert app.version == settings.app.app_version

    def test_openapi_schema_is_valid(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        assert schema["openapi"].startswith("3.")
        assert "/health" in schema["paths"]

    def test_docs_are_served(self, client: TestClient) -> None:
        assert client.get("/docs").status_code == 200


class TestHealthEndpoint:
    def test_returns_200_even_when_dependencies_are_down(
        self, client: TestClient
    ) -> None:
        # test-db and test-kafka do not resolve. The endpoint must still answer,
        # otherwise the container healthcheck would kill a healthy API that is
        # merely waiting for Postgres to come up.
        response = client.get("/health")
        assert response.status_code == 200

    def test_reports_degraded_when_a_dependency_is_down(
        self, client: TestClient
    ) -> None:
        body = client.get("/health").json()
        assert body["status"] == "degraded"

    def test_lists_every_dependency(self, client: TestClient) -> None:
        names = {d["name"] for d in client.get("/health").json()["dependencies"]}
        assert names == {"postgres", "kafka", "models"}

    def test_missing_models_do_not_degrade_the_service(
        self, client: TestClient
    ) -> None:
        """The API works without models, on the historical fallback. That is a
        missing capability, not a fault -- fixed by training, not by paging."""
        deps = {d["name"]: d for d in client.get("/health").json()["dependencies"]}
        assert deps["models"]["state"] in {"up", "unavailable"}

    def test_reports_identity(self, client: TestClient, settings) -> None:
        body = client.get("/health").json()
        assert body["app"] == settings.app.app_name
        assert body["version"] == settings.app.app_version
        assert body["environment"] == "local"

    def test_never_leaks_the_password(self, client: TestClient) -> None:
        assert "test_secret_value" not in client.get("/health").text

    def test_each_dependency_reports_a_target_and_latency(
        self, client: TestClient
    ) -> None:
        for dep in client.get("/health").json()["dependencies"]:
            assert dep["target"]
            assert dep["state"] in {"up", "down", "disabled", "unavailable"}
            assert dep["latency_ms"] is None or dep["latency_ms"] >= 0

    def test_kafka_reported_disabled_when_switched_off(
        self, monkeypatch: pytest.MonkeyPatch, isolated_env: None
    ) -> None:
        from config import get_settings, reset_settings

        monkeypatch.setenv("KAFKA_ENABLED", "false")
        reset_settings()
        with TestClient(create_app(get_settings())) as c:
            deps = {d["name"]: d for d in c.get("/health").json()["dependencies"]}
            assert deps["kafka"]["state"] == "disabled"


class TestCorrelationId:
    def test_response_carries_a_correlation_id(self, client: TestClient) -> None:
        assert client.get("/health").headers.get("X-Correlation-ID")

    def test_incoming_correlation_id_is_echoed(self, client: TestClient) -> None:
        response = client.get("/health", headers={"X-Correlation-ID": "given-id"})
        assert response.headers["X-Correlation-ID"] == "given-id"


class TestErrorHandling:
    def test_unknown_route_uses_the_error_envelope(self, client: TestClient) -> None:
        response = client.get("/does-not-exist")
        assert response.status_code == 404
        body = response.json()
        assert body["error"] == "http_404"
        assert "detail" in body
        assert body["correlation_id"]


class TestCors:
    def test_configured_origin_is_allowed(self, client: TestClient) -> None:
        response = client.get(
            "/health", headers={"Origin": "http://localhost:8501"}
        )
        assert (
            response.headers.get("access-control-allow-origin")
            == "http://localhost:8501"
        )

    def test_unlisted_origin_is_not_allowed(self, client: TestClient) -> None:
        # Requirement 23: CORS is restricted, never "*".
        response = client.get("/health", headers={"Origin": "http://evil.example"})
        assert response.headers.get("access-control-allow-origin") != "*"
        assert (
            response.headers.get("access-control-allow-origin")
            != "http://evil.example"
        )
