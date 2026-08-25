"""Health and readiness endpoints.

The probes are protocol-level, not socket-level. A TCP connect to port 5432 only
proves something is listening; ``SELECT 1`` proves the database will actually
answer a query, and a Kafka metadata request proves the cluster has elected a
controller rather than merely opened a socket. In KRaft mode those two states are
genuinely different, and the gap between them is exactly when a container
healthcheck is most likely to fire.

The endpoint always returns 200 while the process is alive. A dependency being
down downgrades ``status`` to ``degraded`` and is reported in the payload -- but
does not fail the request, because a container healthcheck that kills the API for
being unable to reach Postgres turns a database blip into an outage.
"""

from __future__ import annotations

import socket
import time
from typing import List, Tuple

from fastapi import APIRouter, Depends

from api.schemas import (
    DependencyState,
    DependencyStatus,
    HealthResponse,
    ServiceStatus,
)
from config import Settings, get_settings
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])

#: Deliberately short. A health check must never be the slow part of a deploy,
#: and every dependency here is either local or in the same network -- half a
#: second is already generous. Load balancers typically give up at two to five.
_PROBE_TIMEOUT_SECONDS = 0.5


def _split_host_port(target: str, default_port: int) -> Tuple[str, int]:
    """Split ``host:port``, tolerating a bare host and IPv6 brackets."""
    target = target.strip()
    if target.startswith("["):  # [::1]:9092
        host, _, rest = target[1:].partition("]")
        port = rest.lstrip(":")
        return host, int(port) if port.isdigit() else default_port
    host, _, port = target.rpartition(":")
    if not host:
        return target, default_port
    return host, int(port) if port.isdigit() else default_port


def _probe_tcp(name: str, host: str, port: int) -> DependencyStatus:
    """Attempt a TCP connection and report the outcome."""
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=_PROBE_TIMEOUT_SECONDS):
            elapsed = (time.perf_counter() - started) * 1000
            return DependencyStatus(
                name=name,
                state=DependencyState.UP,
                target=f"{host}:{port}",
                latency_ms=round(elapsed, 2),
            )
    except OSError as exc:
        elapsed = (time.perf_counter() - started) * 1000
        # Expected during startup while dependencies come online -- info, not error.
        logger.info("Dependency probe failed: %s (%s:%s) -- %s", name, host, port, exc)
        return DependencyStatus(
            name=name,
            state=DependencyState.DOWN,
            target=f"{host}:{port}",
            latency_ms=round(elapsed, 2),
            detail=str(exc),
        )


def _probe_database(settings: Settings) -> DependencyStatus:
    """TCP first, then ``SELECT 1``.

    Two steps rather than one, for two reasons. Speed: the pooled engine carries
    ``POSTGRES_CONNECT_TIMEOUT`` (ten seconds by default), which is the right
    number for a request that must succeed and a terrible one for a health check
    that must answer fast -- a cheap socket probe fails in half a second instead.
    Precision: "nothing is listening on 5432" and "something is listening but it
    will not answer a query" are different problems with different fixes, and
    only the two-step probe can tell them apart.
    """
    target = f"{settings.database.host}:{settings.database.port}"

    tcp = _probe_tcp("postgres", settings.database.host, settings.database.port)
    if tcp.state is not DependencyState.UP:
        return tcp

    from database.connection import ping

    started = time.perf_counter()
    answered = ping()
    elapsed = round(tcp.latency_ms + (time.perf_counter() - started) * 1000, 2)

    if answered:
        return DependencyStatus(
            name="postgres", state=DependencyState.UP, target=target, latency_ms=elapsed
        )
    return DependencyStatus(
        name="postgres",
        state=DependencyState.DOWN,
        target=target,
        latency_ms=elapsed,
        detail="the port is open but the server did not answer SELECT 1",
    )


def _probe_kafka(settings: Settings) -> DependencyStatus:
    """A metadata request. In KRaft mode an open socket is not a ready cluster."""
    if not settings.kafka.enabled:
        return DependencyStatus(
            name="kafka",
            state=DependencyState.DISABLED,
            target=settings.kafka.bootstrap_servers,
            detail="KAFKA_ENABLED=false",
        )

    target = settings.kafka.bootstrap_servers
    host, port = _split_host_port(settings.kafka.bootstrap_servers_list[0], 9092)

    # Same two-step shape as the database probe, and here the distinction is
    # sharper still: in KRaft mode a broker accepts TCP connections well before
    # it has elected a controller, so "the socket opened" genuinely does not
    # mean "the cluster is ready".
    tcp = _probe_tcp("kafka", host, port)
    if tcp.state is not DependencyState.UP:
        return DependencyStatus(
            name="kafka",
            state=DependencyState.DOWN,
            target=target,
            latency_ms=tcp.latency_ms,
            detail="the broker is not reachable",
        )

    started = time.perf_counter()
    try:
        from streaming.admin import broker_available

        answered = broker_available(settings, timeout_ms=1_500)
    except Exception as exc:  # pragma: no cover - client not installed
        return DependencyStatus(
            name="kafka",
            state=DependencyState.UNAVAILABLE,
            target=target,
            detail=f"{type(exc).__name__}: {exc}",
        )

    elapsed = round(tcp.latency_ms + (time.perf_counter() - started) * 1000, 2)
    if answered:
        return DependencyStatus(
            name="kafka", state=DependencyState.UP, target=target, latency_ms=elapsed
        )
    return DependencyStatus(
        name="kafka",
        state=DependencyState.DOWN,
        target=target,
        latency_ms=elapsed,
        detail="the port is open but the cluster did not return metadata",
    )


def _model_status() -> tuple[DependencyStatus, dict]:
    """Which model artifacts are serving.

    Missing models are ``unavailable`` rather than ``down``: the service works
    without them, on the historical fallback, and the fix is a training run
    rather than an incident.
    """
    from models.model_registry import get_registry

    registry = get_registry()
    registry.ensure_loaded()
    status_payload = registry.status()

    if registry.is_loaded:
        state = DependencyState.UP
        detail = f"serving {', '.join(registry.loaded.available)}"
    else:
        state = DependencyState.UNAVAILABLE
        detail = "no model artifacts are loaded; pricing uses historical fallback"

    return (
        DependencyStatus(
            name="models",
            state=state,
            target=str(registry.artifact_dir),
            detail=detail,
        ),
        status_payload,
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and dependency status",
    description=(
        "Returns 200 whenever the process is alive. Dependency failures are "
        "reported in the payload and downgrade `status` to `degraded`, but do "
        "not fail the request -- a healthcheck that kills the API for being "
        "unable to reach Postgres turns a database blip into an outage."
    ),
)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Report process liveness, dependency reachability and model availability."""
    model_dependency, model_payload = _model_status()

    dependencies: List[DependencyStatus] = [
        _probe_database(settings),
        _probe_kafka(settings),
        model_dependency,
    ]

    # Only a hard DOWN degrades the service. DISABLED is a configuration
    # choice and UNAVAILABLE is a capability the service runs without.
    degraded = any(d.state is DependencyState.DOWN for d in dependencies)

    return HealthResponse(
        status=ServiceStatus.DEGRADED if degraded else ServiceStatus.OK,
        app=settings.app.app_name,
        version=settings.app.app_version,
        environment=settings.app.environment.value,
        dependencies=dependencies,
        models=model_payload,
    )


__all__ = ["router"]
