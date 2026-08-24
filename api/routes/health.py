"""Health and readiness endpoints.

The probes here are real TCP connects, not placeholders. A socket connect is
enough to answer "is the broker/database reachable from this container", needs
no driver, and costs a millisecond -- which is exactly what a health endpoint
should do. Phase 2 and Phase 3 deepen these into genuine protocol-level checks
(``SELECT 1``, broker API-versions) once the clients exist.
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

#: Deliberately short. A health check must never be the slow part of a deploy.
_PROBE_TIMEOUT_SECONDS = 1.5


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


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and dependency status",
    description=(
        "Returns 200 whenever the process is alive. Dependency failures are "
        "reported in the payload and downgrade `status` to `degraded`, but do "
        "not fail the request."
    ),
)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Report process liveness plus reachability of Postgres and Kafka."""
    dependencies: List[DependencyStatus] = [
        _probe_tcp("postgres", settings.database.host, settings.database.port)
    ]

    if settings.kafka.enabled:
        broker = settings.kafka.bootstrap_servers_list[0]
        host, port = _split_host_port(broker, 9092)
        dependencies.append(_probe_tcp("kafka", host, port))
    else:
        dependencies.append(
            DependencyStatus(
                name="kafka",
                state=DependencyState.DISABLED,
                target=settings.kafka.bootstrap_servers,
                detail="KAFKA_ENABLED=false",
            )
        )

    degraded = any(d.state is DependencyState.DOWN for d in dependencies)

    return HealthResponse(
        status=ServiceStatus.DEGRADED if degraded else ServiceStatus.OK,
        app=settings.app.app_name,
        version=settings.app.app_version,
        environment=settings.app.environment.value,
        dependencies=dependencies,
    )


__all__ = ["router"]
