"""Pydantic request/response schemas for the HTTP edge.

Phase 1 defines only the health contract and the shared error envelope. The
pricing, forecast, hotel and model schemas arrive in Phase 8.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


def _utc_now() -> datetime:
    """Timezone-aware UTC now. The only clock this project reads."""
    return datetime.now(tz=timezone.utc)


class ServiceStatus(str, Enum):
    """Aggregate health of the service."""

    OK = "ok"
    DEGRADED = "degraded"
    ERROR = "error"


class DependencyState(str, Enum):
    """Reachability of a single downstream dependency."""

    UP = "up"
    DOWN = "down"
    DISABLED = "disabled"


class DependencyStatus(BaseModel):
    """Result of probing one dependency."""

    model_config = ConfigDict(protected_namespaces=())

    name: str = Field(description="Dependency identifier, e.g. 'postgres'.")
    state: DependencyState
    target: str = Field(description="host:port that was probed.")
    latency_ms: Optional[float] = Field(
        default=None, description="Probe round-trip in milliseconds."
    )
    detail: Optional[str] = Field(
        default=None, description="Human-readable reason when not 'up'."
    )


class HealthResponse(BaseModel):
    """``GET /health`` payload.

    Always returns HTTP 200 while the process is alive. A dependency being down
    is reported in ``dependencies`` and reflected in ``status``, but does not
    make the endpoint fail -- otherwise the container healthcheck would kill an
    API that is working correctly and merely waiting on Postgres.
    """

    model_config = ConfigDict(protected_namespaces=())

    status: ServiceStatus
    app: str
    version: str
    environment: str
    timestamp: datetime = Field(default_factory=_utc_now)
    dependencies: list[DependencyStatus] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Uniform error envelope for every non-2xx response."""

    model_config = ConfigDict(protected_namespaces=())

    error: str = Field(description="Short machine-readable error code.")
    detail: str = Field(description="Human-readable explanation.")
    correlation_id: Optional[str] = Field(
        default=None, description="Ties this error to the server logs."
    )
    timestamp: datetime = Field(default_factory=_utc_now)
    context: Optional[Dict[str, Any]] = Field(
        default=None, description="Extra, non-sensitive diagnostic fields."
    )


__all__ = [
    "DependencyState",
    "DependencyStatus",
    "ErrorResponse",
    "HealthResponse",
    "ServiceStatus",
]
