"""API authentication, rate limiting and response headers.

TWO SCOPES, AND WHY IT IS TWO AND NOT ONE
-----------------------------------------
``read`` reaches every ``GET`` and price *simulations*. ``write`` additionally
reaches everything that changes state: persisted pricing decisions, competitor
submissions, ingestion runs, retraining.

A single key would be simpler and would give away the most useful property here.
The AI agent is issued the **read** key, so "the agent cannot write" stops being
a promise made by ``ai_agent/tools.py``'s allowlist and becomes a fact about the
network. The allowlist is still there and still worth having -- it fails fast and
documents intent -- but it is now defence in depth rather than the only thing
standing between a prompt injection and a written price.

DISABLED BY DEFAULT, MANDATORY IN PRODUCTION
--------------------------------------------
A local demo that refuses to answer without a header is a demo nobody runs, so
``SECURITY_ENABLED`` defaults to false. ``Settings._production_safety`` then
refuses to boot with ``ENVIRONMENT=production`` unless it is true and both keys
have been changed from their shipped values. The convenient default cannot
survive into production by accident.
"""

from __future__ import annotations

import time
from collections import deque
from threading import Lock
from typing import Deque, Dict

from fastapi import Depends, HTTPException, Request, status

from config import Settings, get_settings
from monitoring.logging_config import get_logger

logger = get_logger(__name__)

#: Sent on every response. Cheap, and each one closes a real hole.
SECURITY_HEADERS = {
    # Stop a browser from second-guessing a declared content type -- the vector
    # for turning a JSON response into executable script.
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    # This API returns JSON and never renders HTML, so everything can be denied.
    "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
    "Cache-Control": "no-store",
}


class RateLimiter:
    """A fixed-window-per-caller limiter, in process.

    Deliberately not Redis. A single API process is the deployment this project
    describes, and an external dependency for a limiter that exists to stop one
    misconfigured script would be a service to run, monitor and fail over for no
    gain at this size.

    **The limit is what that buys and what it costs.** With more than one
    replica each holds its own counter, so the effective limit multiplies by the
    replica count. That is documented rather than hidden: a limiter that is
    quietly wrong under horizontal scaling is worse than one whose bound you can
    reason about. Behind a load balancer, enforce the real quota there.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: Dict[str, Deque[float]] = {}
        self._lock = Lock()

    def allow(self, caller: str) -> bool:
        """Whether ``caller`` may make one more request right now."""
        if self.per_minute <= 0:
            return True

        now = time.monotonic()
        cutoff = now - 60.0

        with self._lock:
            window = self._hits.setdefault(caller, deque())
            while window and window[0] < cutoff:
                window.popleft()
            if len(window) >= self.per_minute:
                return False
            window.append(now)

            # Callers that have gone quiet would otherwise accumulate forever --
            # a slow leak keyed by whatever an attacker chooses to send.
            if len(self._hits) > 1024:
                for key in [k for k, v in self._hits.items() if not v]:
                    del self._hits[key]

            return True

    def retry_after(self, caller: str) -> int:
        """Seconds until the caller's oldest request falls out of the window."""
        with self._lock:
            window = self._hits.get(caller)
            if not window:
                return 60
            return max(1, int(60 - (time.monotonic() - window[0])) + 1)


#: One limiter per process, built on first use.
_limiter: RateLimiter | None = None


def get_limiter(settings: Settings | None = None) -> RateLimiter:
    global _limiter
    if _limiter is None:
        settings = settings or get_settings()
        _limiter = RateLimiter(settings.security.rate_limit_per_minute)
    return _limiter


def reset_limiter() -> None:
    """Drop the limiter. For tests, which must not inherit each other's counts."""
    global _limiter
    _limiter = None


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #


def _caller_id(request: Request, presented: str) -> str:
    """What the rate limiter counts against.

    The key when one is presented, so a shared NAT does not throttle everyone
    behind it; the client address otherwise. Only a prefix of the key is used --
    enough to distinguish callers, never enough to reconstruct one from a log.
    """
    if presented:
        return f"key:{presented[:8]}"
    client = request.client
    return f"ip:{client.host if client else 'unknown'}"


def _authenticate(request: Request, settings: Settings, required: str) -> str:
    """Return the caller's scope, or raise.

    Args:
        required: ``"read"`` or ``"write"``.

    Raises:
        HTTPException: 401 with no key or a bad one, 403 when the key is valid
            but read-only, 429 when the caller is over its limit.
    """
    security = settings.security
    presented = request.headers.get(security.header_name, "")

    limiter = get_limiter(settings)
    caller = _caller_id(request, presented)
    if not limiter.allow(caller):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"rate limit of {security.rate_limit_per_minute}/min exceeded",
            headers={"Retry-After": str(limiter.retry_after(caller))},
        )

    if not security.enabled:
        return "write"

    scope = security.scope_for(presented)
    if scope is None:
        # The message never distinguishes "no key" from "wrong key" -- that
        # difference is only useful to somebody guessing.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"a valid {security.header_name} header is required",
            headers={"WWW-Authenticate": security.header_name},
        )

    if required == "write" and scope != "write":
        logger.warning(
            "Read-scoped key attempted %s %s", request.method, request.url.path
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="this key is read-only; a write-scoped key is required",
        )

    return scope


def require_read(
    request: Request, settings: Settings = Depends(get_settings)
) -> str:
    """Any valid key. Reads and simulations."""
    return _authenticate(request, settings, "read")


def require_write(
    request: Request, settings: Settings = Depends(get_settings)
) -> str:
    """The write-scoped key only. Anything that changes state."""
    return _authenticate(request, settings, "write")


__all__ = [
    "SECURITY_HEADERS",
    "RateLimiter",
    "get_limiter",
    "require_read",
    "require_write",
    "reset_limiter",
]
