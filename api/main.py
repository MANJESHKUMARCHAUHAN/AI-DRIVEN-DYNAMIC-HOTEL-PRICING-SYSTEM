"""FastAPI application factory and process lifespan.

The HTTP edge is intentionally thin: it validates input, maps domain results to
status codes, and serialises. No pricing arithmetic lives here -- that is
:mod:`pricing`'s job (ADR-003).

Ten endpoints across five routers. ``/health`` is mounted at the root because
container healthchecks and load balancers expect it there; everything else sits
under the configured ``/api/v1`` prefix.

Start-up loads the model artifacts once. It does **not** fail when they are
missing: the API comes up, ``/health`` reports the models as unavailable, and
pricing runs on the historical fallback. An API that refuses to boot before a
training job has run cannot be deployed before it is trained, which is a
deployment order nobody wants to be forced into.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routes import competitors, forecasts, health, hotels, ingestion
from api.routes import models as models_routes
from api.routes import pricing
from api.schemas import ErrorResponse
from api.security import SECURITY_HEADERS, require_read
from monitoring import metrics
from config import Settings, get_settings
from monitoring.logging_config import (
    configure_logging,
    get_correlation_id,
    get_logger,
    set_correlation_id,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start-up and shut-down work.

    Directories are created here rather than at import time so that merely
    importing the module (as tests and tooling do) has no filesystem side effects.
    """
    settings: Settings = get_settings()
    configure_logging(settings)
    settings.ensure_directories()

    logger.info("Starting %s v%s", settings.app.app_name, settings.app.app_version)
    for key, value in settings.summary().items():
        logger.info("config | %-22s = %s", key, value)

    # Load the models once, here rather than on the first request, so the first
    # caller does not pay a second of Prophet deserialisation. Failure is
    # logged and tolerated -- see the module docstring.
    from models.model_registry import get_registry

    loaded = get_registry(settings).load()
    if loaded.any_loaded:
        logger.info(
            "Serving model version %s (%s)", loaded.version, ", ".join(loaded.available)
        )
    else:
        logger.warning(
            "No models loaded; pricing will use the historical fallback. "
            "Run scripts/train_models.py or POST %s/models/train",
            settings.api.prefix,
        )

    yield

    from streaming.producer import close_producer

    # Flush whatever is still in the producer's accumulator. linger_ms batches
    # records, so an unflushed exit silently drops the last few events.
    close_producer()
    logger.info("Shutting down %s", settings.app.app_name)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    A factory rather than a module-level singleton so tests can construct an app
    against an alternative configuration.
    """
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.api.title,
        version=settings.app.app_version,
        description=(
            "Dynamic hotel room pricing: demand forecasting, competitor-aware "
            "price optimisation, and business guardrails."
        ),
        lifespan=lifespan,
        docs_url="/docs" if settings.api.docs_enabled else None,
        redoc_url="/redoc" if settings.api.docs_enabled else None,
        openapi_url="/openapi.json" if settings.api.docs_enabled else None,
    )

    # CORS is restricted to the configured origins. Never "*" -- requirement 23.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins_list,
        allow_credentials=settings.api.cors_allow_credentials,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization", "X-Correlation-ID"],
    )

    @app.middleware("http")
    async def correlation_middleware(request: Request, call_next):
        """Bind a correlation id to the request, echo it, and harden the response.

        Both concerns live in one middleware because both must apply to *every*
        response including error paths, and a second middleware would be a
        second thing to keep in the chain in the right order.
        """
        incoming = request.headers.get("X-Correlation-ID")
        cid = set_correlation_id(incoming)

        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started

        response.headers["X-Correlation-ID"] = cid
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)

        # The route TEMPLATE, never the resolved URL. Labelling by
        # /hotels/H001 would create one time series per hotel id -- the
        # cardinality explosion that takes a Prometheus server down.
        route = request.scope.get("route")
        metrics.observe_request(
            method=request.method,
            path=getattr(route, "path", None) or "unmatched",
            status_code=response.status_code,
            seconds=elapsed,
        )
        return response

    @app.get(
        "/metrics",
        include_in_schema=False,
        summary="Prometheus exposition",
    )
    async def prometheus_metrics() -> Response:
        """Scrape target.

        Unauthenticated for the same reason as ``/health``: a scraper is
        infrastructure, not a user. In any real deployment this port is reachable
        only from inside the network, which is where that protection belongs --
        an API key checked into a Prometheus config is not a secret.
        """
        body, content_type = metrics.render()
        return Response(content=body, media_type=content_type)

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=f"http_{exc.status_code}",
                detail=str(exc.detail),
                correlation_id=get_correlation_id(),
            ).model_dump(mode="json"),
            # Propagating exc.headers is not cosmetic. Building a fresh response
            # and dropping them silently discarded `Retry-After` on every 429 and
            # `WWW-Authenticate` on every 401 -- so the API told clients to back
            # off without saying for how long, and challenged them without saying
            # how to authenticate. Both headers are the actionable half of the
            # response, and neither absence produces an error anywhere.
            headers=exc.headers or None,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Only the three fields that help a caller fix their request, and never
        # the raw error objects. Pydantic v2 puts the original exception in
        # ``ctx`` for custom validators, which is not JSON-serialisable -- so
        # echoing ``exc.errors()` wholesale turns every 422 raised by one of our
        # own field validators into a 500. It also echoes ``input`` back, which
        # is how a rejected payload ends up in a log it should not be in.
        errors = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ())[1:])
                or "(body)",
                "message": str(error.get("msg", "")),
                "type": str(error.get("type", "")),
            }
            for error in exc.errors()
        ]
        logger.info(
            "Rejected %s %s: %s",
            request.method,
            request.url.path,
            "; ".join(f"{e['field']}: {e['message']}" for e in errors),
        )
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error",
                detail="Request failed schema validation.",
                correlation_id=get_correlation_id(),
                context={"errors": errors},
            ).model_dump(mode="json"),
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        # Log the full traceback; return an opaque message. Internal details
        # must never reach the client (requirement 23).
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error",
                detail="An internal error occurred.",
                correlation_id=get_correlation_id(),
            ).model_dump(mode="json"),
        )

    # /health sits at the root: container healthchecks and load balancers look
    # for it there, and it must keep answering even if the API prefix changes.
    # Left unauthenticated for the same reason -- a probe cannot carry a key.
    app.include_router(health.router)

    # Every business endpoint requires at least a read-scoped key. Applied here
    # rather than decorating ~12 handlers, so a new route is authenticated the
    # moment it is added rather than whenever somebody remembers. The endpoints
    # that change state add `require_write` individually.
    for router in (
        hotels.router,
        pricing.router,
        forecasts.router,
        competitors.router,
        ingestion.router,
        models_routes.router,
    ):
        app.include_router(
            router,
            prefix=settings.api.prefix,
            dependencies=[Depends(require_read)],
        )

    return app


app = create_app()


__all__ = ["app", "create_app", "lifespan"]
