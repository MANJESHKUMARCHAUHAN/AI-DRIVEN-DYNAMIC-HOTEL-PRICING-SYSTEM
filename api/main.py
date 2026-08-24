"""FastAPI application factory and process lifespan.

The HTTP edge is intentionally thin: it validates input, maps domain results to
status codes, and serialises. No pricing arithmetic lives here -- that is
:mod:`pricing`'s job (ADR-003).

Phase 1 mounts only ``/health``. The remaining nine endpoints land in Phase 8;
the middleware, exception handling and lifespan wiring below are already the
production versions.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from api.routes import health
from api.schemas import ErrorResponse
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

    yield

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
        """Bind a correlation id to the request and echo it back."""
        incoming = request.headers.get("X-Correlation-ID")
        cid = set_correlation_id(incoming)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response

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
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error",
                detail="Request failed schema validation.",
                correlation_id=get_correlation_id(),
                context={"errors": exc.errors()},
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

    app.include_router(health.router)

    return app


app = create_app()


__all__ = ["app", "create_app", "lifespan"]
