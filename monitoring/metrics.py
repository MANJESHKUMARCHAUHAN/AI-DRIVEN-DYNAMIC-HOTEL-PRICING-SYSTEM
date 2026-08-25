"""Prometheus metrics for the running service.

The rest of :mod:`monitoring` answers *"is the data and the model healthy?"* on a
nightly cadence, writing a JSON report. This answers a different question --
*"is the service behaving right now?"* -- continuously, in the format every
alerting stack already speaks.

WHAT IS MEASURED, AND WHY EACH ONE
----------------------------------
``http_request_duration_seconds``   the latency budget the whole design rests on
``http_requests_total``             error rate, by route and status
``pricing_decisions_total``         throughput, split by whether it persisted
``pricing_guardrail_hits_total``    **the interesting one.** A guardrail firing
                                    occasionally is the system working; one
                                    firing on most decisions means the model
                                    wants prices the business will not allow.
                                    That is a retuning signal and it is
                                    invisible unless somebody counts.
``model_version_info``              which artifacts are actually being served
``ingestion_observations_total``    competitor feed liveness, by source
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "Request latency by route.",
    ["method", "path"],
    # Tuned to the budget this system claims: a priced room-night is ~28 ms, so
    # the interesting resolution is below 100 ms. The library's default buckets
    # top out too coarse to show a regression from 28 ms to 90 ms.
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

REQUESTS = Counter(
    "http_requests_total",
    "Requests by route and status class.",
    ["method", "path", "status"],
)

PRICING_DECISIONS = Counter(
    "pricing_decisions_total",
    "Pricing decisions produced, split by whether they were persisted.",
    ["persisted"],
)

GUARDRAIL_HITS = Counter(
    "pricing_guardrail_hits_total",
    "Times each guardrail changed a price.",
    ["rule"],
)

MODEL_INFO = Gauge(
    "model_version_info",
    "1 for the model version currently being served.",
    ["model", "version"],
)

INGESTION_OBSERVATIONS = Counter(
    "ingestion_observations_total",
    "Competitor rate observations collected, by source.",
    ["source"],
)


def observe_request(*, method: str, path: str, status_code: int, seconds: float) -> None:
    """Record one HTTP request.

    ``path`` is the *route template* (``/hotels/{hotel_id}``), never the resolved
    URL. Labelling by resolved path would create one time series per hotel id --
    the cardinality explosion that makes a Prometheus server fall over.
    """
    REQUEST_DURATION.labels(method=method, path=path).observe(seconds)
    REQUESTS.labels(method=method, path=path, status=str(status_code)).inc()


def observe_pricing_decision(
    *, persisted: bool, guardrails: Optional[Iterable[str]] = None
) -> None:
    """Record a pricing decision and any guardrails that fired."""
    PRICING_DECISIONS.labels(persisted=str(persisted).lower()).inc()
    for rule in guardrails or ():
        GUARDRAIL_HITS.labels(rule=str(rule)).inc()


def observe_ingestion(*, source: str, count: int) -> None:
    """Record collected competitor observations."""
    if count:
        INGESTION_OBSERVATIONS.labels(source=source).inc(count)


def set_model_version(*, model: str, version: str) -> None:
    """Publish which artifact version is in service."""
    MODEL_INFO.labels(model=model, version=version).set(1)


def render() -> Tuple[bytes, str]:
    """The exposition payload and its content type."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


__all__ = [
    "CONTENT_TYPE_LATEST",
    "observe_ingestion",
    "observe_pricing_decision",
    "observe_request",
    "render",
    "set_model_version",
]
