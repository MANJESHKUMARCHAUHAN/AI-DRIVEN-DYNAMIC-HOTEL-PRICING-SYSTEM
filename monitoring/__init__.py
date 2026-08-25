"""Cross-cutting observability: logging, data drift, model performance.

Phase 1 ships :mod:`monitoring.logging_config`. Drift and model monitors arrive
in Phase 10. Nothing in this package may block the request path.
"""

from monitoring.logging_config import (
    configure_logging,
    get_correlation_id,
    get_logger,
    set_correlation_id,
)

__all__ = [
    "configure_logging",
    "get_correlation_id",
    "get_logger",
    "set_correlation_id",
]
