"""HTTP route modules.

Phase 1 ships ``health``. Phase 8 adds ``pricing``, ``hotels``, ``forecast``,
``models`` and ``competitors``. ``ingestion`` exposes the competitor feed's
configuration and lets a pass be run on demand.
"""

from api.routes import health

__all__ = ["health"]
