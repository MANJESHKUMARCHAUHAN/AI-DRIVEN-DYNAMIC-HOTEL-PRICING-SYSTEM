"""HTTP route modules.

Phase 1 ships ``health``. Phase 8 adds ``pricing``, ``hotels``, ``forecast``,
``models`` and ``competitors``.
"""

from api.routes import health

__all__ = ["health"]
