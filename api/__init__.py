"""HTTP edge (FastAPI).

Kept free of imports on purpose: ``api.main`` imports ``api.routes``, so
re-exporting the app from this module would create a partially-initialised
package during import. Import the app explicitly instead::

    from api.main import app
"""

__all__: list = []
