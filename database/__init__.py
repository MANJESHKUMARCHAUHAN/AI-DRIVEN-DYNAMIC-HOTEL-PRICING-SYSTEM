"""Persistence layer: engine, session lifecycle and ORM models.

Owns *how* data is stored, never *what it means*. Business rules belong in
:mod:`pricing`; this package only knows about tables, sessions and transactions.

Phase 2 adds :mod:`database.connection`, :mod:`database.models` and
:mod:`database.init_db`.
"""

__all__: list = []
