"""Operator entry points.

Each script is runnable directly (``python scripts/<name>.py``) and inserts the
project root on ``sys.path`` so it works without installation.

Phase 1: ``check_config``. Later phases add ``generate_data``,
``seed_database``, ``create_topics`` and ``train_models``.
"""

__all__: list = []
