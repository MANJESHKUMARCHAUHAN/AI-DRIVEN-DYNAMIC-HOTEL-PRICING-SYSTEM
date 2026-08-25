"""Model definitions, serialisation and the version registry.

Contains fit/predict/save/load and version metadata. It does not decide *when*
to train -- that is :mod:`training`'s job.

Phase 5 adds ``prophet_model``, Phase 6 adds ``gradient_boosting_model``,
Phase 10 adds ``model_registry``. Serialised artifacts live in
``models/artifacts/`` (gitignored).
"""

__all__: list = []
