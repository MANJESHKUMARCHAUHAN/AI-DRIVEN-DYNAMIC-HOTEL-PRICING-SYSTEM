"""Feature engineering: raw rows in, model-ready matrix out.

Functions here are deterministic and side-effect free -- same input, same
output, no network, no clock reads. That property is what makes train/serve
parity testable.

The ordered feature list produced here is persisted with every model artifact
and re-validated at inference time to catch train/serve skew.

Phase 4 adds :mod:`features.feature_engineering` and :mod:`features.feature_store`.
"""

__all__: list = []
