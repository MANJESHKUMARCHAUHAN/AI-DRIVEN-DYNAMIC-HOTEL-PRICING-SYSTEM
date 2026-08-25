"""Feature engineering: raw rows in, model-ready matrix out.

Functions here are deterministic and side-effect free -- same input, same
output, no network, no clock reads. That property is what makes train/serve
parity testable.

The ordered feature list produced here is persisted with every model artifact
and re-validated at inference time to catch train/serve skew.

:mod:`features.calendars` lands early, in Phase 2, because the data generator and
the feature pipeline must agree *exactly* on what ``season``, ``holiday_flag``
and ``local_event_score`` mean. One definition, imported by both.

Phase 4 adds :mod:`features.feature_engineering` and :mod:`features.feature_store`.
"""

from features.calendars import (
    CityEvent,
    Holiday,
    event_names,
    event_score,
    holiday_name,
    holiday_on,
    holiday_proximity,
    is_holiday,
    is_weekend,
    season_of,
)
from features.feature_engineering import (
    FEATURE_COLUMNS,
    FEATURE_VERSION,
    TARGET_COLUMN,
    FeatureBuilder,
    FeatureConfig,
    build_serving_row,
    describe_features,
)
from features.feature_store import (
    FeatureStore,
    FeatureVersionMismatch,
    load_feature_list,
    save_feature_list,
    validate_feature_list,
)

__all__ = [
    "FEATURE_COLUMNS",
    "FEATURE_VERSION",
    "TARGET_COLUMN",
    "CityEvent",
    "FeatureBuilder",
    "FeatureConfig",
    "FeatureStore",
    "FeatureVersionMismatch",
    "Holiday",
    "build_serving_row",
    "describe_features",
    "event_names",
    "event_score",
    "holiday_name",
    "holiday_on",
    "holiday_proximity",
    "is_holiday",
    "is_weekend",
    "load_feature_list",
    "save_feature_list",
    "season_of",
    "validate_feature_list",
]
