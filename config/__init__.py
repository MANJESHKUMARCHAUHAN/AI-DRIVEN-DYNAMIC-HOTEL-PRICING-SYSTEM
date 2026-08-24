"""Configuration package.

The only supported way to read configuration anywhere in this project::

    from config import get_settings

    settings = get_settings()
"""

from config.settings import (
    ENV_FILE,
    PROJECT_ROOT,
    APISettings,
    AppSettings,
    CompetitorSource,
    DatabaseSettings,
    Environment,
    IngestionSettings,
    KafkaSettings,
    LogFormat,
    LogLevel,
    ModelSettings,
    MonitoringSettings,
    PathSettings,
    PricingSettings,
    Settings,
    get_settings,
    reset_settings,
)

__all__ = [
    "ENV_FILE",
    "PROJECT_ROOT",
    "APISettings",
    "AppSettings",
    "CompetitorSource",
    "DatabaseSettings",
    "Environment",
    "IngestionSettings",
    "KafkaSettings",
    "LogFormat",
    "LogLevel",
    "ModelSettings",
    "MonitoringSettings",
    "PathSettings",
    "PricingSettings",
    "Settings",
    "get_settings",
    "reset_settings",
]
