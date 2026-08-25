"""Centralised, environment-driven configuration for the pricing platform.

Every tunable in this system lives here. No other module may read ``os.environ``
directly, and no other module may hardcode a host, port, credential, path or
business threshold. That rule is what makes the same code run unchanged on a
laptop, in Docker Compose, and in a real deployment.

The settings tree is composed of focused sub-models rather than one flat blob::

    settings = get_settings()
    settings.database.url          # SQLAlchemy DSN
    settings.kafka.topics          # every topic name
    settings.pricing.min_price     # a guardrail threshold
    settings.model.model_dir       # where artifacts live

Each sub-model is an independent ``BaseSettings``, so it reads its own variables
from the environment (and from ``.env``) using its own prefix. This keeps the
env-var names flat and conventional -- ``POSTGRES_HOST``, ``KAFKA_BOOTSTRAP_SERVERS``,
``MIN_PRICE`` -- while the Python API stays grouped and discoverable.

Secrets are held as :class:`~pydantic.SecretStr`. They never appear in a repr,
never get logged, and are only unwrapped at the moment a connection string is
built.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, computed_field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

#: Repository root. ``config/settings.py`` -> ``config/`` -> project root.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

#: Location of the dotenv file. Overridable so tests can point at a fixture and
#: so containers can mount a different file without changing code.
ENV_FILE: str = os.getenv("ENV_FILE", str(PROJECT_ROOT / ".env"))


def _config(prefix: str = "") -> SettingsConfigDict:
    """Build a settings config block shared by every sub-model.

    ``protected_namespaces=()`` is required because several legitimate field
    names in this project start with ``model_`` (``model_dir``, ``model_version``).
    Pydantic reserves that prefix by default and would emit warnings for each one.
    """
    return SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        env_prefix=prefix,
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
        validate_default=True,
    )


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
# NOTE: ``str, Enum`` rather than ``enum.StrEnum`` -- StrEnum is 3.11+, and the
# codebase stays importable on 3.10 so it can be verified without a container.


class Environment(str, Enum):
    """Deployment context. Drives defaults and safety checks."""

    LOCAL = "local"
    DOCKER = "docker"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """``console`` for humans, ``json`` for log aggregators."""

    CONSOLE = "console"
    JSON = "json"


class CompetitorSource(str, Enum):
    """Which competitor-data implementation the ingestion layer should use.

    ``DEMO_OTA`` scrapes the local :mod:`demo_ota` service over real HTTP --
    real robots.txt, real HTML, real CSS selectors, real parse errors. It is a
    scraper in every sense except that the site on the other end is one we ship,
    which is what lets it run by default.
    """

    SYNTHETIC = "synthetic"
    DEMO_OTA = "demo_ota"
    BOOKING = "booking"
    EXPEDIA = "expedia"

    @property
    def is_third_party(self) -> bool:
        """Whether this source sends traffic to somebody else's website.

        The distinction that matters for :attr:`IngestionSettings.enable_real_scrapers`.
        That flag exists because automated access to Booking.com or Expedia may
        breach their terms of service and the exposure is the operator's -- a
        legal question, not a technical one. It says nothing about whether HTTP
        is involved, so gating ``demo_ota`` on it would conflate two unrelated
        risks and make the *safe* option the one behind the scary flag.
        """
        return self in (CompetitorSource.BOOKING, CompetitorSource.EXPEDIA)


# --------------------------------------------------------------------------- #
# Application
# --------------------------------------------------------------------------- #


class AppSettings(BaseSettings):
    """Process-wide identity and logging behaviour."""

    model_config = _config()

    app_name: str = Field(default="dynamic-hotel-pricing")
    app_version: str = Field(default="0.1.0")
    environment: Environment = Field(default=Environment.LOCAL)
    debug: bool = Field(default=False)

    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_format: LogFormat = Field(default=LogFormat.CONSOLE)
    log_file: Optional[Path] = Field(default=None)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION


# --------------------------------------------------------------------------- #
# PostgreSQL
# --------------------------------------------------------------------------- #


class DatabaseSettings(BaseSettings):
    """PostgreSQL connection and pool configuration.

    Reads ``POSTGRES_*``. The password is a ``SecretStr`` and is URL-encoded when
    the DSN is assembled, so passwords containing ``@``, ``:`` or ``/`` do not
    silently corrupt the connection string.
    """

    model_config = _config("POSTGRES_")

    host: str = Field(default="localhost")
    port: int = Field(default=5432, ge=1, le=65535)
    db: str = Field(default="hotel_pricing")
    user: str = Field(default="pricing")
    password: SecretStr = Field(default=SecretStr("pricing"))

    # Pool tuning -- POSTGRES_POOL_SIZE, POSTGRES_MAX_OVERFLOW, ...
    pool_size: int = Field(default=5, ge=1, le=100)
    max_overflow: int = Field(default=10, ge=0, le=100)
    pool_timeout: int = Field(default=30, ge=1)
    pool_recycle: int = Field(default=1800, ge=-1)
    pool_pre_ping: bool = Field(default=True)
    echo: bool = Field(default=False)
    connect_timeout: int = Field(default=10, ge=1)

    # ``repr=False`` is load-bearing, not cosmetic. Computed fields are included
    # in a model's repr by default, so without it the plaintext password would
    # be printed by any traceback, debugger frame or pytest assertion dump that
    # touched the settings object -- defeating the SecretStr entirely.
    @computed_field(repr=False)  # type: ignore[prop-decorator]
    @property
    def url(self) -> str:
        """SQLAlchemy DSN. Contains the password -- never log this."""
        pwd = quote_plus(self.password.get_secret_value())
        user = quote_plus(self.user)
        return f"postgresql+psycopg2://{user}:{pwd}@{self.host}:{self.port}/{self.db}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def safe_url(self) -> str:
        """DSN with the password masked. This is the one safe to log."""
        return (
            f"postgresql+psycopg2://{self.user}:***@{self.host}:{self.port}/{self.db}"
        )


# --------------------------------------------------------------------------- #
# Kafka
# --------------------------------------------------------------------------- #


class KafkaSettings(BaseSettings):
    """Kafka broker, topic and client configuration.

    Reads ``KAFKA_*``. ``enabled`` exists so the API can run with Kafka absent --
    a missing broker degrades event publishing to a logged warning rather than
    taking the service down.
    """

    model_config = _config("KAFKA_")

    enabled: bool = Field(default=True)
    bootstrap_servers: str = Field(default="localhost:9092")
    client_id: str = Field(default="hotel-pricing")
    consumer_group: str = Field(default="hotel-pricing-consumers")

    topic_competitor: str = Field(default="hotel.competitor_prices")
    topic_bookings: str = Field(default="hotel.booking_events")
    topic_demand: str = Field(default="hotel.demand_events")
    topic_predictions: str = Field(default="hotel.price_predictions")

    num_partitions: int = Field(default=3, ge=1)
    replication_factor: int = Field(default=1, ge=1)

    auto_offset_reset: str = Field(default="earliest")
    # Offsets are committed manually AFTER the database write completes, which is
    # what makes the consumer at-least-once instead of at-most-once.
    enable_auto_commit: bool = Field(default=False)
    max_poll_records: int = Field(default=500, ge=1)
    session_timeout_ms: int = Field(default=45_000, ge=1_000)
    # Must stay strictly above session_timeout_ms -- see _validate_timeouts.
    request_timeout_ms: int = Field(default=50_000, ge=1_000)
    retry_backoff_ms: int = Field(default=1_000, ge=0)
    producer_acks: str = Field(default="all")
    producer_retries: int = Field(default=5, ge=0)
    producer_linger_ms: int = Field(default=50, ge=0)

    @field_validator("auto_offset_reset")
    @classmethod
    def _validate_offset_reset(cls, v: str) -> str:
        allowed = {"earliest", "latest", "none"}
        if v not in allowed:
            raise ValueError(f"auto_offset_reset must be one of {sorted(allowed)}")
        return v

    @field_validator("producer_acks")
    @classmethod
    def _validate_acks(cls, v: str) -> str:
        allowed = {"0", "1", "all"}
        if str(v) not in allowed:
            raise ValueError(f"producer_acks must be one of {sorted(allowed)}")
        return str(v)

    @model_validator(mode="after")
    def _validate_timeouts(self) -> "KafkaSettings":
        """A consumer's request timeout must exceed its session timeout.

        kafka-python refuses to construct a consumer otherwise, and the error it
        raises appears at *subscribe* time -- deep inside a worker, minutes into
        a run. Checking it here turns a puzzling runtime crash into a
        configuration error at start-up with the offending numbers named.
        """
        if self.request_timeout_ms <= self.session_timeout_ms:
            raise ValueError(
                f"KAFKA_REQUEST_TIMEOUT_MS ({self.request_timeout_ms}) must be "
                f"greater than KAFKA_SESSION_TIMEOUT_MS ({self.session_timeout_ms})"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def bootstrap_servers_list(self) -> List[str]:
        """Broker list split for clients that want a sequence, not a string."""
        return [s.strip() for s in self.bootstrap_servers.split(",") if s.strip()]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def topics(self) -> Dict[str, str]:
        """Logical name -> physical topic name, for topic creation and lookup."""
        return {
            "competitor_prices": self.topic_competitor,
            "booking_events": self.topic_bookings,
            "demand_events": self.topic_demand,
            "price_predictions": self.topic_predictions,
        }


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


class APISettings(BaseSettings):
    """FastAPI server and HTTP-edge configuration. Reads ``API_*``."""

    model_config = _config("API_")

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1)
    prefix: str = Field(default="/api/v1")
    title: str = Field(default="AI-Driven Dynamic Hotel Pricing API")
    docs_enabled: bool = Field(default=True)
    request_timeout_seconds: int = Field(default=30, ge=1)

    # Stored as a delimited string, not List[str], on purpose:
    # pydantic-settings tries json.loads() on complex types read from env, so
    # `API_CORS_ORIGINS=http://a,http://b` would raise a JSON decode error.
    cors_origins: str = Field(default="http://localhost:8501")
    cors_allow_credentials: bool = Field(default=False)

    @field_validator("prefix")
    @classmethod
    def _validate_prefix(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError("API prefix must start with '/'")
        return v.rstrip("/")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


# --------------------------------------------------------------------------- #
# Models / MLOps
# --------------------------------------------------------------------------- #


class ModelSettings(BaseSettings):
    """Artifact locations and model-selection knobs.

    No prefix: the spec fixes ``MODEL_DIR`` as the variable name, and a
    ``MODEL_`` prefix on a ``model_dir`` field would produce ``MODEL_MODEL_DIR``.
    """

    model_config = _config()

    model_dir: Path = Field(default=PROJECT_ROOT / "models" / "artifacts")
    model_registry_file: str = Field(default="registry.json")
    model_active_version: Optional[str] = Field(default=None)

    # Weight given to the Prophet forecast when blending it with the Gradient
    # Boosting prediction:  blended = w * prophet + (1 - w) * gbr
    model_prophet_blend_weight: float = Field(default=0.5, ge=0.0, le=1.0)

    prophet_forecast_horizons: str = Field(default="7,14,30")

    @field_validator("model_dir", mode="before")
    @classmethod
    def _expand(cls, v: Any) -> Any:
        """Resolve relative artifact paths against the project root."""
        if v is None or v == "":
            return PROJECT_ROOT / "models" / "artifacts"
        p = Path(str(v)).expanduser()
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def forecast_horizons(self) -> List[int]:
        return [
            int(h.strip())
            for h in self.prophet_forecast_horizons.split(",")
            if h.strip()
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def registry_path(self) -> Path:
        return self.model_dir / self.model_registry_file


# --------------------------------------------------------------------------- #
# Pricing + guardrails
# --------------------------------------------------------------------------- #


class PricingSettings(BaseSettings):
    """The pricing engine's business parameters and hard guardrails.

    These are the numbers a revenue manager tunes. They are deliberately flat,
    unprefixed env vars (``MIN_PRICE``, ``MAX_PRICE``, ...) because operators
    edit them by hand.
    """

    model_config = _config()

    currency: str = Field(default="INR", min_length=3, max_length=3)
    default_base_price: float = Field(default=5000.0, gt=0)

    # --- absolute guardrails -------------------------------------------------
    min_price: float = Field(default=2500.0, gt=0)
    max_price: float = Field(default=25000.0, gt=0)

    # --- relative guardrails -------------------------------------------------
    max_daily_change_percent: float = Field(default=0.15, gt=0, le=1.0)
    competitor_upper_bound_percent: float = Field(default=0.20, ge=0, le=2.0)
    competitor_lower_bound_percent: float = Field(default=0.20, ge=0, le=1.0)

    # Below this occupancy the guardrails forbid any price increase, no matter
    # what the demand model says.
    low_occupancy_threshold: float = Field(default=0.40, ge=0.0, le=1.0)

    # --- per-signal adjustment clamps ---------------------------------------
    # Each adjustment is clamped before the four are summed, so no single signal
    # can dominate the final multiplier.
    max_demand_adjustment: float = Field(default=0.25, ge=0, le=1.0)
    max_competitor_adjustment: float = Field(default=0.15, ge=0, le=1.0)
    max_occupancy_adjustment: float = Field(default=0.20, ge=0, le=1.0)
    max_event_adjustment: float = Field(default=0.15, ge=0, le=1.0)
    # Smaller than the rest on purpose: seasonality is already baked into the
    # room's base rate by whoever set it, so this only corrects the residual.
    max_season_adjustment: float = Field(default=0.10, ge=0, le=1.0)

    # Demand at which the demand adjustment is neutral. Above it the engine
    # pushes price up, below it down. 0.65 is roughly the dataset's mean
    # occupancy, so an average night gets an average price.
    baseline_demand: float = Field(default=0.65, gt=0.0, le=1.0)

    price_rounding: int = Field(default=0, ge=0, le=2)

    @field_validator("currency")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.upper()

    @model_validator(mode="after")
    def _check_bounds(self) -> "PricingSettings":
        if self.min_price >= self.max_price:
            raise ValueError(
                f"MIN_PRICE ({self.min_price}) must be strictly less than "
                f"MAX_PRICE ({self.max_price})"
            )
        if not self.min_price <= self.default_base_price <= self.max_price:
            raise ValueError(
                f"DEFAULT_BASE_PRICE ({self.default_base_price}) must lie within "
                f"[MIN_PRICE, MAX_PRICE] = [{self.min_price}, {self.max_price}]"
            )
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_total_adjustment(self) -> float:
        """Upper bound on the combined multiplier, useful for sanity tests."""
        return (
            self.max_demand_adjustment
            + self.max_competitor_adjustment
            + self.max_occupancy_adjustment
            + self.max_event_adjustment
            + self.max_season_adjustment
        )


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


class IngestionSettings(BaseSettings):
    """Competitor-data acquisition. Reads ``INGESTION_*``.

    Real scrapers are opt-in twice over: ``source`` must name one *and*
    ``enable_real_scrapers`` must be true. See ADR-004 in docs/architecture.md.
    """

    model_config = _config("INGESTION_")

    source: CompetitorSource = Field(default=CompetitorSource.SYNTHETIC)
    enable_real_scrapers: bool = Field(default=False)

    request_timeout_seconds: int = Field(default=15, ge=1)
    rate_limit_seconds: float = Field(default=2.0, ge=0)
    max_retries: int = Field(default=3, ge=0)
    user_agent: str = Field(default="dynamic-hotel-pricing/0.1 (research)")

    #: Where the bundled demo OTA is served. A service name under Compose, a
    #: localhost port when running natively.
    demo_ota_base_url: str = Field(default="http://localhost:8900")

    #: Which markup the demo OTA should serve. ``v2`` simulates a site redesign
    #: that breaks every selector, so the ``ScraperParseError`` path can be
    #: demonstrated on demand rather than only being described. The dashboard
    #: exposes this as a toggle.
    demo_ota_layout: str = Field(default="v1", pattern="^v[12]$")

    #: Days ahead the scraper collects rates for on each pass. Kept short
    #: because every horizon multiplies the request count by the number of
    #: hotels and room types, and the rate limiter makes that time.
    scrape_horizons: str = Field(default="1,3,7,14,30")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def horizons(self) -> List[int]:
        return [int(h.strip()) for h in self.scrape_horizons.split(",") if h.strip()]

    synthetic_seed: int = Field(default=42)
    # Eight is the size of the shipped hotel catalogue: six cities, two of them
    # with two properties each, which is what makes a competitor set local.
    synthetic_hotels: int = Field(default=8, ge=1)
    synthetic_history_days: int = Field(default=365, ge=30)
    synthetic_interval_seconds: float = Field(default=2.0, gt=0)

    @model_validator(mode="after")
    def _guard_real_scrapers(self) -> "IngestionSettings":
        if self.source.is_third_party and not self.enable_real_scrapers:
            raise ValueError(
                f"INGESTION_SOURCE={self.source.value} requires "
                "INGESTION_ENABLE_REAL_SCRAPERS=true. Scraping a third-party "
                "site may breach its terms of service, so it is never a default. "
                "For a working scraping pipeline use INGESTION_SOURCE=demo_ota, "
                "which scrapes the bundled demo site over real HTTP."
            )
        return self


# --------------------------------------------------------------------------- #
# Monitoring
# --------------------------------------------------------------------------- #


class MonitoringSettings(BaseSettings):
    """Drift and performance monitoring thresholds. Reads ``MONITORING_*``."""

    model_config = _config("MONITORING_")

    enabled: bool = Field(default=True)
    drift_psi_threshold: float = Field(default=0.2, gt=0)
    prediction_sigma_threshold: float = Field(default=3.0, gt=0)
    performance_degradation_percent: float = Field(default=0.20, gt=0, le=1.0)
    metrics_window_days: int = Field(default=30, ge=1)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


class PathSettings(BaseSettings):
    """Filesystem layout. Reads ``DATA_DIR``; the rest are derived."""

    model_config = _config()

    data_dir: Path = Field(default=PROJECT_ROOT / "data")

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand(cls, v: Any) -> Any:
        if v is None or v == "":
            return PROJECT_ROOT / "data"
        p = Path(str(v)).expanduser()
        return p if p.is_absolute() else (PROJECT_ROOT / p)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def synthetic_dir(self) -> Path:
        return self.data_dir / "synthetic"


# --------------------------------------------------------------------------- #
# Root settings object
# --------------------------------------------------------------------------- #


class SecuritySettings(BaseSettings):
    """API authentication and abuse limits. Reads ``SECURITY_*``.

    TWO KEYS, TWO SCOPES
    --------------------
    ``read_key`` reaches every ``GET`` plus ``POST /pricing/predict`` with
    ``persist=false`` -- reads and simulations. ``write_key`` additionally
    reaches the endpoints that change state: persisted pricing decisions,
    competitor submissions, ingestion runs and retraining.

    The split exists for a specific reason. The AI agent is issued the *read*
    key, which turns its read-only guarantee from an in-process allowlist into a
    network-level fact: a prompt-injected agent, or a bug in the allowlist, or
    an entirely different process holding the same key, still cannot write. That
    was the documented gap in ``docs/ai_agent_design.md``, and this closes it.

    Disabled by default, because a local demo that demands a header before it
    will answer is a demo nobody runs. Production is the opposite -- see
    :meth:`Settings._production_safety`, which refuses to boot without it.
    """

    model_config = _config("SECURITY_")

    enabled: bool = Field(default=False)

    read_key: SecretStr = Field(default=SecretStr("dev-read-key"))
    write_key: SecretStr = Field(default=SecretStr("dev-write-key"))

    #: Header carrying the key. ``X-API-Key`` rather than ``Authorization``
    #: because these are static service keys, not bearer tokens with a lifetime.
    header_name: str = Field(default="X-API-Key")

    #: Requests per minute per key. 0 disables the limiter.
    rate_limit_per_minute: int = Field(default=240, ge=0)

    #: Endpoints that stay open even when auth is on: container healthchecks and
    #: load balancers cannot be expected to carry a credential, and Prometheus
    #: scrapes are protected at the network layer in every deployment worth
    #: having.
    public_paths: str = Field(default="/health,/metrics,/docs,/redoc,/openapi.json")

    @computed_field  # type: ignore[prop-decorator]
    @property
    def public_path_list(self) -> List[str]:
        return [p.strip() for p in self.public_paths.split(",") if p.strip()]

    def scope_for(self, presented: str) -> Optional[str]:
        """``"write"``, ``"read"``, or ``None`` for an unrecognised key.

        Compared with :func:`secrets.compare_digest` so a wrong key takes the
        same time to reject regardless of how much of it was correct.
        """
        import secrets

        if not presented:
            return None
        if secrets.compare_digest(presented, self.write_key.get_secret_value()):
            return "write"
        if secrets.compare_digest(presented, self.read_key.get_secret_value()):
            return "read"
        return None


class Settings(BaseSettings):
    """Aggregate of every settings group. Obtain via :func:`get_settings`."""

    model_config = _config()

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    api: APISettings = Field(default_factory=APISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    pricing: PricingSettings = Field(default_factory=PricingSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    paths: PathSettings = Field(default_factory=PathSettings)

    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @model_validator(mode="after")
    def _production_safety(self) -> "Settings":
        """Refuse to boot in production with development defaults.

        Every check here is something that is fine locally and unacceptable in
        production, and each one fails at *startup* rather than on the first
        request that happens to touch it. A misconfiguration that only shows up
        under traffic is a misconfiguration that ships.
        """
        if self.app.environment is not Environment.PRODUCTION:
            return self

        weak = {"pricing", "postgres", "password", "changeme", ""}
        if self.database.password.get_secret_value() in weak:
            raise ValueError(
                "POSTGRES_PASSWORD is a default/development value but "
                "ENVIRONMENT=production. Set a real secret."
            )
        if self.app.debug:
            raise ValueError("DEBUG must be false when ENVIRONMENT=production")

        # An unauthenticated pricing API on the internet lets anyone read the
        # commercial audit trail and trigger retraining. Not a default worth
        # trusting an operator to remember.
        if not self.security.enabled:
            raise ValueError(
                "SECURITY_ENABLED must be true when ENVIRONMENT=production. "
                "Set SECURITY_READ_KEY and SECURITY_WRITE_KEY as well."
            )

        shipped = {"dev-read-key", "dev-write-key", "changeme", ""}
        for name, key in (
            ("SECURITY_READ_KEY", self.security.read_key),
            ("SECURITY_WRITE_KEY", self.security.write_key),
        ):
            if key.get_secret_value() in shipped:
                raise ValueError(
                    f"{name} is still the shipped development value but "
                    f"ENVIRONMENT=production. Generate one: "
                    f"python -c \"import secrets; print(secrets.token_urlsafe(32))\""
                )

        if self.security.read_key.get_secret_value() == self.security.write_key.get_secret_value():
            raise ValueError(
                "SECURITY_READ_KEY and SECURITY_WRITE_KEY must differ -- "
                "identical keys collapse the two scopes and hand every reader "
                "write access."
            )

        if "*" in self.api.cors_origins_list:
            raise ValueError("API_CORS_ORIGINS must not be '*' when ENVIRONMENT=production")

        return self

    def ensure_directories(self) -> None:
        """Create the directories the system writes to. Safe to call repeatedly."""
        for path in (
            self.paths.data_dir,
            self.paths.raw_dir,
            self.paths.processed_dir,
            self.paths.synthetic_dir,
            self.model.model_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def summary(self) -> Dict[str, Any]:
        """Redacted snapshot suitable for logging and the health endpoint.

        Deliberately hand-built rather than ``model_dump()`` so a future field
        cannot leak a secret into logs just by being added.
        """
        return {
            "app": self.app.app_name,
            "version": self.app.app_version,
            "environment": self.app.environment.value,
            "debug": self.app.debug,
            "log_level": self.app.log_level.value,
            "database": self.database.safe_url,
            "kafka_enabled": self.kafka.enabled,
            "kafka_bootstrap": self.kafka.bootstrap_servers,
            "kafka_topics": list(self.kafka.topics.values()),
            "api": f"{self.api.host}:{self.api.port}{self.api.prefix}",
            "model_dir": str(self.model.model_dir),
            "competitor_source": self.ingestion.source.value,
            "real_scrapers_enabled": self.ingestion.enable_real_scrapers,
            "guardrails": {
                "min_price": self.pricing.min_price,
                "max_price": self.pricing.max_price,
                "max_daily_change_percent": self.pricing.max_daily_change_percent,
                "currency": self.pricing.currency,
            },
        }


def _current_env_file() -> Optional[str]:
    """Resolve the dotenv path *at call time*.

    The ``env_file`` baked into each ``model_config`` is fixed when the class is
    created, which makes it impossible to change afterwards. Resolving it here
    instead means ``ENV_FILE`` can be set by a container at runtime, and tests
    can point at a fixture (or disable dotenv entirely) without reimporting.

    Returns ``None`` when the file does not exist, which tells pydantic-settings
    to read the environment only.
    """
    candidate = os.getenv("ENV_FILE", str(PROJECT_ROOT / ".env"))
    return candidate if Path(candidate).is_file() else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so the ``.env`` file is parsed once and every module observes an
    identical configuration. Tests that need a different environment should call
    :func:`reset_settings` after patching ``os.environ``.

    Sub-models are constructed explicitly rather than through their default
    factories so that the resolved dotenv path is threaded into every one of
    them consistently.
    """
    env_file = _current_env_file()
    kw: Dict[str, Any] = {"_env_file": env_file}
    return Settings(
        app=AppSettings(**kw),
        database=DatabaseSettings(**kw),
        kafka=KafkaSettings(**kw),
        api=APISettings(**kw),
        model=ModelSettings(**kw),
        pricing=PricingSettings(**kw),
        ingestion=IngestionSettings(**kw),
        monitoring=MonitoringSettings(**kw),
        paths=PathSettings(**kw),
    )


def reset_settings() -> None:
    """Clear the settings cache. For tests only."""
    get_settings.cache_clear()


__all__ = [
    "PROJECT_ROOT",
    "ENV_FILE",
    "Environment",
    "LogLevel",
    "LogFormat",
    "CompetitorSource",
    "AppSettings",
    "DatabaseSettings",
    "KafkaSettings",
    "APISettings",
    "ModelSettings",
    "PricingSettings",
    "IngestionSettings",
    "MonitoringSettings",
    "PathSettings",
    "Settings",
    "get_settings",
    "reset_settings",
]
