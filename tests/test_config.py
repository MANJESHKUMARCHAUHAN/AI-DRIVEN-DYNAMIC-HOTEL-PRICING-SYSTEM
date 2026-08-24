"""Phase 1 tests: configuration loading, validation and secret hygiene.

These assert the contracts other phases will rely on -- that env vars are
actually read, that invalid business thresholds are rejected at boot rather than
at runtime, and that credentials never leak into logs or health payloads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from config import (
    APISettings,
    CompetitorSource,
    DatabaseSettings,
    Environment,
    IngestionSettings,
    KafkaSettings,
    ModelSettings,
    PricingSettings,
    Settings,
    get_settings,
    reset_settings,
)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


class TestSettingsLoading:
    def test_environment_variables_are_read(self, settings: Settings) -> None:
        assert settings.database.host == "test-db"
        assert settings.database.port == 5433
        assert settings.database.db == "test_pricing"
        assert settings.kafka.bootstrap_servers == "test-kafka:9092"
        assert settings.api.port == 8000

    def test_settings_are_cached(self, isolated_env: None) -> None:
        assert get_settings() is get_settings()

    def test_reset_clears_the_cache(self, isolated_env: None) -> None:
        first = get_settings()
        reset_settings()
        assert get_settings() is not first

    def test_every_group_is_present(self, settings: Settings) -> None:
        for group in (
            "app", "database", "kafka", "api",
            "model", "pricing", "ingestion", "monitoring", "paths",
        ):
            assert getattr(settings, group) is not None, group


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #


class TestDatabaseSettings:
    def test_dsn_is_well_formed(self, settings: Settings) -> None:
        url = settings.database.url
        assert url.startswith("postgresql+psycopg2://")
        assert "test-db:5433" in url
        assert url.endswith("/test_pricing")

    def test_password_is_url_encoded(self) -> None:
        db = DatabaseSettings(
            _env_file=None, host="h", user="u", password="p@ss:w/rd", db="d"
        )
        # Raw special characters would corrupt the DSN; they must be escaped.
        assert "p%40ss%3Aw%2Frd" in db.url
        assert "p@ss:w/rd" not in db.url

    def test_safe_url_masks_the_password(self, settings: Settings) -> None:
        assert "test_secret_value" not in settings.database.safe_url
        assert "***" in settings.database.safe_url

    def test_repr_does_not_leak_the_password(self, settings: Settings) -> None:
        assert "test_secret_value" not in repr(settings.database)

    def test_port_is_range_checked(self) -> None:
        with pytest.raises(ValidationError):
            DatabaseSettings(_env_file=None, port=70000)


# --------------------------------------------------------------------------- #
# Kafka
# --------------------------------------------------------------------------- #


class TestKafkaSettings:
    def test_all_four_topics_are_configured(self, settings: Settings) -> None:
        assert set(settings.kafka.topics) == {
            "competitor_prices",
            "booking_events",
            "demand_events",
            "price_predictions",
        }
        assert settings.kafka.topic_competitor == "hotel.competitor_prices"

    def test_bootstrap_servers_split_into_a_list(self) -> None:
        kafka = KafkaSettings(_env_file=None, bootstrap_servers="a:9092, b:9092 ,c:9092")
        assert kafka.bootstrap_servers_list == ["a:9092", "b:9092", "c:9092"]

    def test_auto_commit_is_disabled(self, settings: Settings) -> None:
        # At-least-once delivery depends on manual commits after the DB write.
        assert settings.kafka.enable_auto_commit is False

    def test_invalid_offset_reset_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            KafkaSettings(_env_file=None, auto_offset_reset="sometimes")

    def test_invalid_acks_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            KafkaSettings(_env_file=None, producer_acks="most")


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


class TestAPISettings:
    def test_cors_origins_parse_from_a_delimited_string(self) -> None:
        # Stored as str, not List[str]: pydantic-settings would try json.loads()
        # on a list-typed field and choke on comma-separated input.
        api = APISettings(_env_file=None, cors_origins="http://a:1, http://b:2")
        assert api.cors_origins_list == ["http://a:1", "http://b:2"]

    def test_prefix_must_be_absolute(self) -> None:
        with pytest.raises(ValidationError):
            APISettings(_env_file=None, prefix="api/v1")

    def test_trailing_slash_is_stripped(self) -> None:
        assert APISettings(_env_file=None, prefix="/api/v1/").prefix == "/api/v1"


# --------------------------------------------------------------------------- #
# Pricing guardrails
# --------------------------------------------------------------------------- #


class TestPricingSettings:
    def test_defaults_match_the_specification(self, settings: Settings) -> None:
        assert settings.pricing.min_price == 2500.0
        assert settings.pricing.max_price == 25000.0
        assert settings.pricing.max_daily_change_percent == 0.15

    def test_min_price_above_max_price_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="strictly less than"):
            PricingSettings(_env_file=None, min_price=30000, max_price=25000)

    def test_base_price_outside_the_band_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must lie within"):
            PricingSettings(
                _env_file=None, min_price=2500, max_price=25000,
                default_base_price=100,
            )

    def test_daily_change_percent_must_be_a_fraction(self) -> None:
        with pytest.raises(ValidationError):
            PricingSettings(_env_file=None, max_daily_change_percent=15)

    def test_negative_price_is_rejected(self) -> None:
        with pytest.raises(ValidationError):
            PricingSettings(_env_file=None, min_price=-1)

    def test_total_adjustment_is_bounded(self, settings: Settings) -> None:
        # A sanity bound the pricing engine will rely on in Phase 7.
        assert 0 < settings.pricing.max_total_adjustment <= 1.0

    def test_currency_is_normalised(self) -> None:
        assert PricingSettings(_env_file=None, currency="inr").currency == "INR"


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #


class TestIngestionSettings:
    def test_synthetic_is_the_default_source(self, settings: Settings) -> None:
        assert settings.ingestion.source is CompetitorSource.SYNTHETIC
        assert settings.ingestion.enable_real_scrapers is False

    def test_real_scraper_requires_explicit_opt_in(self) -> None:
        # ADR-004: naming a real source is not enough on its own.
        with pytest.raises(ValidationError, match="ENABLE_REAL_SCRAPERS"):
            IngestionSettings(_env_file=None, source="booking")

    def test_real_scraper_works_when_both_flags_are_set(self) -> None:
        cfg = IngestionSettings(
            _env_file=None, source="booking", enable_real_scrapers=True
        )
        assert cfg.source is CompetitorSource.BOOKING


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #


class TestModelSettings:
    def test_model_dir_is_absolute(self, settings: Settings) -> None:
        assert settings.model.model_dir.is_absolute()

    def test_relative_path_resolves_against_the_project_root(self) -> None:
        cfg = ModelSettings(_env_file=None, model_dir="models/artifacts")
        assert cfg.model_dir.is_absolute()
        assert cfg.model_dir.parts[-2:] == ("models", "artifacts")

    def test_forecast_horizons_parse(self) -> None:
        cfg = ModelSettings(_env_file=None, prophet_forecast_horizons="7, 14,30")
        assert cfg.forecast_horizons == [7, 14, 30]

    def test_blend_weight_is_a_fraction(self) -> None:
        with pytest.raises(ValidationError):
            ModelSettings(_env_file=None, model_prophet_blend_weight=1.5)


# --------------------------------------------------------------------------- #
# Production safety + secret hygiene
# --------------------------------------------------------------------------- #


class TestProductionSafety:
    def test_default_password_is_rejected_in_production(
        self, monkeypatch: pytest.MonkeyPatch, isolated_env: None
    ) -> None:
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("POSTGRES_PASSWORD", "pricing")
        reset_settings()
        with pytest.raises(ValidationError, match="real secret"):
            get_settings()

    def test_debug_is_rejected_in_production(
        self, monkeypatch: pytest.MonkeyPatch, isolated_env: None
    ) -> None:
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("DEBUG", "true")
        reset_settings()
        with pytest.raises(ValidationError, match="DEBUG must be false"):
            get_settings()

    def test_strong_password_is_accepted_in_production(
        self, monkeypatch: pytest.MonkeyPatch, isolated_env: None
    ) -> None:
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("POSTGRES_PASSWORD", "a-real-generated-secret")
        reset_settings()
        assert get_settings().app.environment is Environment.PRODUCTION


class TestSummaryRedaction:
    def test_summary_contains_no_secret(self, settings: Settings) -> None:
        rendered = str(settings.summary())
        assert "test_secret_value" not in rendered

    def test_summary_reports_the_guardrails(self, settings: Settings) -> None:
        assert settings.summary()["guardrails"]["min_price"] == 2500.0

    def test_summary_keys_are_stable(self, settings: Settings) -> None:
        # The health endpoint and startup log depend on these.
        for key in ("app", "version", "environment", "database", "kafka_topics"):
            assert key in settings.summary()


# --------------------------------------------------------------------------- #
# Directories
# --------------------------------------------------------------------------- #


class TestDirectories:
    def test_ensure_directories_creates_everything(self, settings: Settings) -> None:
        settings.ensure_directories()
        for path in (
            settings.paths.data_dir,
            settings.paths.raw_dir,
            settings.paths.processed_dir,
            settings.paths.synthetic_dir,
            settings.model.model_dir,
        ):
            assert path.is_dir(), path

    def test_ensure_directories_is_idempotent(self, settings: Settings) -> None:
        settings.ensure_directories()
        settings.ensure_directories()
        assert settings.paths.data_dir.is_dir()

    def test_data_subdirectories_nest_correctly(self, settings: Settings) -> None:
        assert settings.paths.raw_dir.parent == settings.paths.data_dir
        assert Path(settings.paths.synthetic_dir).name == "synthetic"
