"""Phase 1 tests: logging format, correlation ids and secret redaction.

The redaction tests matter most. Requirement 21 forbids logging secrets, and a
filter is the only way to enforce that against future call sites that forget.
"""

from __future__ import annotations

import json
import logging

import pytest

from monitoring.logging_config import (
    ConsoleFormatter,
    CorrelationIdFilter,
    JsonFormatter,
    SecretRedactionFilter,
    configure_logging,
    get_correlation_id,
    set_correlation_id,
)


def _record(msg: str, args: object = None) -> logging.LogRecord:
    """Build a LogRecord without going through a logger."""
    return logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=None,
    )


class TestCorrelationId:
    def test_set_and_get_round_trip(self) -> None:
        set_correlation_id("abc123")
        assert get_correlation_id() == "abc123"

    def test_generated_when_not_supplied(self) -> None:
        cid = set_correlation_id()
        assert cid and len(cid) == 16
        assert get_correlation_id() == cid

    def test_filter_attaches_the_id(self) -> None:
        set_correlation_id("trace-me")
        record = _record("hello")
        assert CorrelationIdFilter().filter(record) is True
        assert record.correlation_id == "trace-me"


class TestSecretRedaction:
    @pytest.mark.parametrize(
        "message",
        [
            "connecting with password=hunter2 now",
            "PASSWORD: hunter2",
            "api_key=hunter2",
            "token = hunter2",
            "Authorization: hunter2",
        ],
    )
    def test_credential_patterns_are_masked(self, message: str) -> None:
        record = _record(message)
        SecretRedactionFilter().filter(record)
        assert "hunter2" not in record.msg
        assert SecretRedactionFilter.MASK in record.msg

    def test_uri_credentials_are_masked(self) -> None:
        record = _record("dsn postgresql://user:s3cretpw@db:5432/app")
        SecretRedactionFilter().filter(record)
        assert "s3cretpw" not in record.msg
        assert "user" in record.msg  # username is not a secret

    def test_known_literal_is_masked_anywhere(self) -> None:
        record = _record("the value my-db-password appears bare")
        SecretRedactionFilter(literals=["my-db-password"]).filter(record)
        assert "my-db-password" not in record.msg

    def test_args_are_scrubbed_too(self) -> None:
        # logger.info("dsn=%s", url) keeps the secret in record.args until
        # format time, so the filter has to reach into args as well.
        record = _record("dsn=%s", ("postgresql://u:leakedpw@h:5432/d",))
        SecretRedactionFilter(literals=["leakedpw"]).filter(record)
        assert "leakedpw" not in str(record.args)

    def test_short_literals_are_ignored(self) -> None:
        # Redacting a 1-character literal would mangle every line.
        record = _record("a normal message")
        SecretRedactionFilter(literals=["a"]).filter(record)
        assert record.msg == "a normal message"

    def test_dict_args_are_scrubbed(self) -> None:
        # The mapping is wrapped in a tuple because that is what
        # ``logger.info("%(dsn)s", mapping)`` actually passes; LogRecord then
        # unwraps it onto ``record.args``. Handing the bare mapping straight to
        # LogRecord raises inside the standard library, so this shape is both
        # more realistic and the only one that works.
        record = _record("%(dsn)s", ({"dsn": "postgresql://u:leakedpw@h/d"},))
        assert isinstance(record.args, dict)  # the unwrap happened

        SecretRedactionFilter(literals=["leakedpw"]).filter(record)
        assert "leakedpw" not in str(record.args)


class TestJsonFormatter:
    def test_output_is_valid_json_with_required_keys(self) -> None:
        set_correlation_id("cid-1")
        record = _record("something happened")
        CorrelationIdFilter().filter(record)

        payload = json.loads(JsonFormatter().format(record))

        assert payload["message"] == "something happened"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "test.logger"
        assert payload["correlation_id"] == "cid-1"
        assert "timestamp" in payload

    def test_extra_fields_are_merged(self) -> None:
        record = _record("priced")
        record.hotel_id = "H001"  # type: ignore[attr-defined]
        record.final_price = 6200  # type: ignore[attr-defined]

        payload = json.loads(JsonFormatter().format(record))

        assert payload["hotel_id"] == "H001"
        assert payload["final_price"] == 6200

    def test_timestamp_is_utc(self) -> None:
        payload = json.loads(JsonFormatter().format(_record("x")))
        assert payload["timestamp"].endswith("+00:00")


class TestConsoleFormatter:
    def test_includes_level_logger_and_message(self) -> None:
        set_correlation_id("cid-2")
        record = _record("readable line")
        CorrelationIdFilter().filter(record)

        line = ConsoleFormatter().format(record)

        assert "INFO" in line
        assert "test.logger" in line
        assert "readable line" in line
        assert "cid-2" in line


class TestConfigureLogging:
    def test_configures_root_and_is_idempotent(self, settings) -> None:
        configure_logging(settings, force=True)
        root = logging.getLogger()
        assert root.handlers
        before = len(root.handlers)

        configure_logging(settings)  # no force -> must be a no-op
        assert len(root.handlers) == before

    def test_respects_the_configured_level(self, settings) -> None:
        configure_logging(settings, force=True)
        assert logging.getLogger().level == logging.WARNING

    def test_db_password_is_registered_for_redaction(self, settings, caplog) -> None:
        configure_logging(settings, force=True)
        handler = logging.getLogger().handlers[0]
        redactors = [
            f for f in handler.filters if isinstance(f, SecretRedactionFilter)
        ]
        assert redactors, "a SecretRedactionFilter must be installed"
        record = _record(f"dsn has {settings.database.password.get_secret_value()}")
        redactors[0].filter(record)
        assert "test_secret_value" not in record.msg
