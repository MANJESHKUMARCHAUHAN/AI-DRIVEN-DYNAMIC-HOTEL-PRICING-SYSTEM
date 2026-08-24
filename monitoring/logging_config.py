"""Structured logging for the whole platform.

Two output modes, chosen by ``LOG_FORMAT``:

``console``
    Human-readable, aligned, colour-free. The default for local development.

``json``
    One JSON object per line. The default inside containers, where logs are
    scraped rather than read.

Two cross-cutting concerns are handled here rather than at every call site:

**Correlation ids.** :func:`set_correlation_id` stores an id in a
:class:`~contextvars.ContextVar`; every record emitted downstream carries it.
That is what lets a single request be traced from the HTTP edge, through the
pricing engine, to the guardrail that clamped it.

**Secret redaction.** :class:`SecretRedactionFilter` scrubs the configured
database password and common credential patterns out of every record before it
is formatted. Requirement 21 says "do not log secrets"; a filter enforces that
even when a future caller forgets.
"""

from __future__ import annotations

import json
import logging
import logging.config
import re
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from config import LogFormat, Settings, get_settings

# --------------------------------------------------------------------------- #
# Correlation id
# --------------------------------------------------------------------------- #

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")


def set_correlation_id(value: Optional[str] = None) -> str:
    """Bind a correlation id to the current context and return it."""
    cid = value or uuid.uuid4().hex[:16]
    _correlation_id.set(cid)
    return cid


def get_correlation_id() -> str:
    """Return the correlation id bound to the current context."""
    return _correlation_id.get()


class CorrelationIdFilter(logging.Filter):
    """Attach ``record.correlation_id`` to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #

#: Patterns that look like credentials regardless of who logged them.
_REDACTION_PATTERNS = (
    # key=value / key: value  for sensitive-looking keys
    re.compile(
        r"(?i)\b(password|passwd|pwd|secret|token|api[_-]?key|authorization)\b"
        r"\s*[=:]\s*(\"[^\"]*\"|'[^']*'|[^\s,;)}]+)"
    ),
    # credentials embedded in a URI:  scheme://user:password@host
    re.compile(r"(?i)://([^:/@\s]+):([^@/\s]+)@"),
)


class SecretRedactionFilter(logging.Filter):
    """Remove secret material from log records.

    Redacts two ways: literal known secrets (the configured DB password), and
    anything matching a credential-shaped pattern. Both the message and the
    formatting arguments are scrubbed, because ``logger.info("dsn=%s", url)``
    keeps the secret in ``record.args`` until format time.
    """

    MASK = "***REDACTED***"

    def __init__(self, literals: Optional[Iterable[str]] = None) -> None:
        super().__init__()
        # Ignore trivially short values -- redacting "a" would mangle every line.
        self._literals = [s for s in (literals or []) if s and len(s) >= 4]

    def _scrub(self, text: str) -> str:
        for literal in self._literals:
            if literal in text:
                text = text.replace(literal, self.MASK)
        text = _REDACTION_PATTERNS[0].sub(rf"\1={self.MASK}", text)
        text = _REDACTION_PATTERNS[1].sub(rf"://\1:{self.MASK}@", text)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._scrub(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._scrub(v) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self._scrub(a) if isinstance(a, str) else a for a in record.args
                )
        return True


# --------------------------------------------------------------------------- #
# Formatters
# --------------------------------------------------------------------------- #

#: Attributes present on every LogRecord; anything else was added by the caller
#: via ``extra=`` and therefore belongs in the structured payload.
_STANDARD_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName", "taskName",
        "correlation_id",
    }
)


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per record, with ``extra=`` fields merged in."""

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
            "module": record.module,
            "line": record.lineno,
        }

        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Aligned, readable single-line output for terminals."""

    _FMT = "%(asctime)s | %(levelname)-8s | %(cid)-16s | %(name)-28s | %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        record.cid = getattr(record, "correlation_id", "-")
        return super().format(record)


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

_configured = False


def configure_logging(
    settings: Optional[Settings] = None, *, force: bool = False
) -> None:
    """Install handlers, filters and formatters on the root logger.

    Idempotent: repeated calls are ignored unless ``force=True``. This matters
    because uvicorn, Streamlit and pytest each import application modules in
    their own way, and double configuration produces duplicated log lines.
    """
    global _configured
    if _configured and not force:
        return

    settings = settings or get_settings()

    formatter: logging.Formatter = (
        JsonFormatter()
        if settings.app.log_format is LogFormat.JSON
        else ConsoleFormatter()
    )

    redaction = SecretRedactionFilter(
        literals=[settings.database.password.get_secret_value()]
    )
    correlation = CorrelationIdFilter()

    handlers: list[logging.Handler] = []

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    stream.addFilter(correlation)
    stream.addFilter(redaction)
    handlers.append(stream)

    if settings.app.log_file is not None:
        settings.app.log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(settings.app.log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(correlation)
        file_handler.addFilter(redaction)
        handlers.append(file_handler)

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(settings.app.log_level.value)

    # Third-party loggers are chatty at INFO and drown application signal.
    for noisy, level in {
        "kafka": logging.WARNING,
        "kafka.conn": logging.ERROR,
        "urllib3": logging.WARNING,
        "asyncio": logging.WARNING,
        "sqlalchemy.engine": (
            logging.INFO if settings.database.echo else logging.WARNING
        ),
        "uvicorn.access": logging.WARNING,
        "watchdog": logging.WARNING,
    }.items():
        logging.getLogger(noisy).setLevel(level)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring logging on first use.

    Modules should call this at import time::

        logger = get_logger(__name__)
    """
    configure_logging()
    return logging.getLogger(name)


__all__ = [
    "ConsoleFormatter",
    "CorrelationIdFilter",
    "JsonFormatter",
    "SecretRedactionFilter",
    "configure_logging",
    "get_correlation_id",
    "get_logger",
    "set_correlation_id",
]
