"""Kafka producer: typed payloads in, durable records out.

The design constraint that shapes this module is requirement 9's "graceful Kafka
failure handling". A pricing API whose ``/pricing/predict`` returns 500 because
an *analytics* topic is unreachable has traded a working product for a working
event log. So:

* **Publishing is best-effort by default.** :meth:`EventProducer.send` returns a
  bool and logs; it does not raise. Callers that genuinely cannot proceed
  without the event pass ``strict=True``.
* **A missing broker is a degraded mode, not a crash.** With
  ``KAFKA_ENABLED=false`` -- or with the broker simply absent -- the producer
  counts drops and carries on, and ``/health`` reports it.
* **Connection is lazy and retried.** Compose starts everything at once;
  the first send may land before the broker has finished electing itself.

``kafka-python`` is imported inside functions rather than at module scope so that
:mod:`streaming` stays importable -- and unit-testable -- on a machine with no
Kafka client installed at all.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from config import Settings, get_settings
from monitoring.logging_config import get_logger
from streaming.events import EventEnvelope, EventPayload
from streaming.topics import TopicName, producer_config, resolve_topic

logger = get_logger(__name__)


class KafkaUnavailable(RuntimeError):
    """Raised only in strict mode, when a send could not be completed."""


@dataclass
class ProducerStats:
    """Counters for monitoring and for the ``/health`` endpoint."""

    sent: int = 0
    failed: int = 0
    dropped_disabled: int = 0
    connect_attempts: int = 0
    last_error: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sent": self.sent,
            "failed": self.failed,
            "dropped_disabled": self.dropped_disabled,
            "connect_attempts": self.connect_attempts,
            "last_error": self.last_error,
        }


class EventProducer:
    """Publishes :class:`~streaming.events.EventPayload` objects to Kafka.

    Example::

        with EventProducer() as producer:
            producer.send(payload, TopicName.COMPETITOR_PRICES)

    The context manager flushes on exit, which matters: ``linger_ms`` batches
    records, so a process that exits without flushing silently loses whatever is
    still buffered.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        *,
        client: Optional[Any] = None,
        connect_retries: int = 3,
        connect_backoff_seconds: float = 2.0,
    ) -> None:
        """
        Args:
            settings: Configuration source.
            client: A pre-built producer. Tests inject a fake; production leaves
                it ``None`` so the client is created lazily on first send.
            connect_retries: Attempts before giving up on a connection.
            connect_backoff_seconds: Initial delay between attempts; doubles.
        """
        self.settings = settings or get_settings()
        self.stats = ProducerStats()
        self._client = client
        self._connect_retries = connect_retries
        self._connect_backoff = connect_backoff_seconds
        # A producer is shared across FastAPI request handlers, so client
        # creation must happen exactly once even under concurrent first calls.
        self._lock = threading.Lock()
        self._connect_failed = False

    # -- lifecycle --------------------------------------------------------- #

    @property
    def enabled(self) -> bool:
        """Whether publishing is switched on at all."""
        return self.settings.kafka.enabled

    @property
    def connected(self) -> bool:
        """Whether a client exists. Does not prove the broker is still up."""
        return self._client is not None

    def _connect(self) -> Optional[Any]:
        """Create the underlying client, retrying with backoff.

        Returns ``None`` -- rather than raising -- when the broker cannot be
        reached, so the caller degrades instead of failing.
        """
        from kafka import KafkaProducer  # noqa: PLC0415 - deliberate lazy import
        from kafka.errors import KafkaError

        config = producer_config(self.settings)
        delay = self._connect_backoff

        for attempt in range(1, self._connect_retries + 1):
            self.stats.connect_attempts += 1
            try:
                client = KafkaProducer(**config)
                logger.info(
                    "Kafka producer connected to %s",
                    self.settings.kafka.bootstrap_servers,
                )
                return client
            except KafkaError as exc:
                self.stats.last_error = f"{type(exc).__name__}: {exc}"
                if attempt == self._connect_retries:
                    logger.warning(
                        "Kafka unreachable at %s after %d attempt(s); event "
                        "publishing is degraded",
                        self.settings.kafka.bootstrap_servers,
                        attempt,
                    )
                    return None
                logger.warning(
                    "Kafka connection attempt %d/%d failed (%s); retrying in %.1fs",
                    attempt,
                    self._connect_retries,
                    type(exc).__name__,
                    delay,
                )
                time.sleep(delay)
                delay *= 2
        return None

    def _client_or_none(self) -> Optional[Any]:
        """Return the client, connecting once if needed."""
        if self._client is not None:
            return self._client
        if self._connect_failed:
            # Do not pay the retry cost on every single send once we know the
            # broker is down. reset() clears this.
            return None
        with self._lock:
            if self._client is None and not self._connect_failed:
                self._client = self._connect()
                self._connect_failed = self._client is None
        return self._client

    def reset(self) -> None:
        """Forget a failed connection so the next send retries the broker."""
        with self._lock:
            self._client = None
            self._connect_failed = False

    # -- publishing -------------------------------------------------------- #

    def send(
        self,
        payload: EventPayload,
        topic: TopicName,
        *,
        source: Optional[str] = None,
        strict: bool = False,
        envelope: Optional[EventEnvelope] = None,
    ) -> bool:
        """Publish one payload.

        Args:
            payload: The typed event body.
            topic: Logical topic; the physical name comes from configuration.
            source: Overrides the envelope's ``source`` field.
            strict: Raise :class:`KafkaUnavailable` instead of returning False.
            envelope: Pre-built envelope, when the caller needs to control the
                event id (for instance to make a retry idempotent).

        Returns:
            ``True`` if the record was handed to the client.
        """
        if not self.enabled:
            self.stats.dropped_disabled += 1
            logger.debug("KAFKA_ENABLED=false; dropping %s event", payload.event_type.value)
            if strict:
                raise KafkaUnavailable("Kafka is disabled by configuration")
            return False

        envelope = envelope or EventEnvelope.wrap(
            payload, source=source or self.settings.ingestion.source.value
        )
        return self.send_envelope(envelope, topic, strict=strict)

    def send_envelope(
        self, envelope: EventEnvelope, topic: TopicName, *, strict: bool = False
    ) -> bool:
        """Publish an already-built envelope."""
        client = self._client_or_none()
        if client is None:
            self.stats.failed += 1
            if strict:
                raise KafkaUnavailable(
                    f"cannot publish {envelope.event_type.value}: "
                    f"broker {self.settings.kafka.bootstrap_servers} unreachable"
                )
            return False

        from kafka.errors import KafkaError

        topic_name = resolve_topic(topic, self.settings)
        key = envelope.partition_key

        try:
            client.send(
                topic_name,
                value=envelope.to_bytes(),
                key=key.encode("utf-8") if key else None,
            )
            self.stats.sent += 1
            logger.debug(
                "Published %s to %s (key=%s, event_id=%s)",
                envelope.event_type.value,
                topic_name,
                key,
                envelope.event_id,
            )
            return True
        except KafkaError as exc:
            self.stats.failed += 1
            self.stats.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "Failed to publish %s to %s: %s",
                envelope.event_type.value,
                topic_name,
                type(exc).__name__,
            )
            if strict:
                raise KafkaUnavailable(str(exc)) from exc
            return False

    def send_many(
        self,
        payloads: list,
        topic: TopicName,
        *,
        source: Optional[str] = None,
    ) -> int:
        """Publish a batch. Returns how many were accepted."""
        return sum(1 for p in payloads if self.send(p, topic, source=source))

    # -- teardown ---------------------------------------------------------- #

    def flush(self, timeout: Optional[float] = None) -> None:
        """Block until buffered records are delivered.

        ``linger_ms`` batches records for throughput, so an unflushed exit loses
        whatever is still in the accumulator.
        """
        if self._client is None:
            return
        from kafka.errors import KafkaError

        try:
            self._client.flush(timeout=timeout)
        except KafkaError as exc:
            logger.warning("Flush failed: %s", type(exc).__name__)

    def close(self, timeout: float = 10.0) -> None:
        """Flush and close the client."""
        if self._client is None:
            return
        try:
            self._client.close(timeout=timeout)
            logger.info(
                "Kafka producer closed (sent=%d failed=%d)",
                self.stats.sent,
                self.stats.failed,
            )
        except Exception as exc:  # pragma: no cover - best-effort teardown
            logger.warning("Error closing producer: %s", type(exc).__name__)
        finally:
            self._client = None

    def __enter__(self) -> "EventProducer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


#: Process-wide producer, created on first use. The API and the dashboard share
#: one client; creating a producer per request would open a new TCP connection
#: and a new metadata fetch every time.
_shared: Optional[EventProducer] = None
_shared_lock = threading.Lock()


def get_producer(settings: Optional[Settings] = None) -> EventProducer:
    """Return the process-wide producer."""
    global _shared
    if _shared is None:
        with _shared_lock:
            if _shared is None:
                _shared = EventProducer(settings)
    return _shared


def close_producer() -> None:
    """Close and forget the shared producer. Called on API shutdown."""
    global _shared
    with _shared_lock:
        if _shared is not None:
            _shared.close()
            _shared = None


__all__ = [
    "EventProducer",
    "KafkaUnavailable",
    "ProducerStats",
    "close_producer",
    "get_producer",
]
