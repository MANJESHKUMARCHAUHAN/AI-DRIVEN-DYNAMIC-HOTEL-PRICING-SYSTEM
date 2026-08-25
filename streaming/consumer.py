"""Kafka consumer: topics in, PostgreSQL rows out.

The delivery guarantee is **at-least-once**, and every design choice below
follows from it:

* ``enable.auto.commit`` is off. Offsets are committed only *after* the database
  transaction commits. Auto-commit would acknowledge messages the database never
  saw, turning a crash into silent data loss.
* Because offsets lag the read position, a crash or a rebalance replays
  messages. Every handler is therefore idempotent (see
  :mod:`streaming.handlers`), which is what makes replay harmless.
* A database failure rewinds the consumer to the start of the batch and retries
  with backoff. Nothing is skipped.
* A **poison message** -- bytes that are not a decodable event -- is counted,
  logged and skipped, and its offset *is* committed. The alternative is a single
  malformed record blocking its partition permanently, which is a far worse
  failure than losing one message that was never valid to begin with.

The processing core is deliberately separated from the polling loop:
:meth:`EventConsumer.process_records` takes raw bytes and needs no broker, so
the interesting behaviour -- idempotency, poison handling, validation, rollback
-- is unit-testable against SQLite.
"""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from config import Settings, get_settings
from database.connection import get_sessionmaker
from ingestion.validator import EventRejected, EventValidator
from monitoring.logging_config import get_logger, set_correlation_id
from streaming.events import EventDecodeError, EventEnvelope
from streaming.handlers import Handler, HandlerOutcome, build_handlers
from streaming.topics import TopicName, consumer_config, resolve_topic

logger = get_logger(__name__)


@dataclass
class ConsumerStats:
    """Counters for logging, tests and the monitoring page."""

    polled: int = 0
    written: int = 0
    duplicates: int = 0
    ignored: int = 0
    rejected: int = 0
    poison: int = 0
    batches: int = 0
    db_errors: int = 0
    commits: int = 0
    by_reject_reason: Dict[str, int] = field(default_factory=dict)

    def record_reject(self, reason: str) -> None:
        self.rejected += 1
        self.by_reject_reason[reason] = self.by_reject_reason.get(reason, 0) + 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "polled": self.polled,
            "written": self.written,
            "duplicates": self.duplicates,
            "ignored": self.ignored,
            "rejected": self.rejected,
            "poison": self.poison,
            "batches": self.batches,
            "db_errors": self.db_errors,
            "commits": self.commits,
            "by_reject_reason": dict(self.by_reject_reason),
        }


class EventConsumer:
    """Consumes events from Kafka and persists them.

    Example::

        consumer = EventConsumer([TopicName.COMPETITOR_PRICES])
        consumer.run(max_messages=1000)
    """

    def __init__(
        self,
        topics: Sequence[TopicName],
        *,
        settings: Optional[Settings] = None,
        handlers: Optional[Dict[Any, Handler]] = None,
        validator: Optional[EventValidator] = None,
        session_factory: Optional[sessionmaker] = None,
        client: Optional[Any] = None,
        group_suffix: str = "",
        max_db_retries: int = 5,
        db_retry_backoff_seconds: float = 1.0,
    ) -> None:
        """
        Args:
            topics: Logical topics to subscribe to.
            handlers: Event type -> handler. Defaults to
                :data:`streaming.handlers.DEFAULT_HANDLERS`.
            validator: Semantic validator. Defaults to a fresh
                :class:`~ingestion.validator.EventValidator`.
            session_factory: Session source. Tests pass a SQLite-bound factory.
            client: Pre-built Kafka consumer, for tests.
            group_suffix: Appended to the configured consumer group, so a
                second consumer can read the same topics independently.
            max_db_retries: Attempts before a batch is abandoned.
            db_retry_backoff_seconds: Initial delay between attempts; doubles.
        """
        self.settings = settings or get_settings()
        self.topics = list(topics)
        self.handlers = handlers or build_handlers()
        self.validator = validator or EventValidator()
        self._session_factory = session_factory
        self._client = client
        self._group_suffix = group_suffix
        self.max_db_retries = max_db_retries
        self.db_retry_backoff_seconds = db_retry_backoff_seconds

        self.stats = ConsumerStats()
        self._stop = threading.Event()

    # -- session ----------------------------------------------------------- #

    @property
    def session_factory(self) -> sessionmaker:
        """Lazily resolved so constructing a consumer opens no connections."""
        if self._session_factory is None:
            self._session_factory = get_sessionmaker()
        return self._session_factory

    # -- processing core (no Kafka required) -------------------------------- #

    def process_message(self, raw: bytes, session: Session) -> Optional[HandlerOutcome]:
        """Decode, validate and persist one message inside an open transaction.

        Returns:
            The handler's outcome, or ``None`` if the message was dropped as
            poison or rejected by validation. Neither is an error the caller
            should retry -- replaying them would produce the same result.
        """
        try:
            envelope = EventEnvelope.from_bytes(raw)
            payload = envelope.decode_payload()
        except EventDecodeError as exc:
            self.stats.poison += 1
            logger.warning(
                "Poison message skipped: %s | first 200 bytes: %r",
                exc,
                raw[:200],
            )
            return None

        # One correlation id per event, so a row in the database can be traced
        # back to the log line that wrote it.
        set_correlation_id(envelope.event_id.replace("-", "")[:16])

        try:
            self.validator.validate(payload, session)
        except EventRejected as exc:
            self.stats.record_reject(exc.reason)
            logger.warning(
                "Event %s rejected (%s): %s", envelope.event_id, exc.reason, exc.detail
            )
            return None

        handler = self.handlers.get(envelope.event_type)
        if handler is None:
            self.stats.record_reject("no_handler")
            logger.warning("No handler for event type %s", envelope.event_type.value)
            return None

        result = handler(envelope, payload, session)

        if result.outcome is HandlerOutcome.WRITTEN:
            self.stats.written += 1
        elif result.outcome is HandlerOutcome.DUPLICATE:
            self.stats.duplicates += 1
            logger.debug("Duplicate event %s ignored", envelope.event_id)
        else:
            self.stats.ignored += 1

        return result.outcome

    def process_records(self, records: Iterable[bytes]) -> ConsumerStats:
        """Process a batch in one transaction. Commits or rolls back as a unit.

        Raises:
            SQLAlchemyError: On a database failure, after rolling back. The
                caller must not commit offsets for this batch.
        """
        session = self.session_factory()
        try:
            count = 0
            for raw in records:
                self.process_message(raw, session)
                count += 1
            session.commit()
            self.stats.batches += 1
            self.stats.polled += count
            return self.stats
        except SQLAlchemyError:
            session.rollback()
            self.stats.db_errors += 1
            raise
        finally:
            session.close()

    # -- polling loop ------------------------------------------------------- #

    def _connect(self) -> Any:
        """Create and subscribe the Kafka consumer."""
        from kafka import KafkaConsumer

        config = consumer_config(self.settings, group_suffix=self._group_suffix)
        topic_names = [resolve_topic(topic, self.settings) for topic in self.topics]

        client = KafkaConsumer(*topic_names, **config)
        logger.info(
            "Consumer subscribed to %s as group %r",
            ", ".join(topic_names),
            config["group_id"],
        )
        return client

    def stop(self) -> None:
        """Ask the polling loop to finish the current batch and exit."""
        self._stop.set()

    def _install_signal_handlers(self) -> None:
        """Turn SIGINT/SIGTERM into a graceful stop.

        Only possible on the main thread; in a worker thread the caller is
        expected to drive :meth:`stop` itself.
        """
        if threading.current_thread() is not threading.main_thread():
            return

        def _handle(signum, _frame):  # type: ignore[no-untyped-def]
            logger.info("Signal %s received; shutting down after this batch", signum)
            self.stop()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle)
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                pass

    def run(
        self,
        *,
        max_messages: Optional[int] = None,
        poll_timeout_ms: int = 1_000,
        idle_timeout_seconds: Optional[float] = None,
    ) -> ConsumerStats:
        """Poll until stopped, or until ``max_messages`` have been processed.

        Args:
            max_messages: Stop after this many messages. ``None`` runs forever.
            poll_timeout_ms: How long each poll waits for records.
            idle_timeout_seconds: Stop after this long with no records. Used by
                the end-to-end script so it terminates instead of hanging.

        Returns:
            The final statistics.
        """
        self._install_signal_handlers()
        client = self._client or self._connect()
        self._client = client

        last_record_at = time.monotonic()

        try:
            while not self._stop.is_set():
                batches = client.poll(
                    timeout_ms=poll_timeout_ms,
                    max_records=self.settings.kafka.max_poll_records,
                )

                if not batches:
                    if (
                        idle_timeout_seconds is not None
                        and time.monotonic() - last_record_at > idle_timeout_seconds
                    ):
                        logger.info(
                            "Idle for %.0fs with no records; stopping",
                            idle_timeout_seconds,
                        )
                        break
                    continue

                last_record_at = time.monotonic()
                messages = [
                    record for records in batches.values() for record in records
                ]

                if self._process_with_retry(client, batches, messages):
                    client.commit()
                    self.stats.commits += 1

                if max_messages is not None and self.stats.polled >= max_messages:
                    logger.info("Reached max_messages=%d; stopping", max_messages)
                    break
        finally:
            self._close(client)

        logger.info("Consumer finished: %s", self.stats.as_dict())
        return self.stats

    def _process_with_retry(
        self, client: Any, batches: Dict[Any, List[Any]], messages: List[Any]
    ) -> bool:
        """Process a poll batch, retrying database failures with backoff.

        On failure the consumer is rewound to the first offset of each partition
        in the batch, so the retry genuinely reprocesses the same records rather
        than skipping them.

        Returns:
            ``True`` when the batch was committed to the database and its
            offsets may now be committed to Kafka.
        """
        delay = self.db_retry_backoff_seconds
        starts = {
            partition: records[0].offset
            for partition, records in batches.items()
            if records
        }

        for attempt in range(1, self.max_db_retries + 1):
            try:
                self.process_records(message.value for message in messages)
                return True
            except SQLAlchemyError as exc:
                if attempt == self.max_db_retries:
                    logger.error(
                        "Batch of %d message(s) abandoned after %d attempts: %s. "
                        "Offsets are NOT committed; the batch will be redelivered.",
                        len(messages),
                        attempt,
                        type(exc).__name__,
                    )
                    return False
                logger.warning(
                    "Database error on attempt %d/%d (%s); rewinding and retrying "
                    "in %.1fs",
                    attempt,
                    self.max_db_retries,
                    type(exc).__name__,
                    delay,
                )
                for partition, offset in starts.items():
                    client.seek(partition, offset)
                time.sleep(delay)
                delay *= 2
        return False

    def _close(self, client: Any) -> None:
        try:
            client.close()
        except Exception as exc:  # pragma: no cover - best-effort teardown
            logger.warning("Error closing consumer: %s", type(exc).__name__)


__all__ = ["ConsumerStats", "EventConsumer"]
