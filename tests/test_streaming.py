"""Tests for the streaming layer: event contracts, producer, consumer, handlers.

No broker is involved. The producer takes an injected client and the consumer's
processing core takes raw bytes, so every behaviour that actually matters --
idempotency, poison handling, rollback, graceful degradation -- is tested
deterministically in milliseconds. The Kafka wiring itself is verified
separately by an end-to-end run against a real broker.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from database.models import (
    Booking,
    Competitor,
    CompetitorPrice,
    DemandFeature,
    RoomType,
    Season,
)
from ingestion.validator import EventValidator, ReferenceData
from streaming.consumer import EventConsumer
from streaming.events import (
    MAX_HORIZON_DAYS,
    SCHEMA_VERSION,
    BookingPayload,
    CompetitorPricePayload,
    DemandSignalPayload,
    EventDecodeError,
    EventEnvelope,
    EventType,
    PricePredictionPayload,
    deserialize,
    serialize,
)
from streaming.handlers import build_handlers
from streaming.producer import EventProducer, KafkaUnavailable
from streaming.topics import TopicName

STAY = date(2026, 9, 15)


# --------------------------------------------------------------------------- #
# Fixtures and fakes
# --------------------------------------------------------------------------- #


class FakeKafkaClient:
    """Records what would have been sent. Stands in for ``KafkaProducer``."""

    def __init__(self, fail_on: Optional[int] = None) -> None:
        self.records: List[Dict[str, Any]] = []
        self.flushed = False
        self.closed = False
        self._fail_on = fail_on

    def send(self, topic: str, value: bytes, key: Optional[bytes] = None) -> object:
        if self._fail_on is not None and len(self.records) == self._fail_on:
            from kafka.errors import KafkaTimeoutError

            raise KafkaTimeoutError("simulated broker timeout")
        self.records.append({"topic": topic, "key": key, "value": value})
        return object()

    def flush(self, timeout: Optional[float] = None) -> None:
        self.flushed = True

    def close(self, timeout: Optional[float] = None) -> None:
        self.closed = True


def competitor_payload(**overrides) -> CompetitorPricePayload:
    row = dict(
        hotel_id="H001",
        competitor=Competitor.BOOKING,
        room_type=RoomType.DELUXE,
        check_in_date=STAY,
        price=6_200.0,
    )
    row.update(overrides)
    return CompetitorPricePayload(**row)


def booking_payload(**overrides) -> BookingPayload:
    row = dict(
        hotel_id="H001",
        room_type=RoomType.DELUXE,
        booking_date=STAY - timedelta(days=10),
        check_in_date=STAY,
        check_out_date=STAY + timedelta(days=1),
        booking_count=6,
        cancellation_count=1,
        revenue=32_000.0,
        adr=6_400.0,
    )
    row.update(overrides)
    return BookingPayload(**row)


def demand_payload(**overrides) -> DemandSignalPayload:
    row = dict(
        hotel_id="H001",
        room_type=RoomType.DELUXE,
        stay_date=STAY,
        search_demand=0.72,
        weather_score=0.61,
        local_event_score=0.4,
    )
    row.update(overrides)
    return DemandSignalPayload(**row)


@pytest.fixture
def consumer(seeded_session, db_engine) -> EventConsumer:
    """A consumer wired to the in-memory database, with no Kafka client."""
    factory = sessionmaker(bind=db_engine, expire_on_commit=False, future=True)
    validator = EventValidator(reference=ReferenceData())
    validator.reference.refresh(seeded_session)
    return EventConsumer(
        [TopicName.COMPETITOR_PRICES],
        handlers=build_handlers(),
        validator=validator,
        session_factory=factory,
    )


# --------------------------------------------------------------------------- #
# Event contracts
# --------------------------------------------------------------------------- #


class TestEventEnvelope:
    def test_round_trip_preserves_the_payload(self) -> None:
        payload = competitor_payload()
        envelope, decoded = deserialize(serialize(payload))
        assert decoded == payload
        assert envelope.event_type is EventType.COMPETITOR_PRICE
        assert envelope.version == SCHEMA_VERSION

    def test_every_event_gets_a_unique_id(self) -> None:
        """The id is the idempotency key, so collisions would silently drop data."""
        ids = {EventEnvelope.wrap(competitor_payload()).event_id for _ in range(100)}
        assert len(ids) == 100

    def test_partition_key_is_the_hotel(self) -> None:
        """Keying by hotel keeps one property's events ordered relative to each
        other, which is the only ordering the pipeline needs."""
        assert EventEnvelope.wrap(competitor_payload()).partition_key == "H001"

    def test_naive_timestamps_are_treated_as_utc(self) -> None:
        envelope = EventEnvelope.wrap(
            competitor_payload(), timestamp=datetime(2026, 9, 1, 10, 0)
        )
        assert envelope.timestamp.tzinfo is timezone.utc

    def test_aware_timestamps_are_converted_to_utc(self) -> None:
        ist = timezone(timedelta(hours=5, minutes=30))
        envelope = EventEnvelope.wrap(
            competitor_payload(), timestamp=datetime(2026, 9, 1, 15, 30, tzinfo=ist)
        )
        assert envelope.timestamp == datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)

    def test_wire_format_has_the_documented_shape(self) -> None:
        document = json.loads(serialize(competitor_payload()).decode())
        assert set(document) == {
            "event_id", "event_type", "version", "timestamp", "source", "payload"
        }

    def test_future_schema_version_is_refused(self) -> None:
        """A consumer must reject what it cannot understand, not guess."""
        envelope = EventEnvelope.wrap(competitor_payload())
        envelope.version = SCHEMA_VERSION + 1
        with pytest.raises(EventDecodeError, match="schema version"):
            envelope.decode_payload()

    @pytest.mark.parametrize(
        ("label", "raw"),
        [
            ("not json", b"<html>bot check</html>"),
            ("json but not an object", b'"just a string"'),
            ("empty", b""),
            ("unknown event type", b'{"event_type":"nonsense","payload":{}}'),
            ("payload fails validation", b'{"event_type":"competitor_price","payload":{}}'),
        ],
    )
    def test_poison_messages_are_rejected(self, label: str, raw: bytes) -> None:
        with pytest.raises(EventDecodeError):
            deserialize(raw)


class TestPayloadValidation:
    """Requirement 8's rejection list, one test per item."""

    @pytest.mark.parametrize("price", [0.0, -1.0, -6200.0])
    def test_non_positive_prices_rejected(self, price: float) -> None:
        with pytest.raises(ValueError):
            competitor_payload(price=price)

    def test_absurd_price_rejected(self) -> None:
        with pytest.raises(ValueError):
            competitor_payload(price=50_000_000.0)

    def test_unknown_room_type_rejected(self) -> None:
        with pytest.raises(ValueError):
            competitor_payload(room_type="penthouse")

    def test_unknown_competitor_rejected(self) -> None:
        with pytest.raises(ValueError):
            competitor_payload(competitor="some_ota")

    def test_missing_required_field_rejected(self) -> None:
        with pytest.raises(ValueError):
            CompetitorPricePayload(hotel_id="H001", price=100.0)  # type: ignore[call-arg]

    def test_empty_hotel_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            competitor_payload(hotel_id="")

    def test_unparseable_date_rejected(self) -> None:
        with pytest.raises(ValueError):
            competitor_payload(check_in_date="15/09/2026")

    def test_implausibly_distant_date_rejected(self) -> None:
        far = date.today() + timedelta(days=MAX_HORIZON_DAYS + 30)
        with pytest.raises(ValueError, match="days ahead"):
            competitor_payload(check_in_date=far)

    def test_unexpected_field_rejected(self) -> None:
        """extra='forbid': an unknown field is a version mismatch or a bug."""
        with pytest.raises(ValueError):
            competitor_payload(discount_code="SUMMER20")

    def test_currency_is_normalised(self) -> None:
        assert competitor_payload(currency="inr").currency == "INR"

    def test_booking_with_zero_night_stay_rejected(self) -> None:
        with pytest.raises(ValueError, match="check_out_date"):
            booking_payload(check_out_date=STAY)

    def test_booking_with_more_cancellations_than_bookings_rejected(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            booking_payload(booking_count=2, cancellation_count=5)

    def test_booking_lead_time_is_derived(self) -> None:
        assert booking_payload().lead_time_days == 10

    @pytest.mark.parametrize("value", [-0.1, 1.4])
    def test_demand_scores_outside_zero_one_rejected(self, value: float) -> None:
        with pytest.raises(ValueError):
            demand_payload(search_demand=value)

    def test_prediction_payload_requires_positive_prices(self) -> None:
        with pytest.raises(ValueError):
            PricePredictionPayload(
                hotel_id="H001",
                room_type=RoomType.DELUXE,
                check_in_date=STAY,
                prediction_id="p1",
                base_price=5000,
                raw_recommended_price=0,
                final_recommended_price=6000,
                price_change_percent=1.0,
                blended_demand=0.8,
                confidence=0.9,
                model_version="v1.0",
            )


# --------------------------------------------------------------------------- #
# Producer
# --------------------------------------------------------------------------- #


class TestProducer:
    def test_sends_to_the_configured_topic_keyed_by_hotel(self, settings) -> None:
        client = FakeKafkaClient()
        producer = EventProducer(settings, client=client)

        assert producer.send(competitor_payload(), TopicName.COMPETITOR_PRICES) is True

        record = client.records[0]
        assert record["topic"] == settings.kafka.topic_competitor
        assert record["key"] == b"H001"
        assert json.loads(record["value"])["payload"]["price"] == 6200.0
        assert producer.stats.sent == 1

    def test_disabled_kafka_drops_events_without_raising(
        self, settings, monkeypatch
    ) -> None:
        """A pricing API must not fail because an analytics topic is off."""
        monkeypatch.setattr(settings.kafka, "enabled", False)
        producer = EventProducer(settings, client=FakeKafkaClient())

        assert producer.send(competitor_payload(), TopicName.COMPETITOR_PRICES) is False
        assert producer.stats.dropped_disabled == 1
        assert producer.stats.sent == 0

    def test_disabled_kafka_raises_in_strict_mode(self, settings, monkeypatch) -> None:
        monkeypatch.setattr(settings.kafka, "enabled", False)
        producer = EventProducer(settings, client=FakeKafkaClient())
        with pytest.raises(KafkaUnavailable):
            producer.send(competitor_payload(), TopicName.COMPETITOR_PRICES, strict=True)

    def test_unreachable_broker_degrades_instead_of_crashing(
        self, settings, monkeypatch
    ) -> None:
        producer = EventProducer(settings)
        monkeypatch.setattr(producer, "_connect", lambda: None)

        assert producer.send(competitor_payload(), TopicName.COMPETITOR_PRICES) is False
        assert producer.stats.failed == 1

    def test_unreachable_broker_is_not_retried_on_every_send(
        self, settings, monkeypatch
    ) -> None:
        """Once the broker is known down, paying the connect cost per event
        would turn a degraded dependency into a latency incident."""
        attempts = {"count": 0}

        def _fail() -> None:
            attempts["count"] += 1
            return None

        producer = EventProducer(settings)
        monkeypatch.setattr(producer, "_connect", _fail)

        for _ in range(5):
            producer.send(competitor_payload(), TopicName.COMPETITOR_PRICES)

        assert attempts["count"] == 1
        producer.reset()
        producer.send(competitor_payload(), TopicName.COMPETITOR_PRICES)
        assert attempts["count"] == 2

    def test_send_failure_is_counted_and_reported(self, settings) -> None:
        client = FakeKafkaClient(fail_on=1)
        producer = EventProducer(settings, client=client)

        assert producer.send(competitor_payload(), TopicName.COMPETITOR_PRICES) is True
        assert producer.send(competitor_payload(), TopicName.COMPETITOR_PRICES) is False
        assert producer.stats.failed == 1
        assert "KafkaTimeoutError" in producer.stats.last_error

    def test_send_many_reports_the_accepted_count(self, settings) -> None:
        producer = EventProducer(settings, client=FakeKafkaClient())
        payloads = [competitor_payload(competitor=c) for c in Competitor]
        assert producer.send_many(payloads, TopicName.COMPETITOR_PRICES) == 4

    def test_context_manager_closes_the_client(self, settings) -> None:
        """linger_ms batches records; an unflushed exit loses the buffer."""
        client = FakeKafkaClient()
        with EventProducer(settings, client=client) as producer:
            producer.send(competitor_payload(), TopicName.COMPETITOR_PRICES)
        assert client.closed is True


# --------------------------------------------------------------------------- #
# Handlers and the consumer core
# --------------------------------------------------------------------------- #


class TestConsumerProcessing:
    def test_competitor_event_is_persisted(self, consumer, db_engine) -> None:
        consumer.process_records([serialize(competitor_payload())])

        with sessionmaker(bind=db_engine)() as session:
            row = session.execute(select(CompetitorPrice)).scalar_one()
        assert row.hotel_id == "H001"
        assert row.price == 6200.0
        assert row.event_id is not None
        assert consumer.stats.written == 1

    def test_observation_time_comes_from_the_envelope(self, consumer, db_engine) -> None:
        """A message that sat in a topic for an hour still describes the rate as
        it was when it was scraped."""
        observed = datetime(2026, 9, 1, 6, 30, tzinfo=timezone.utc)
        envelope = EventEnvelope.wrap(competitor_payload(), timestamp=observed)
        consumer.process_records([envelope.to_bytes()])

        with sessionmaker(bind=db_engine)() as session:
            row = session.execute(select(CompetitorPrice)).scalar_one()
        assert row.collected_at.replace(tzinfo=timezone.utc) == observed

    def test_redelivery_is_a_no_op(self, consumer, db_engine) -> None:
        """At-least-once delivery replays messages; the handler must absorb it."""
        raw = serialize(competitor_payload())
        consumer.process_records([raw])
        consumer.process_records([raw])

        with sessionmaker(bind=db_engine)() as session:
            count = session.execute(
                select(func.count()).select_from(CompetitorPrice.__table__)
            ).scalar_one()
        assert count == 1
        assert consumer.stats.written == 1
        assert consumer.stats.duplicates == 1

    def test_poison_message_is_skipped_not_fatal(self, consumer, db_engine) -> None:
        """One malformed record must not block its partition forever."""
        consumer.process_records(
            [b"<html>captcha</html>", serialize(competitor_payload())]
        )

        with sessionmaker(bind=db_engine)() as session:
            assert session.execute(select(func.count()).select_from(
                CompetitorPrice.__table__)).scalar_one() == 1
        assert consumer.stats.poison == 1
        assert consumer.stats.written == 1

    def test_unknown_hotel_is_rejected_by_the_validator(self, consumer) -> None:
        consumer.process_records([serialize(competitor_payload(hotel_id="H999"))])
        assert consumer.stats.rejected == 1
        assert consumer.stats.by_reject_reason == {"unknown_hotel": 1}
        assert consumer.stats.written == 0

    def test_room_type_the_hotel_does_not_sell_is_rejected(
        self, db_engine, seeded_session
    ) -> None:
        from database.models import Room

        seeded_session.execute(
            Room.__table__.delete().where(Room.room_type == RoomType.SUITE)
        )
        seeded_session.commit()

        factory = sessionmaker(bind=db_engine, expire_on_commit=False, future=True)
        validator = EventValidator(reference=ReferenceData())
        validator.reference.refresh(seeded_session)
        consumer = EventConsumer(
            [TopicName.COMPETITOR_PRICES], validator=validator, session_factory=factory
        )

        consumer.process_records([serialize(competitor_payload(room_type=RoomType.SUITE))])
        assert consumer.stats.by_reject_reason == {"unknown_room_type": 1}

    def test_booking_event_is_persisted(self, consumer, db_engine) -> None:
        consumer.process_records([serialize(booking_payload())])

        with sessionmaker(bind=db_engine)() as session:
            row = session.execute(select(Booking)).scalar_one()
        assert row.booking_count == 6
        assert row.lead_time_days == 10

    def test_demand_signal_upserts_rather_than_duplicating(
        self, consumer, db_engine
    ) -> None:
        """Demand signals are a current view, not an append-only log."""
        consumer.process_records([serialize(demand_payload(search_demand=0.4))])
        consumer.process_records([serialize(demand_payload(search_demand=0.9))])

        with sessionmaker(bind=db_engine)() as session:
            rows = session.execute(select(DemandFeature)).scalars().all()
        assert len(rows) == 1
        assert rows[0].search_demand == 0.9

    def test_demand_signal_does_not_clobber_derived_features(
        self, consumer, db_engine, seeded_session
    ) -> None:
        """The feature pipeline owns the derived half of the row; a signal
        update must leave it alone."""
        consumer.process_records([serialize(demand_payload(search_demand=0.4))])

        with sessionmaker(bind=db_engine, expire_on_commit=False)() as session:
            row = session.execute(select(DemandFeature)).scalar_one()
            row.occupancy_rate = 0.83
            row.target_demand = 0.79
            row.feature_version = "v1"
            session.commit()

        consumer.process_records([serialize(demand_payload(search_demand=0.95))])

        with sessionmaker(bind=db_engine)() as session:
            row = session.execute(select(DemandFeature)).scalar_one()
        assert row.search_demand == 0.95
        assert row.occupancy_rate == 0.83
        assert row.target_demand == 0.79
        assert row.feature_version == "v1"

    def test_calendar_columns_are_recomputed_not_trusted(
        self, consumer, db_engine
    ) -> None:
        """A producer that disagrees about which day is a Saturday must not be
        able to poison the feature store."""
        consumer.process_records([serialize(demand_payload(stay_date=date(2026, 9, 19)))])

        with sessionmaker(bind=db_engine)() as session:
            row = session.execute(select(DemandFeature)).scalar_one()
        assert row.is_weekend is True  # 2026-09-19 is a Saturday
        assert row.day_of_week == 5
        assert row.season is Season.MONSOON

    def test_price_predictions_are_ignored_by_the_consumer(
        self, consumer, db_engine
    ) -> None:
        """The API already persisted them; re-inserting would duplicate rows."""
        payload = PricePredictionPayload(
            hotel_id="H001",
            room_type=RoomType.DELUXE,
            check_in_date=STAY,
            prediction_id="pred-1",
            base_price=6400.0,
            raw_recommended_price=7100.0,
            final_recommended_price=6900.0,
            price_change_percent=7.8,
            blended_demand=0.81,
            confidence=0.88,
            model_version="v1.0",
        )
        consumer.process_records([serialize(payload)])
        assert consumer.stats.ignored == 1
        assert consumer.stats.written == 0

    def test_mixed_batch_is_handled_in_one_transaction(self, consumer, db_engine) -> None:
        consumer.process_records(
            [
                serialize(competitor_payload()),
                serialize(booking_payload()),
                serialize(demand_payload()),
                b"garbage",
                serialize(competitor_payload(hotel_id="H999")),
            ]
        )
        assert consumer.stats.written == 3
        assert consumer.stats.poison == 1
        assert consumer.stats.rejected == 1
        assert consumer.stats.batches == 1

    def test_database_error_rolls_back_and_propagates(
        self, consumer, db_engine, monkeypatch
    ) -> None:
        """Offsets must not be committed for a batch the database refused."""
        def _explode(envelope, payload, session):
            raise SQLAlchemyError("connection reset")

        monkeypatch.setitem(consumer.handlers, EventType.COMPETITOR_PRICE, _explode)

        with pytest.raises(SQLAlchemyError):
            consumer.process_records([serialize(competitor_payload())])

        assert consumer.stats.db_errors == 1
        with sessionmaker(bind=db_engine)() as session:
            assert session.execute(
                select(func.count()).select_from(CompetitorPrice.__table__)
            ).scalar_one() == 0

    def test_batch_write_then_read_back(self, consumer, db_engine) -> None:
        """The Phase 3 acceptance criterion: event published -> row in Postgres."""
        payloads = [
            competitor_payload(competitor=c, price=6000.0 + index * 100)
            for index, c in enumerate(Competitor)
        ]
        consumer.process_records([serialize(p) for p in payloads])

        with sessionmaker(bind=db_engine)() as session:
            rows = session.execute(
                select(CompetitorPrice).order_by(CompetitorPrice.price)
            ).scalars().all()

        assert [r.competitor for r in rows] == list(Competitor)
        assert [r.price for r in rows] == [6000.0, 6100.0, 6200.0, 6300.0]
