"""Kafka streaming layer: event contracts, topic catalogue, producer, consumer.

Named ``streaming`` rather than ``kafka`` on purpose. A top-level package named
``kafka`` would shadow the ``kafka-python`` library on ``sys.path``, so
``from kafka import KafkaProducer`` would resolve to this package and fail with a
misleading ImportError. See ADR-001 in ``docs/architecture.md``.

Module map::

    events.py     the wire format -- envelope, typed payloads, (de)serialisation
    topics.py     the catalogue: names, partitions, retention. No Kafka import.
    admin.py      cluster operations: create topics, wait for the broker
    producer.py   publish, with graceful degradation when Kafka is absent
    consumer.py   poll, validate, persist, then commit offsets
    handlers.py   one persistence function per event type, all idempotent

Only ``events`` and ``topics`` are re-exported here. The producer and consumer
import ``kafka-python`` lazily, so importing this package never requires a Kafka
client to be installed.
"""

from streaming.events import (
    SCHEMA_VERSION,
    BookingPayload,
    CompetitorPricePayload,
    DemandSignalPayload,
    EventDecodeError,
    EventEnvelope,
    EventPayload,
    EventType,
    PricePredictionPayload,
    deserialize,
    serialize,
)
from streaming.topics import (
    TopicName,
    TopicSpec,
    all_topic_names,
    build_topic_specs,
    consumer_config,
    producer_config,
    resolve_topic,
)

__all__ = [
    "SCHEMA_VERSION",
    "BookingPayload",
    "CompetitorPricePayload",
    "DemandSignalPayload",
    "EventDecodeError",
    "EventEnvelope",
    "EventPayload",
    "EventType",
    "PricePredictionPayload",
    "TopicName",
    "TopicSpec",
    "all_topic_names",
    "build_topic_specs",
    "consumer_config",
    "deserialize",
    "producer_config",
    "resolve_topic",
    "serialize",
]
