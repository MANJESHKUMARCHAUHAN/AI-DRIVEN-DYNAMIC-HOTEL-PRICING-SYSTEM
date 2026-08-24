"""Kafka streaming layer: topic catalogue, producer, consumer.

Named ``streaming`` rather than ``kafka`` on purpose. A top-level package named
``kafka`` would shadow the ``kafka-python`` library on ``sys.path``, so
``from kafka import KafkaProducer`` would resolve to this package and fail with a
misleading ImportError. See ADR-001 in ``docs/architecture.md``.

Phase 1 provides :mod:`streaming.topics`. The producer and consumer arrive in
Phase 3.
"""

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
    "TopicName",
    "TopicSpec",
    "all_topic_names",
    "build_topic_specs",
    "consumer_config",
    "producer_config",
    "resolve_topic",
]
