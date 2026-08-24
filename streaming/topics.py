"""Kafka topic catalogue and client configuration.

This module is deliberately free of any Kafka client import. It declares *what*
the topics are and *how* clients should be configured; the producer and consumer
(Phase 3) consume these declarations. Keeping the catalogue importable without a
broker -- or even without ``kafka-python`` installed -- means topic names can be
unit-tested and referenced from the API and dashboard for free.

See ADR-001 in ``docs/architecture.md`` for why this package is called
``streaming`` and not ``kafka``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from config import Settings, get_settings


class TopicName(str, Enum):
    """Logical topic identifiers used throughout the codebase.

    Code refers to ``TopicName.COMPETITOR_PRICES``; the physical topic string
    comes from configuration, so it can differ per environment.
    """

    COMPETITOR_PRICES = "competitor_prices"
    BOOKING_EVENTS = "booking_events"
    DEMAND_EVENTS = "demand_events"
    PRICE_PREDICTIONS = "price_predictions"


@dataclass(frozen=True)
class TopicSpec:
    """Everything needed to create a topic and reason about it.

    Attributes:
        name: Physical Kafka topic name.
        partitions: Partition count. Ordering is guaranteed per partition, and
            all events are keyed by ``hotel_id``, so a hotel's events stay
            ordered relative to each other.
        replication_factor: 1 for single-broker development.
        retention_ms: How long the broker keeps records.
        description: Why this topic exists.
    """

    name: str
    partitions: int
    replication_factor: int
    retention_ms: int
    description: str

    @property
    def configs(self) -> Dict[str, str]:
        """Topic-level broker configs, as Kafka expects them (string values)."""
        return {
            "retention.ms": str(self.retention_ms),
            "cleanup.policy": "delete",
        }


_DAY_MS = 24 * 60 * 60 * 1000


def build_topic_specs(settings: Optional[Settings] = None) -> Dict[TopicName, TopicSpec]:
    """Return the full topic catalogue for the given configuration."""
    settings = settings or get_settings()
    k = settings.kafka

    return {
        TopicName.COMPETITOR_PRICES: TopicSpec(
            name=k.topic_competitor,
            partitions=k.num_partitions,
            replication_factor=k.replication_factor,
            retention_ms=7 * _DAY_MS,
            description="Competitor rates observed by the ingestion layer.",
        ),
        TopicName.BOOKING_EVENTS: TopicSpec(
            name=k.topic_bookings,
            partitions=k.num_partitions,
            replication_factor=k.replication_factor,
            retention_ms=7 * _DAY_MS,
            description="Reservations and cancellations as they occur.",
        ),
        TopicName.DEMAND_EVENTS: TopicSpec(
            name=k.topic_demand,
            partitions=k.num_partitions,
            replication_factor=k.replication_factor,
            retention_ms=7 * _DAY_MS,
            description="Derived demand signals emitted by the feature pipeline.",
        ),
        TopicName.PRICE_PREDICTIONS: TopicSpec(
            name=k.topic_predictions,
            partitions=k.num_partitions,
            replication_factor=k.replication_factor,
            retention_ms=3 * _DAY_MS,
            description="Final recommended prices published by the API.",
        ),
    }


def resolve_topic(topic: TopicName, settings: Optional[Settings] = None) -> str:
    """Map a logical topic to its configured physical name."""
    return build_topic_specs(settings)[topic].name


def producer_config(settings: Optional[Settings] = None) -> Dict[str, Any]:
    """kafka-python ``KafkaProducer`` keyword arguments.

    ``acks='all'`` with retries is the durable setting: the leader waits for all
    in-sync replicas before acknowledging. On a single-broker dev cluster that is
    cheap, and it means the production configuration is the same code path.
    """
    settings = settings or get_settings()
    k = settings.kafka
    return {
        "bootstrap_servers": k.bootstrap_servers_list,
        "client_id": f"{k.client_id}-producer",
        "acks": k.producer_acks,
        "retries": k.producer_retries,
        "linger_ms": k.producer_linger_ms,
        "request_timeout_ms": k.request_timeout_ms,
        "retry_backoff_ms": k.retry_backoff_ms,
    }


def consumer_config(
    settings: Optional[Settings] = None, *, group_suffix: str = ""
) -> Dict[str, Any]:
    """kafka-python ``KafkaConsumer`` keyword arguments.

    ``enable_auto_commit`` is forced off. Offsets are committed by the consumer
    only after the database write succeeds, which is what makes the pipeline
    at-least-once rather than at-most-once.
    """
    settings = settings or get_settings()
    k = settings.kafka
    group = f"{k.consumer_group}{group_suffix}" if group_suffix else k.consumer_group
    return {
        "bootstrap_servers": k.bootstrap_servers_list,
        "client_id": f"{k.client_id}-consumer",
        "group_id": group,
        "auto_offset_reset": k.auto_offset_reset,
        "enable_auto_commit": k.enable_auto_commit,
        "max_poll_records": k.max_poll_records,
        "session_timeout_ms": k.session_timeout_ms,
        "request_timeout_ms": k.request_timeout_ms,
        "retry_backoff_ms": k.retry_backoff_ms,
    }


def all_topic_names(settings: Optional[Settings] = None) -> List[str]:
    """Every physical topic name, for creation and admin operations."""
    return [spec.name for spec in build_topic_specs(settings).values()]


__all__ = [
    "TopicName",
    "TopicSpec",
    "all_topic_names",
    "build_topic_specs",
    "consumer_config",
    "producer_config",
    "resolve_topic",
]
