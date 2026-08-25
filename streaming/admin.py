"""Cluster administration: create topics, describe them, wait for the broker.

Split from :mod:`streaming.topics` on purpose. ``topics.py`` is the *catalogue* --
pure data, no Kafka import, usable from the API and the dashboard and testable
without a broker. This module is the part that actually talks to a cluster.

Topics are created explicitly rather than left to ``auto.create.topics.enable``.
An auto-created topic gets the broker's defaults: one partition, the broker's
retention, no thought given to either. Creating them here means partition count
and retention are decisions recorded in code and reviewable in a diff.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from config import Settings, get_settings
from monitoring.logging_config import get_logger
from streaming.topics import TopicSpec, build_topic_specs

logger = get_logger(__name__)


class BrokerUnavailable(RuntimeError):
    """Raised when the cluster does not answer within the configured retries."""


def _admin_client(settings: Settings, *, timeout_ms: int = 10_000) -> Any:
    """Build a ``KafkaAdminClient``. Lazy import -- see the module docstring."""
    from kafka.admin import KafkaAdminClient

    return KafkaAdminClient(
        bootstrap_servers=settings.kafka.bootstrap_servers_list,
        client_id=f"{settings.kafka.client_id}-admin",
        request_timeout_ms=timeout_ms,
    )


def broker_available(settings: Optional[Settings] = None, *, timeout_ms: int = 5_000) -> bool:
    """Return ``True`` if the cluster answers a metadata request. Never raises."""
    settings = settings or get_settings()
    if not settings.kafka.enabled:
        return False
    try:
        client = _admin_client(settings, timeout_ms=timeout_ms)
        try:
            client.list_topics()
            return True
        finally:
            client.close()
    except Exception as exc:
        logger.debug("Broker probe failed: %s", type(exc).__name__)
        return False


def wait_for_broker(
    settings: Optional[Settings] = None,
    *,
    retries: int = 30,
    delay_seconds: float = 2.0,
) -> None:
    """Block until the cluster answers, or raise :class:`BrokerUnavailable`.

    Kafka in KRaft mode accepts TCP connections before it has finished electing
    a controller, so "the port is open" is not the same as "the cluster works".
    A metadata request is the cheapest thing that proves both.
    """
    settings = settings or get_settings()
    for attempt in range(1, retries + 1):
        if broker_available(settings):
            logger.info("Kafka reachable (attempt %d/%d)", attempt, retries)
            return
        if attempt == retries:
            raise BrokerUnavailable(
                f"Kafka at {settings.kafka.bootstrap_servers} unreachable after "
                f"{retries} attempts"
            )
        logger.warning(
            "Kafka not ready (attempt %d/%d), retrying in %.1fs",
            attempt,
            retries,
            delay_seconds,
        )
        time.sleep(delay_seconds)


def existing_topics(settings: Optional[Settings] = None) -> List[str]:
    """Every topic the cluster currently holds, including internal ones."""
    settings = settings or get_settings()
    client = _admin_client(settings)
    try:
        return sorted(client.list_topics())
    finally:
        client.close()


def create_topics(
    settings: Optional[Settings] = None, *, dry_run: bool = False
) -> Dict[str, str]:
    """Create every topic in the catalogue that does not already exist.

    Idempotent: an existing topic is reported as ``"exists"``, not an error.
    Note that Kafka will not *modify* an existing topic's partition count here --
    changing that is a deliberate operation, not a side effect of a start-up
    script.

    Args:
        dry_run: Report what would happen without touching the cluster.

    Returns:
        Physical topic name -> ``"created"`` | ``"exists"`` | ``"would create"``.
    """
    settings = settings or get_settings()
    specs = build_topic_specs(settings)

    if dry_run:
        return {spec.name: "would create" for spec in specs.values()}

    from kafka.admin import NewTopic
    from kafka.errors import TopicAlreadyExistsError

    client = _admin_client(settings)
    result: Dict[str, str] = {}
    try:
        present = set(client.list_topics())
        to_create = [spec for spec in specs.values() if spec.name not in present]

        for spec in specs.values():
            if spec.name in present:
                result[spec.name] = "exists"

        if not to_create:
            logger.info("All %d topic(s) already exist", len(specs))
            return result

        new_topics = [
            NewTopic(
                name=spec.name,
                num_partitions=spec.partitions,
                replication_factor=spec.replication_factor,
                topic_configs=spec.configs,
            )
            for spec in to_create
        ]

        try:
            client.create_topics(new_topics=new_topics, validate_only=False)
            for spec in to_create:
                result[spec.name] = "created"
                logger.info(
                    "Created topic %s (partitions=%d, replication=%d, retention=%dh)",
                    spec.name,
                    spec.partitions,
                    spec.replication_factor,
                    spec.retention_ms // 3_600_000,
                )
        except TopicAlreadyExistsError:
            # Another process won the race. That is a success, not a failure.
            for spec in to_create:
                result[spec.name] = "exists"
            logger.info("Topics were created concurrently by another process")
    finally:
        client.close()

    return result


def delete_topics(settings: Optional[Settings] = None) -> List[str]:
    """Delete every topic in the catalogue. Destructive; used by tests and resets."""
    settings = settings or get_settings()
    specs = build_topic_specs(settings)
    names = [spec.name for spec in specs.values()]

    from kafka.errors import UnknownTopicOrPartitionError

    client = _admin_client(settings)
    try:
        present = [n for n in names if n in set(client.list_topics())]
        if not present:
            return []
        try:
            client.delete_topics(present)
            logger.warning("Deleted topic(s): %s", ", ".join(present))
            return present
        except UnknownTopicOrPartitionError:  # pragma: no cover - race
            return []
    finally:
        client.close()


def describe_topics(settings: Optional[Settings] = None) -> List[Dict[str, Any]]:
    """Partition and replica layout for the catalogue's topics.

    Used by ``scripts/create_topics.py --describe`` and by the monitoring page,
    which needs to show that the pipeline's plumbing is actually there.
    """
    settings = settings or get_settings()
    specs: Dict[Any, TopicSpec] = build_topic_specs(settings)
    wanted = {spec.name for spec in specs.values()}

    client = _admin_client(settings)
    try:
        present = [name for name in client.list_topics() if name in wanted]
        if not present:
            return []
        descriptions = client.describe_topics(present)
    finally:
        client.close()

    rows: List[Dict[str, Any]] = []
    for description in descriptions:
        partitions = description.get("partitions", [])
        rows.append(
            {
                "topic": description.get("topic"),
                "partitions": len(partitions),
                "replicas": max((len(p.get("replicas", [])) for p in partitions), default=0),
                "leaders": sorted({p.get("leader") for p in partitions}),
            }
        )
    return sorted(rows, key=lambda r: str(r["topic"]))


__all__ = [
    "BrokerUnavailable",
    "broker_available",
    "create_topics",
    "delete_topics",
    "describe_topics",
    "existing_topics",
    "wait_for_broker",
]
