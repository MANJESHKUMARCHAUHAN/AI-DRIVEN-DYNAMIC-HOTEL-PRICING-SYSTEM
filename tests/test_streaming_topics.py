"""Phase 1 tests: Kafka topic catalogue and client configuration.

No broker is required. These verify the declarative half of the streaming layer
so Phase 3 starts from a known-good catalogue.
"""

from __future__ import annotations

import pytest

from streaming import (
    TopicName,
    TopicSpec,
    all_topic_names,
    build_topic_specs,
    consumer_config,
    producer_config,
    resolve_topic,
)


class TestTopicCatalogue:
    def test_all_four_topics_are_declared(self, settings) -> None:
        specs = build_topic_specs(settings)
        assert set(specs) == set(TopicName)
        assert len(specs) == 4

    def test_names_come_from_configuration(self, settings) -> None:
        specs = build_topic_specs(settings)
        assert specs[TopicName.COMPETITOR_PRICES].name == "hotel.competitor_prices"
        assert specs[TopicName.PRICE_PREDICTIONS].name == "hotel.price_predictions"

    def test_resolve_topic_maps_logical_to_physical(self, settings) -> None:
        assert resolve_topic(TopicName.BOOKING_EVENTS, settings) == "hotel.booking_events"

    def test_topic_names_are_unique(self, settings) -> None:
        names = all_topic_names(settings)
        assert len(names) == len(set(names))

    def test_every_spec_is_described(self, settings) -> None:
        for spec in build_topic_specs(settings).values():
            assert spec.description.strip()

    def test_specs_are_immutable(self, settings) -> None:
        spec = build_topic_specs(settings)[TopicName.DEMAND_EVENTS]
        with pytest.raises(Exception):
            spec.name = "mutated"  # type: ignore[misc]

    def test_broker_configs_are_stringly_typed(self, settings) -> None:
        # Kafka's admin API rejects non-string config values.
        configs = build_topic_specs(settings)[TopicName.DEMAND_EVENTS].configs
        assert all(isinstance(v, str) for v in configs.values())

    def test_predictions_retain_for_less_time(self, settings) -> None:
        specs = build_topic_specs(settings)
        assert (
            specs[TopicName.PRICE_PREDICTIONS].retention_ms
            < specs[TopicName.COMPETITOR_PRICES].retention_ms
        )


class TestClientConfig:
    def test_producer_is_configured_durably(self, settings) -> None:
        cfg = producer_config(settings)
        assert cfg["acks"] == "all"
        assert cfg["retries"] >= 1
        assert cfg["bootstrap_servers"] == ["test-kafka:9092"]

    def test_consumer_never_auto_commits(self, settings) -> None:
        # At-least-once semantics depend on committing after the DB write.
        assert consumer_config(settings)["enable_auto_commit"] is False

    def test_consumer_group_suffix_isolates_consumers(self, settings) -> None:
        base = consumer_config(settings)["group_id"]
        suffixed = consumer_config(settings, group_suffix="-features")["group_id"]
        assert suffixed == f"{base}-features"

    def test_client_ids_are_distinguishable(self, settings) -> None:
        assert producer_config(settings)["client_id"].endswith("-producer")
        assert consumer_config(settings)["client_id"].endswith("-consumer")

    def test_configs_are_plain_dicts(self, settings) -> None:
        # They are splatted into kafka-python constructors as **kwargs.
        assert isinstance(producer_config(settings), dict)
        assert isinstance(consumer_config(settings), dict)


class TestTopicSpecDataclass:
    def test_configs_render_retention(self) -> None:
        spec = TopicSpec(
            name="t", partitions=1, replication_factor=1,
            retention_ms=1000, description="d",
        )
        assert spec.configs["retention.ms"] == "1000"
        assert spec.configs["cleanup.policy"] == "delete"
