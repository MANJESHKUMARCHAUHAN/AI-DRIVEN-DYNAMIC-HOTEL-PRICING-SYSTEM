#!/usr/bin/env bash
#
# Start a single-node Kafka broker in KRaft mode.
#
# Configuration comes from KAFKA_* environment variables, matching the names the
# official apache/kafka image uses, so this container is a drop-in replacement.
# Anything not set falls back to a working single-node development default.
#
# KRaft (ADR-002) means no ZooKeeper: the broker is also the controller, the
# metadata lives in an internal log, and the cluster is one process instead of
# two. Fewer moving parts to explain and one less container to keep healthy.

set -euo pipefail

KAFKA_HOME="${KAFKA_HOME:-/opt/kafka}"
DATA_DIR="${KAFKA_LOG_DIRS:-/var/lib/kafka/data}"
CONFIG_FILE="/tmp/kraft-server.properties"

# A stable cluster id keeps the metadata log valid across container restarts.
# Derived from the node id rather than random so a restart re-attaches to the
# existing volume instead of refusing to start against "another" cluster.
CLUSTER_ID="${CLUSTER_ID:-hotel-pricing-kafka-01}"

node_id="${KAFKA_NODE_ID:-1}"

cat > "${CONFIG_FILE}" <<EOF
process.roles=${KAFKA_PROCESS_ROLES:-broker,controller}
node.id=${node_id}
controller.quorum.voters=${KAFKA_CONTROLLER_QUORUM_VOTERS:-${node_id}@localhost:9093}

listeners=${KAFKA_LISTENERS:-PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093}
advertised.listeners=${KAFKA_ADVERTISED_LISTENERS:-PLAINTEXT://localhost:9092}
listener.security.protocol.map=${KAFKA_LISTENER_SECURITY_PROTOCOL_MAP:-CONTROLLER:PLAINTEXT,PLAINTEXT:PLAINTEXT}
controller.listener.names=${KAFKA_CONTROLLER_LISTENER_NAMES:-CONTROLLER}
inter.broker.listener.name=${KAFKA_INTER_BROKER_LISTENER_NAME:-PLAINTEXT}

log.dirs=${DATA_DIR}
num.partitions=${KAFKA_NUM_PARTITIONS:-3}

# Single broker: every replication factor must be 1 or the internal topics can
# never reach their required ISR and the broker hangs at startup.
offsets.topic.replication.factor=${KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR:-1}
transaction.state.log.replication.factor=${KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR:-1}
transaction.state.log.min.isr=${KAFKA_TRANSACTION_STATE_LOG_MIN_ISR:-1}
default.replication.factor=${KAFKA_DEFAULT_REPLICATION_FACTOR:-1}

# Development convenience: no rebalance delay, and topics appear on first use so
# a mis-ordered start-up is a warning rather than a crash. Topic creation is
# still done explicitly by scripts/create_topics.py, which sets partitions and
# retention properly.
group.initial.rebalance.delay.ms=${KAFKA_GROUP_INITIAL_REBALANCE_DELAY_MS:-0}
auto.create.topics.enable=${KAFKA_AUTO_CREATE_TOPICS_ENABLE:-true}
log.retention.hours=${KAFKA_LOG_RETENTION_HOURS:-168}
EOF

# --ignore-formatted makes this idempotent: format on a fresh volume, no-op on
# an existing one. Without it, every restart with a persisted volume fails.
"${KAFKA_HOME}/bin/kafka-storage.sh" format \
    --cluster-id "${CLUSTER_ID}" \
    --config "${CONFIG_FILE}" \
    --ignore-formatted

echo "Starting Kafka ${KAFKA_VERSION:-3.9.0} (KRaft, node ${node_id}, cluster ${CLUSTER_ID})"
exec "${KAFKA_HOME}/bin/kafka-server-start.sh" "${CONFIG_FILE}"
