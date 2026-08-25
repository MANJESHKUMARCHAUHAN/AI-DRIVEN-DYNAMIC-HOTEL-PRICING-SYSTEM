# A single-node Apache Kafka broker in KRaft mode, built from the Apache release
# tarball.
#
# WHY THIS EXISTS
# ---------------
# docker-compose.yml asks for `apache/kafka:4.1.2`, the image the Kafka project
# publishes. On a network that can reach Docker Hub, nothing here is needed --
# Compose pulls that image and this directory is dead weight.
#
# It exists because Docker only pulls an image it does not already have locally.
# On a restricted network (corporate TLS interception, an air-gapped build box,
# a blocked registry CDN) you can build this file and tag the result
# `apache/kafka:4.1.2`; Compose then finds it locally and the stack comes up
# unchanged. No compose file edits, no environment-specific overrides:
#
#     docker build -t apache/kafka:4.1.2 docker/kafka
#
# Only the Apache archive host and an official Java base image are required.
#
# The image is deliberately close in behaviour to the upstream one: the same
# KAFKA_* environment variables configure it, so a reader who knows the official
# image already knows this one.

FROM eclipse-temurin:17-jre

# 4.x is KRaft-only: the ZooKeeper code paths were removed from the release
# itself, so ADR-002 is enforced by the distribution rather than by our config.
ARG KAFKA_VERSION=4.1.2
ARG SCALA_VERSION=2.13

LABEL org.opencontainers.image.title="apache-kafka-kraft" \
      org.opencontainers.image.description="Single-node Apache Kafka in KRaft mode" \
      org.opencontainers.image.version="${KAFKA_VERSION}"

ENV KAFKA_HOME=/opt/kafka \
    PATH="/opt/kafka/bin:${PATH}"

# ADD fetches over the daemon's network and needs no package manager inside the
# image -- which matters, because the same networks that block a registry
# usually block the distro mirrors too.
#
# dlcdn.apache.org is the Apache CDN and carries only *current* releases. When
# ${KAFKA_VERSION} ages out of it, switch the host to archive.apache.org, which
# keeps every release forever:
#
#     https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/...
#
# The archive host is markedly slower, which is why it is the fallback and not
# the default.
ADD https://dlcdn.apache.org/kafka/${KAFKA_VERSION}/kafka_${SCALA_VERSION}-${KAFKA_VERSION}.tgz /tmp/kafka.tgz

RUN mkdir -p "${KAFKA_HOME}" \
    && tar -xzf /tmp/kafka.tgz -C "${KAFKA_HOME}" --strip-components=1 \
    && rm -f /tmp/kafka.tgz \
    && mkdir -p /var/lib/kafka/data \
    && groupadd -r kafka && useradd -r -g kafka kafka \
    && chown -R kafka:kafka "${KAFKA_HOME}" /var/lib/kafka

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

USER kafka
WORKDIR ${KAFKA_HOME}

# 9092 client traffic, 9093 the KRaft controller quorum.
EXPOSE 9092 9093

# A broker that answers an API request is up; a running process is not enough.
HEALTHCHECK --interval=10s --timeout=10s --start-period=30s --retries=10 \
    CMD /opt/kafka/bin/kafka-broker-api-versions.sh \
        --bootstrap-server localhost:9092 > /dev/null 2>&1 || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
