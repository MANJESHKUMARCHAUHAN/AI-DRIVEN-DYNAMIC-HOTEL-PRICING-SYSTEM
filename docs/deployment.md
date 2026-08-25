# Deployment

Running the stack, operating it, and the failures you are most likely to hit.

---

## The stack

```
docker compose up --build
```

Seven services:

| Service | Image | Purpose |
|---|---|---|
| `postgres` | `postgres:16-alpine` | The database |
| `kafka` | `apache/kafka:4.1.2` | Broker, KRaft mode — no ZooKeeper |
| `init` | app image | One-shot: schema → topics → data → seed → features → train |
| `api` | app image | FastAPI on 8000 |
| `consumer` | app image | Kafka → PostgreSQL |
| `producer` | app image | Synthetic competitor rates → Kafka |
| `dashboard` | app image | Streamlit on 8501 |

`api`, `dashboard`, `consumer`, `producer` and `init` all run the **same image**
with a different `command`. They share every dependency, so three images would
mean three copies of Prophet, NumPy and scikit-learn.

- API and Swagger: <http://localhost:8000/docs>
- Dashboard: <http://localhost:8501>
- Postgres: `localhost:55432` (configurable — a developer machine very often has
  a native Postgres already on 5432)
- Kafka from the host: `localhost:29092`

### What `init` does

It runs to completion and exits; `api`, `dashboard`, `consumer` and `producer`
wait on `service_completed_successfully`. A cold `docker compose up` therefore
produces a stack that can already price a room, rather than one that needs six
manual commands afterwards.

```
--> schema          python -m database.init_db
--> topics          python scripts/create_topics.py
--> synthetic data  python scripts/generate_data.py
--> seeding         python scripts/seed_database.py        (~312,000 rows)
--> features        python scripts/build_features.py       (11,648 rows)
--> training        python scripts/train_models.py         (both models)
--> ready
```

About two minutes on a warm image.

---

## Networking

**Service names, never `localhost`.** Inside a container `localhost` is that
container. `postgres`, `kafka` and `api` are DNS names on the Compose network,
and every inter-service setting uses them:

```yaml
POSTGRES_HOST: postgres          # not localhost
KAFKA_BOOTSTRAP_SERVERS: kafka:9092
API_BASE_URL: http://api:8000    # the dashboard reaching the API
```

This is also why `docker-compose.yml` overrides `POSTGRES_HOST` rather than
reading it from `.env`, which holds host-machine values for running things
natively.

### Kafka's two listeners

```yaml
KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092,EXTERNAL://localhost:29092
```

Two on purpose. A broker tells the client which address to connect to, and only
one of those names resolves on each side: containers need `kafka`, tools on your
machine need `localhost`. A single listener cannot serve both.

---

## Health checks

| Service | Check | Why |
|---|---|---|
| `postgres` | `pg_isready -U … -d …` | Naming the user and database matters: bare `pg_isready` returns success while initdb is still running |
| `kafka` | `kafka-broker-api-versions.sh` | In KRaft mode the socket opens well before the cluster can serve metadata |
| `api` | `GET /health` via urllib | Uses the venv's Python, so the image needs no curl |
| `dashboard` | `GET /healthz` | Streamlit's own endpoint |

`depends_on: condition: service_healthy` reduces the startup race but does not
remove it, so the application code retries as well (`wait_for_database`,
`wait_for_broker`). A cold start converges instead of crash-looping.

---

## Operations

```bash
make up          # build and start everything
make down        # stop, keep the data
make clean       # stop and delete the volumes — destroys everything
make logs        # follow every service
make ps          # status
make shell       # a shell inside the API container

docker compose logs -f api
docker compose restart consumer
docker compose exec postgres psql -U pricing -d hotel_pricing
```

### Retraining a running stack

```bash
docker compose exec api python scripts/train_models.py
curl -X POST http://localhost:8000/api/v1/models/train -d '{}' -H 'Content-Type: application/json'
```

Either writes a new version into the shared `model-artifacts` volume. The API
endpoint reloads immediately; the CLI needs an `api` restart, or a call to the
endpoint, to pick it up.

### Monitoring

```bash
docker compose exec api python scripts/monitor.py
docker compose exec api python scripts/monitor.py --fail-on warning   # for cron
```

The report lands in the shared `app-data` volume, where the dashboard's
Monitoring page reads it.

---

## Volumes

| Volume | Holds | Mounted by |
|---|---|---|
| `postgres-data` | The database | `postgres` |
| `kafka-data` | Topic logs and KRaft metadata | `kafka` |
| `model-artifacts` | Trained models, feature list, training reports | `init`, `api`, `dashboard` (read-only) |
| `app-data` | Generated CSVs, monitoring reports | `init`, `api`, `consumer`, `producer`, `dashboard` (read-only) |

The dashboard mounts both **read-only**. It is a read-only consumer of the API
and must never write into either.

---

## Configuration

Compose reads `.env` for host-side values and overrides the inter-service ones.
Everything else has a default:

```bash
POSTGRES_PASSWORD=change_me        # required in anything non-local
POSTGRES_HOST_PORT=55432           # host port for the database
API_HOST_PORT=8000
DASHBOARD_HOST_PORT=8501
KAFKA_HOST_PORT=29092
MIN_PRICE=2500
MAX_PRICE=25000
MAX_DAILY_CHANGE_PERCENT=0.15
LOG_LEVEL=INFO
```

`ENVIRONMENT=production` activates a safety check: the application refuses to
start if `POSTGRES_PASSWORD` is still a development default, or if `DEBUG` is
true.

---

## Troubleshooting

### `init` fails with `ModuleNotFoundError: No module named 'sqlalchemy'`

The container's `command` is using a **login shell** (`bash -lc`). A login shell
re-reads `/etc/profile` and rebuilds `PATH`, dropping the `/opt/venv/bin` the
image sets — so every command runs against the system Python. Use `bash -c`.

### The registry cannot be reached

If `docker pull` fails with `tls: protocol version not supported` or a TLS
handshake timeout, the network is likely intercepting registry traffic. Two
workarounds, in order of preference:

1. **The AWS ECR public mirror** carries every *official library* image:

   ```bash
   docker pull public.ecr.aws/docker/library/postgres:16-alpine
   docker tag public.ecr.aws/docker/library/postgres:16-alpine postgres:16-alpine
   ```

   Same for `python:3.11-slim-bookworm`. Compose then finds them locally.

2. **Build Kafka yourself.** Kafka has no official *library* image, so the mirror
   does not carry it. `docker/kafka/` builds it from the Apache release tarball
   on top of an official Java base:

   ```bash
   make kafka-image      # docker build -t apache/kafka:4.1.2 docker/kafka
   ```

   Tagged exactly as `docker-compose.yml` asks for, so nothing else changes.
   Uses `dlcdn.apache.org`; when a version ages out of the CDN, switch the
   `ADD` line to `archive.apache.org`, which keeps every release forever but is
   markedly slower.

### Port 5432 already in use

A native PostgreSQL is running. Change `POSTGRES_HOST_PORT` in `.env` — the
default is already 55432 for this reason.

### Kafka restarts in a loop

Almost always a replication factor above 1 on a single-broker cluster: the
internal topics can never reach their required ISR and the broker hangs on boot.
Every `*_REPLICATION_FACTOR` must be 1.

If you have changed `CLUSTER_ID` since the volume was created, the broker will
refuse to start against "another" cluster's metadata. `make clean` and start
again.

### The API starts but `/health` says models are `unavailable`

Nothing has been trained. The API is working correctly — pricing falls back to
stored historical demand. Run `docker compose exec api python
scripts/train_models.py`, or `POST /api/v1/models/train`.

This is by design: an API that refuses to start before a training job has run
cannot be deployed before it is trained.

### `POST /pricing/predict` returns 409 on `/models/train`

There is not enough labelled data. Run the seed and feature steps first:

```bash
docker compose exec api python scripts/seed_database.py
docker compose exec api python scripts/build_features.py
```

### The dashboard says the API is unreachable

Check `API_BASE_URL`. Inside Compose it must be `http://api:8000` — a service
name. `localhost` inside the dashboard container is the dashboard.

### Prophet fails with `'Prophet' object has no attribute 'stan_backend'`

`cmdstanpy` has been upgraded past the pinned 1.2.4. Versions 1.3+ tightened
path validation and reject the trimmed CmdStan directory the Prophet wheel
ships; Prophet swallows the real error and every later call fails with that
misleading message. Reinstall with `pip install -e ".[dev]"` — the pin lives in
`pyproject.toml` and is commented there as load-bearing.

---

## Production checklist

This project is built to be run locally and explained in an interview. Before
anything internet-facing:

- [ ] **Authentication.** The API is unauthenticated by design here.
- [ ] `ENVIRONMENT=production`, a real `POSTGRES_PASSWORD`, `DEBUG=false`.
- [ ] Managed PostgreSQL with backups; the Compose volume is not a backup.
- [ ] Kafka with at least three brokers and replication factor 3.
- [ ] Move `POST /models/train` to a task queue.
- [ ] Ship the JSON logs somewhere; the correlation id is already in every line.
- [ ] Run `scripts/monitor.py --fail-on warning` on a schedule and alert on it.
- [ ] Pin `MODEL_ACTIVE_VERSION` rather than serving whatever trained last.
- [ ] Replace the synthetic feed with a licensed competitor rate source.
- [ ] TLS termination and rate limiting in front of the API.

---

## Resource notes

| | |
|---|---|
| App image | ~1.7 GB (Prophet, scikit-learn and NumPy dominate) |
| Build time | ~10 min cold, seconds warm — requirements are a separate layer |
| Postgres after `init` | ~250 MB for 312,000 rows |
| API memory | ~400 MB with both models loaded |
| Training | ~20 s without Prophet backtesting, ~35 s with |
| Pricing latency | ~28 ms end to end |
