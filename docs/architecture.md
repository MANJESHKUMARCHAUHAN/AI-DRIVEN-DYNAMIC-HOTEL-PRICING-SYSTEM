# Architecture — AI-Driven Dynamic Hotel Pricing System

> **Status:** Design baseline. Written before Phase 1 implementation.
> **Audience:** Engineers implementing the system, and interviewers evaluating it.
> This document is the contract. Code must conform to it; where code diverges, this document
> gets updated in the same change.

---

## 1. Purpose

A hotel does not have one correct room price. It has a *correct price for a given room type, on a
given date, given how full it currently is, how far out the booking is, what competitors are
charging, and how demand is trending.* Setting that price by hand across hundreds of
hotel × room-type × date combinations is not feasible, and setting it by a static rule
("weekends +20%") leaves money on the table in both directions.

This system computes that price automatically, continuously, and — critically — **explainably**.
Every price it emits comes with the arithmetic that produced it and the list of business rules
that were applied to it.

### What "done" means

| Capability | Acceptance |
|---|---|
| Competitor rates flow in continuously | Producer → Kafka → Consumer → Postgres, observable end-to-end |
| Demand is forecast, not guessed | Prophet produces 7/14/30-day forecasts with confidence intervals |
| Demand responds to context | GradientBoostingRegressor predicts demand from 14 engineered features |
| Price is explainable | Response contains base price + each named adjustment + final price |
| Price is safe | No emitted price can violate configured floor, ceiling, or daily-change cap |
| It is operable | `docker compose up --build` yields a working API + dashboard |
| It is verifiable | `pytest` passes against real logic, not mocks of itself |

### Non-goals

Explicitly out of scope, so nobody looks for them:

- **Real-money booking integration.** No PMS/channel-manager write-back. The system *recommends*
  prices; publishing them to inventory is a downstream concern.
- **Live Booking.com / Expedia scraping as the default path.** See ADR-004.
- **Multi-tenant SaaS.** Single-operator deployment.
- **Distributed training.** Models are small; single-node scikit-learn and Prophet are correct here.
- **Kubernetes.** Docker Compose is the deployment target.

---

## 2. Design principles

These drive the decisions further down. When two designs compete, the one better aligned with
these wins.

1. **Transparent over clever.** A pricing number nobody can explain is a pricing number nobody
   will deploy. Every adjustment is a named, logged, individually-inspectable factor.
2. **Guardrails are not optional and not advisory.** They are the last stage before emission.
   There is exactly one function that produces a final price, and it always runs guardrails.
3. **Degrade, don't die.** Kafka down → API still serves. Competitor data missing → fall back to
   the hotel's own recent rate and lower the confidence score. Model artifact missing → return a
   clear `503`, not a garbage number.
4. **Config comes from the environment. Always.** No literals for hosts, ports, credentials, or
   business thresholds anywhere in the source tree.
5. **Layers depend inward.** `pricing/` knows nothing about FastAPI. `models/` knows nothing about
   Postgres. Frameworks are edges, not the core.
6. **Synthetic data is first-class.** The demo path is not a degraded path. It is the supported
   path, and it produces data realistic enough that the models learn genuine structure from it.

---

## 3. System context

Who and what touches this system.

```
        ┌──────────────────┐        ┌──────────────────┐
        │  Revenue Manager │        │ External Client  │
        │    (human)       │        │  (PMS / script)  │
        └────────┬─────────┘        └────────┬─────────┘
                 │ views, filters            │ HTTP + JSON
                 │                           │
                 ▼                           ▼
        ┌────────────────────────────────────────────────┐
        │      AI-DRIVEN DYNAMIC HOTEL PRICING SYSTEM    │
        │                                                │
        │   ingest → stream → store → learn → price      │
        └───────┬──────────────────────────────┬─────────┘
                │                              │
                │ (optional, off by default)   │ reads/writes
                ▼                              ▼
        ┌──────────────────┐         ┌──────────────────┐
        │ Competitor sites │         │   PostgreSQL     │
        │ Booking, Expedia │         │  (system of      │
        │  — ADR-004 —     │         │    record)       │
        └──────────────────┘         └──────────────────┘
```

---

## 4. End-to-end pipeline

The canonical ASCII diagram. This is the one that goes in the README.

```
 ┌───────────────────────────────────────────────────────────────────────────┐
 │                          DATA SOURCES                                     │
 │   SyntheticCompetitorGenerator (default)  │  Booking / Expedia (opt-in)   │
 └───────────────────────────────┬───────────────────────────────────────────┘
                                 │  CompetitorPriceEvent
                                 ▼
                    ┌─────────────────────────┐
                    │   ingestion/            │
                    │   scraper_base.py       │   ← one interface, 3 impls
                    │   data_validator.py     │   ← reject bad events at the door
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │   streaming/producer.py │
                    └────────────┬────────────┘
                                 │  JSON, key = hotel_id
                                 ▼
   ╔═════════════════════════════════════════════════════════════════════════╗
   ║                          APACHE KAFKA  (KRaft)                          ║
   ║   hotel.competitor_prices   hotel.booking_events                        ║
   ║   hotel.demand_events       hotel.price_predictions                     ║
   ╚═════════════════════════════════╤═══════════════════════════════════════╝
                                     │
                                     ▼
                    ┌─────────────────────────┐
                    │  streaming/consumer.py  │
                    │  validate → transform   │
                    └────────────┬────────────┘
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
      ┌────────────────────┐         ┌──────────────────────────┐
      │    PostgreSQL      │────────▶│  features/               │
      │  9 tables, indexed │  reads  │  feature_engineering.py  │
      │  UTC everywhere    │         │  feature_store.py        │
      └─────────┬──────────┘         └────────────┬─────────────┘
                │                                 │ 14-feature matrix
                │                                 ▼
                │                    ┌────────────────────────────┐
                │                    │   training/pipeline.py     │
                │                    │                            │
                │                    │   ┌──────────────────────┐ │
                │                    │   │ Prophet              │ │
                │                    │   │ demand time-series   │ │
                │                    │   │ 7 / 14 / 30-day      │ │
                │                    │   └──────────┬───────────┘ │
                │                    │              │             │
                │                    │   ┌──────────▼───────────┐ │
                │                    │   │ GradientBoosting     │ │
                │                    │   │ Regressor            │ │
                │                    │   │ → future_demand      │ │
                │                    │   └──────────┬───────────┘ │
                │                    └──────────────┼─────────────┘
                │                                   │ joblib artifacts
                │                                   ▼
                │                    ┌────────────────────────────┐
                │◀───────────────────│  models/model_registry.py  │
                │  version metadata  │  versions, metrics, hashes │
                │                    └──────────────┬─────────────┘
                │                                   │ loads active model
                │                                   ▼
                │                    ┌────────────────────────────┐
                │                    │   pricing/                 │
                │                    │                            │
                │                    │   demand_engine.py         │
                │                    │        │                   │
                │                    │        ▼                   │
                │                    │   pricing_engine.py        │
                │                    │   base + adjustments       │
                │                    │        │                   │
                │                    │        ▼  raw_price        │
                │                    │   ┌────────────────────┐   │
                │                    │   │   guardrails.py    │   │  ◀── HARD GATE
                │                    │   │  floor / ceiling   │   │      nothing bypasses
                │                    │   │  Δ cap / comp band │   │
                │                    │   └─────────┬──────────┘   │
                │                    └─────────────┼──────────────┘
                │                                  │ final_price
                │                                  ▼
                │                    ┌────────────────────────────┐
                └───────────────────▶│      api/  (FastAPI)       │
                    persist decision │      10 endpoints          │
                                     │      OpenAPI / Swagger     │
                                     └──────┬──────────────┬──────┘
                                            │              │
                          ┌─────────────────┘              └──────────────┐
                          ▼                                               ▼
              ┌───────────────────────┐                      ┌───────────────────────┐
              │  dashboard/ Streamlit │                      │   External Client     │
              │  7 pages, Plotly      │                      │   HTTP consumer       │
              └───────────────────────┘                      └───────────────────────┘

              ┌──────────────────────────────────────────────────────────┐
              │  monitoring/  — cross-cutting, observes every stage above │
              │  data drift · prediction distribution · model perf · logs │
              └──────────────────────────────────────────────────────────┘
```

---

## 5. Deployment topology

Five containers on one user-defined bridge network. Services address each other by **service
name**, never `localhost`.

```
  docker network: pricing-net
  ┌─────────────────────────────────────────────────────────────────────┐
  │                                                                     │
  │  ┌───────────┐    ┌───────────┐   ┌───────────────────────────────┐ │
  │  │  kafka    │    │ postgres  │   │  api                          │ │
  │  │  :9092    │    │  :5432    │   │  uvicorn :8000                │ │
  │  │  KRaft    │    │  pg16     │   │  depends_on: pg+kafka healthy │ │
  │  │  no ZK    │    │  volume   │   │                               │ │
  │  └─────┬─────┘    └─────┬─────┘   └──────────────┬────────────────┘ │
  │        │                │                        │                  │
  │        └────────────────┴────────────┬───────────┘                  │
  │                                      │                              │
  │  ┌──────────────────┐   ┌────────────▼───────────┐                  │
  │  │ streaming-worker │   │  dashboard             │                  │
  │  │ producer+consumer│   │  streamlit :8501       │                  │
  │  │ restart: always  │   │  talks to api by name  │                  │
  │  └──────────────────┘   └────────────────────────┘                  │
  └─────────────────────────────────────────────────────────────────────┘
       host ports published:  8000 → api      8501 → dashboard
                              5432 → postgres (dev convenience only)
```

**Health gating.** `api` and `dashboard` do not start until their dependencies report healthy:

| Service | Healthcheck |
|---|---|
| postgres | `pg_isready -U $POSTGRES_USER -d $POSTGRES_DB` |
| kafka | broker API-versions probe on the internal listener |
| api | `GET /health` returns 200 |
| dashboard | Streamlit's `/_stcore/health` |

**Volumes.** `pgdata` (database), `model-artifacts` (shared read-write between `api` and any
training run, so a model trained via `POST /api/v1/models/train` is immediately loadable).

---

## 6. Package layout and responsibilities

```
AI-DRIVEN DYNAMIC HOTEL PRICING SYSTEM/
│
├── config/          Settings. Single source of truth for every tunable.
├── database/        SQLAlchemy engine, session lifecycle, ORM models, schema init.
├── ingestion/       Competitor data acquisition + validation at the boundary.
├── streaming/       Kafka producer, consumer, topic definitions.   ← see ADR-001
├── features/        Raw rows → model-ready feature matrix. Deterministic, pure.
├── models/          Model classes, persistence, registry. No training loops here.
├── training/        Orchestrates training. Owns the run, writes the registry entry.
├── pricing/         The business core. Zero framework imports.
├── api/             HTTP edge. Thin. Translates HTTP ↔ domain, nothing more.
├── dashboard/       Streamlit UI. Read-only consumer of the API.
├── monitoring/      Logging config, drift detection, model performance tracking.
├── scripts/         Operator entry points (generate, seed, train, create topics).
├── tests/           pytest suite.
├── data/            raw / processed / synthetic. Gitignored except .gitkeep.
└── docs/            This file and its siblings.
```

### Responsibility contracts

| Package | Owns | Must NOT |
|---|---|---|
| `config` | Reading + validating env, exposing typed `Settings` | Import any other project package |
| `database` | Connection pool, session scope, table definitions | Contain business rules |
| `ingestion` | Fetching competitor rates, rejecting malformed events | Write to Postgres directly |
| `streaming` | Kafka wire format, delivery, offset handling | Contain pricing logic |
| `features` | Feature computation, ordering, null policy | Read env directly, hit the network |
| `models` | Fit/predict/save/load, version metadata | Decide *when* to train |
| `training` | Sequencing a training run, evaluation, registration | Serve HTTP |
| `pricing` | Demand→price arithmetic, guardrail enforcement | Import FastAPI, SQLAlchemy, or Kafka |
| `api` | Validation, status codes, serialization, DI wiring | Contain pricing arithmetic |
| `dashboard` | Visualization | Query Postgres directly (goes through API) |
| `monitoring` | Drift + performance metrics, structured logging | Block the request path |

**The dependency rule.** Arrows point one way only:

```
  api ──▶ pricing ──▶ models ──▶ features ──▶ database ──▶ config
   │         │                                    ▲            ▲
   └─────────┴────────────────────────────────────┘            │
                       (all packages) ────────────────────────┘

  dashboard ──▶ api        (HTTP only, never a direct import)
  streaming ──▶ database, ingestion, config
  training  ──▶ models, features, database, config
```

`pricing/` importing `fastapi` is an architecture violation, not a style nit. It is the one rule
worth being pedantic about, because it is what makes the pricing logic unit-testable without
spinning up a web server or a database.

---

## 7. Data flows

Three distinct flows with different latency and consistency requirements.

### Flow A — Ingestion (continuous, async, at-least-once)

```
  generator/scraper
      │  produces CompetitorPriceEvent
      ▼
  data_validator.validate()        ── invalid → log WARN, drop, increment counter
      │  valid
      ▼
  producer.publish(topic, key=hotel_id, value=json)
      │
      ▼
  [ hotel.competitor_prices ]
      │
      ▼
  consumer.poll()
      │
      ├─ deserialize + re-validate  (never trust the topic)
      ├─ upsert → competitor_prices table
      └─ commit offset AFTER successful write   ← at-least-once, not at-most-once
```

Keying by `hotel_id` guarantees per-hotel ordering within a partition, which matters because a
later price for the same hotel/room/date must not be overwritten by an earlier one.

Offsets commit *after* the database write. A crash mid-write replays the event; the write is an
idempotent upsert on `(hotel_id, competitor, room_type, check_in_date, collected_at)`, so replay
is harmless.

### Flow B — Training (batch, on demand)

```
  scripts/train_models.py  or  POST /api/v1/models/train
      │
      ▼
  load historical rows from Postgres
      │
      ▼
  data_validator: schema + range + null checks     ── fail → abort run, record FAILED
      │
      ▼
  feature_engineering.build_matrix()
      │
      ├──────────────────────────┐
      ▼                          ▼
  train Prophet              train GradientBoostingRegressor
  (ds, y = demand)           (14 features → future_demand)
      │                          │
      ▼                          ▼
  evaluate: MAE, RMSE, MAPE  evaluate: MAE, RMSE, R²
      │                          │
      └──────────┬───────────────┘
                 ▼
      joblib.dump → models/artifacts/<version>/
                 ▼
      registry.register(version, metrics, feature_list, dataset_hash, path)
                 ▼
      mark version ACTIVE  (previous stays on disk — rollback is a flag flip)
```

Training is **never** in the request path. `POST /models/train` schedules it and returns
`202 Accepted` with a run id.

### Flow C — Serving (synchronous, low latency)

```
  POST /api/v1/pricing/predict
      │
      ▼
  Pydantic validation        ── invalid → 422 with field-level detail
      │
      ▼
  resolve hotel + room_type  ── unknown → 404
      │
      ▼
  assemble features:
      request payload  +  latest competitor rates from DB  +  derived calendar features
      │                            │
      │                            └── missing → fallback to own trailing rate, confidence ↓
      ▼
  demand_engine:
      Prophet forecast for check_in_date  ─┐
      GBR point prediction                ─┴──▶ blended demand signal
      │
      ▼
  pricing_engine: base × (1+adjustments) → raw_price
      │
      ▼
  guardrails.apply(raw_price, context) → final_price + applied[]
      │
      ▼
  persist to predictions + pricing_decisions
      │
      ├──▶ publish to hotel.price_predictions   (fire-and-forget; failure must not fail the request)
      │
      ▼
  200 + full explainable response
```

---

## 8. Data model

Nine tables. All timestamps `TIMESTAMPTZ`, stored UTC, no exceptions.

```
  hotels ──1:N──▶ rooms ──1:N──▶ ┌─ competitor_prices
    │                            ├─ bookings
    │                            ├─ demand_features
    │                            └─ predictions ──1:1──▶ pricing_decisions
    │
  model_versions ──1:N──▶ training_runs
        │
        └──────────── referenced by ──────────▶ predictions.model_version
```

| Table | Grain | Key indexes |
|---|---|---|
| `hotels` | one row per property | `hotel_id` unique |
| `rooms` | hotel × room_type | `(hotel_id, room_type)` unique |
| `competitor_prices` | hotel × competitor × room × check-in × collected_at | `(hotel_id, check_in_date)`, `(collected_at DESC)` |
| `bookings` | one row per booking event | `(hotel_id, check_in_date)` |
| `demand_features` | hotel × room × date (the feature store) | `(hotel_id, room_type, date)` unique |
| `predictions` | one row per prediction served | `(hotel_id, created_at DESC)` |
| `pricing_decisions` | the audit trail: inputs, adjustments, guardrails | `(prediction_id)` |
| `model_versions` | one row per registered model | `(model_type, is_active)` partial |
| `training_runs` | one row per training attempt incl. failures | `(started_at DESC)` |

`pricing_decisions` is deliberately verbose — it stores every adjustment factor and every
guardrail that fired. When someone asks "why was room 204 priced at ₹7,340 last Tuesday", this
table answers it without re-running anything.

---

## 9. Kafka topics and contracts

| Topic | Producer | Consumer | Key | Retention |
|---|---|---|---|---|
| `hotel.competitor_prices` | ingestion | streaming consumer | `hotel_id` | 7d |
| `hotel.booking_events` | booking simulator | streaming consumer | `hotel_id` | 7d |
| `hotel.demand_events` | feature pipeline | streaming consumer | `hotel_id` | 7d |
| `hotel.price_predictions` | api | dashboard / external | `hotel_id` | 3d |

Event envelope — every message carries these, plus a typed `payload`:

```json
{
  "event_id":   "uuid4",
  "event_type": "competitor_price",
  "version":    1,
  "timestamp":  "2026-09-15T10:30:00Z",
  "payload":    { "...": "type-specific" }
}
```

`version` is present from day one. Adding it later is painful; adding it now costs nothing and
lets consumers reject or adapt to schema changes instead of crashing on them.

Example `competitor_price` payload:

```json
{
  "hotel_id": "H001",
  "competitor": "booking",
  "room_type": "deluxe",
  "check_in_date": "2026-09-15",
  "price": 6200,
  "currency": "INR"
}
```

---

## 10. The pricing algorithm

This is the part an interviewer will interrogate. It is intentionally boring arithmetic — because
boring arithmetic is defensible.

### Composition

```
  base_price          = room's configured base rate (per hotel × room_type)

  demand_adjustment   = f(blended_demand ÷ baseline_demand)      →  e.g. +0.12
  competitor_adjustment = f(base_price vs competitor median)     →  e.g. +0.05
  occupancy_adjustment  = f(occupancy_rate, days_to_checkin)     →  e.g. +0.08
  event_adjustment      = f(local_event_score, is_weekend,
                            is_holiday, season)                  →  e.g. +0.03

  raw_price = base_price × (1 + demand_adj + competitor_adj
                              + occupancy_adj + event_adj)

  final_price, applied[] = guardrails(raw_price, context)
```

Adjustments are **additive within the multiplier**, not chained multiplicatively. Reason: additive
factors are readable ("+12% demand, +5% competitor, so +17% total") and their combined effect is
bounded and predictable. Chained multiplication compounds in ways that surprise people, and
surprising a revenue manager is how a pricing system gets switched off.

Each adjustment is individually clamped before summing, so no single signal can dominate.

### The occupancy × lead-time interaction

The one genuinely non-obvious piece. High occupancy alone does not justify a price rise — it
depends on how much time is left to sell the remaining rooms:

| | far out (30d) | near (3d) |
|---|---|---|
| **high occupancy** | raise hard — demand is real and early | raise modestly — nearly sold out anyway |
| **low occupancy** | hold — plenty of time | discount — use-it-or-lose-it inventory |

Principle 3 ("do not raise price when occupancy is low") is enforced here *and* re-enforced in
guardrails. Belt and braces, deliberately.

### Demand blending

Two models, one number:

```
  blended_demand = w · prophet_forecast + (1 − w) · gbr_prediction
```

They answer different questions and neither subsumes the other. Prophet knows *"mid-September
Tuesdays trend like this"* — calendar structure, seasonality, trend. GBR knows *"given 72%
occupancy, a competitor at ₹6,500, and 14 days lead time, demand looks like this"* — contextual
response to the current situation. Prophet cannot see today's competitor price; GBR cannot see
that next Thursday is Diwali. Blending is not hedging, it is combining complementary signal.

`w` is configurable. When one model is unavailable, weight collapses to the other and `confidence`
in the response drops accordingly.

---

## 11. Guardrails

The last gate. Structurally impossible to bypass: `pricing_engine` returns a `RawPrice` type, and
the only function that converts `RawPrice` → `FinalPrice` is `guardrails.apply()`. The API can
only serialize a `FinalPrice`.

| Rule | Env var | Behavior on breach |
|---|---|---|
| Absolute floor | `MIN_PRICE` | clamp up, record `MIN_PRICE_FLOOR` |
| Absolute ceiling | `MAX_PRICE` | clamp down, record `MAX_PRICE_CEILING` |
| Max daily rise | `MAX_DAILY_CHANGE_PERCENT` | clamp to `prev × (1+pct)`, record |
| Max daily fall | `MAX_DAILY_CHANGE_PERCENT` | clamp to `prev × (1−pct)`, record |
| Competitor upper band | `COMPETITOR_UPPER_BOUND_PERCENT` | clamp to competitor max × bound |
| Competitor lower band | `COMPETITOR_LOWER_BOUND_PERCENT` | clamp to competitor min × bound |
| Low-occupancy rise block | `LOW_OCCUPANCY_THRESHOLD` | if occupancy < threshold, forbid increase |

Order matters and is fixed: **relative rules first** (daily change, competitor band), **absolute
rules last** (floor, ceiling). Absolutes must win — a floor that a relative rule can undercut is
not a floor.

Every rule that fires appends to `guardrails_applied[]` in the response and is logged at WARN with
the before/after values. A guardrail firing constantly is a signal the pricing model needs
retuning; making them visible is how that gets noticed.

---

## 12. API surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + dependency status |
| GET | `/api/v1/hotels` | List hotels |
| GET | `/api/v1/hotels/{hotel_id}` | Hotel detail + rooms |
| POST | `/api/v1/pricing/predict` | **Core endpoint** — recommend a price |
| GET | `/api/v1/pricing/{hotel_id}` | Recent pricing decisions |
| GET | `/api/v1/forecast/{hotel_id}` | Prophet forecast + intervals |
| GET | `/api/v1/models` | Registered versions + metrics |
| POST | `/api/v1/models/train` | Trigger training (202 + run id) |
| GET | `/api/v1/competitors/{hotel_id}` | Competitor rates on record |
| POST | `/api/v1/competitors/events` | Ingest a competitor event |

### Status code policy

| Code | When |
|---|---|
| 200 | Success |
| 202 | Training accepted |
| 400 | Semantically invalid (check-out before check-in) |
| 404 | Unknown hotel / room type |
| 422 | Pydantic schema violation |
| 503 | Model artifact missing, or database unreachable |

`503` for a missing model is the important one. Returning a made-up price when the model is
unavailable would be the single worst failure mode this system could have.

---

## 13. MLOps and the model registry

```
  training run
      │
      ▼
  artifacts/  gb_v20260915_103000/
      ├── model.joblib
      ├── scaler.joblib
      ├── feature_list.json      ← ordered; guards against train/serve skew
      └── metadata.json
      │
      ▼
  model_versions table
      version · model_type · trained_at · dataset_hash
      metrics{} · feature_list[] · artifact_path · is_active
```

Tracked per version: metrics, training timestamp, dataset hash, exact ordered feature list,
artifact path, active flag.

**`feature_list.json` is not bookkeeping.** It is loaded at serve time and the incoming feature
vector is validated against it. Train/serve skew — where feature order or membership silently
drifts between training and inference — is the most common way a working ML system quietly starts
producing wrong numbers. Making the contract explicit and checked turns a silent corruption into a
loud startup error.

**Rollback** is flipping `is_active`. Old artifacts are never deleted by the training pipeline.

MLflow is not in the default Compose stack (ADR-005), but the registry sits behind a
`ModelRegistry` interface, so an `MLflowModelRegistry` implementation drops in without touching
any caller.

---

## 14. Monitoring

| Monitor | Method | Alerts when |
|---|---|---|
| Data drift | PSI on feature distributions vs training baseline | PSI > 0.2 |
| Prediction drift | Rolling mean/std of predicted demand | 3σ excursion |
| Model performance | Predicted vs realized demand, when actuals arrive | MAPE degrades > 20% vs training |
| Data quality | Null rate, range violations per batch | any threshold breach |
| Price stability | Distribution of day-over-day changes | guardrail firing rate spikes |

All logs are structured (JSON in container mode, human-readable locally), carry a correlation id
through the request path, and **never** contain credentials, connection strings, or full
connection URLs.

---

## 15. Architecture decisions

Each of these is a real fork in the road with a real cost on the other side.

### ADR-001 — `streaming/` instead of `kafka/`

**Decision:** the Kafka package is named `streaming/`, not `kafka/`.

**Why:** the `kafka-python` library is imported as `import kafka`. A top-level project directory
named `kafka/` shadows it on `sys.path`. `from kafka import KafkaProducer` would then resolve to
our own package and fail with a confusing `ImportError` that looks like a missing dependency —
including inside Docker, where it is far more annoying to debug. This is a genuine trap, not a
theoretical one.

**Alternatives:** use `confluent-kafka` (imports as `confluent_kafka`, so no collision) — but that
adds a librdkafka C build step. Or keep `kafka/` and manipulate `sys.path` — fragile and
surprising.

**Cost:** one directory name differs from the original spec. The spec permits structural
improvements. Worth it.

### ADR-002 — Kafka in KRaft mode, no ZooKeeper

**Decision:** single-node Kafka in KRaft mode.

**Why:** one fewer container, faster cold start, and ZooKeeper is deprecated for Kafka. For a
single-broker dev deployment there is no upside to running it.

**Cost:** requires Kafka ≥ 3.3 and a `CLUSTER_ID`. Both trivial.

### ADR-003 — Pricing core has zero framework dependencies

**Decision:** `pricing/` imports only stdlib, NumPy, and project config types.

**Why:** it makes the business logic testable in milliseconds without Postgres, Kafka, or an HTTP
server. Since the pricing calculation is the part most likely to be scrutinized and changed, it is
the part that most needs a fast, honest test suite.

**Cost:** explicit DTOs at the boundary rather than passing ORM objects straight through. Small,
recurring, worth paying.

### ADR-004 — Synthetic competitor data is the default path

**Decision:** `SyntheticCompetitorGenerator` is the default. `BookingScraper` and `ExpediaScraper`
exist, implement the same interface, and are **disabled unless explicitly enabled by config**.

**Why:** scraping Booking.com or Expedia at any real rate violates their terms of service, and the
system must be runnable by anyone, immediately, with no credentials and no legal exposure. Making
the synthetic path *default* rather than *fallback* also means the demo path is the well-tested
path.

**Cost:** the generator must produce genuinely structured data — seasonality, weekday effects,
trend, event spikes, autocorrelated competitor moves, and noise. Random uniform numbers would let
the models train and score well while learning nothing, which would make the entire ML layer
theatre. This is the highest-risk piece of Phase 2 and will be treated as such.

#### ADR-004a — amended: a fourth source, `demo_ota`

The decision above left the *scraping pipeline itself* undemonstrated, and in fact
undemonstrable: Booking.com and Expedia both disallow the search paths their scrapers need in
`robots.txt`, and `HttpCompetitorScraper` honours `robots.txt` — so enabling either yields
`ScraperBlocked`. Correct behaviour, and a dead end for showing the pipeline works.

Two options: delete the robots check, or scrape something that genuinely permits it. `demo_ota/`
is the second — a real HTTP server serving real HTML with its own `robots.txt` allowing `/search`.
`DemoOTAScraper` fetches it over the network, parses it with CSS selectors, and raises
`ScraperParseError` when the markup stops matching. Every part of the scraping stack runs for
real; only the operator of the far end differs.

**This forced a distinction the original ADR had conflated.** `INGESTION_ENABLE_REAL_SCRAPERS`
gates *third-party* sites, because the risk it manages is terms of service and legal exposure —
not whether HTTP is involved. Gating `demo_ota` behind it would merge two unrelated risks and put
the safe option behind the frightening switch. `CompetitorSource.is_third_party` now carries that
distinction explicitly instead of leaving it to an if-list.

**Cost:** one more implementation of the interface and one more Compose service. Note also that
the demo site's rate model is deliberately *not* the feature pipeline's event calendar: reusing
`event_score` would make `competitor_avg_price` a deterministic function of `local_event_score`,
and the gradient booster would learn a relationship that exists only because we created it. That
is the `search_demand` 0.83-correlation mistake in a new costume, so the demo site has its own
windows, its own weights, and per-observation noise.

### ADR-005 — File + database registry, MLflow-ready

**Decision:** ship a `ModelRegistry` backed by the filesystem and Postgres. No MLflow container by
default.

**Why:** MLflow adds a server, a backend store, and an artifact store to Compose — meaningful
complexity for a two-model system, working against the one-command-startup requirement. Everything
MLflow would give us here (versions, metrics, artifacts, lineage) is already modeled in
`model_versions` and `training_runs`.

**Cost:** no MLflow UI. Mitigated by the interface boundary — swapping in MLflow later is an
additive change, not a refactor.

### ADR-006 — Prophet's build cost is accepted, and contained

**Decision:** keep Prophet; install it in a builder stage of a multi-stage Docker build.

**Why:** Prophet pulls `cmdstanpy`/`holidays` and needs a compiler on slim images. A naive
single-stage build is slow and produces a fat image. A multi-stage build compiles once and copies
only the installed packages into the runtime layer.

**Cost:** slightly more complex Dockerfile, and a slow *first* build. Layer caching makes every
subsequent build fast.

---

## 16. Phase map

What got built when, and what "done" meant for each. **All 13 phases are
complete**; the table is kept as the build record.

| Phase | Scope | Done when |
|---|---|---|
| 1 | Structure, `config/`, logging, env, deps | `from config import get_settings` works; settings load from `.env` |
| 2 | ORM models, `init_db`, synthetic generator | Seeded DB with months of realistic multi-hotel data |
| 3 | Topics, producer, consumer | Event published → visible in Postgres |
| 4 | Feature engineering + feature store | Deterministic 14-feature matrix, tested |
| 5 | Prophet model + training + eval | Forecast with MAE/RMSE/MAPE, artifact saved |
| 6 | GBR model + training + eval | Trained model, feature importances, artifact saved |
| 7 | Demand engine, pricing engine, guardrails, rules | Price with full breakdown; guardrails provably unbypassable |
| 8 | FastAPI, schemas, routes, error handling | All 10 endpoints live, Swagger correct |
| 9 | Streamlit, 7 pages, Plotly, filters | Dashboard renders against live API |
| 10 | Registry, drift, model monitoring | Metrics computed, warnings logged |
| 11 | Dockerfiles, Compose, healthchecks | `docker compose up --build` works cold |
| 12 | pytest across all layers | Suite green |
| 13 | README + docs | A stranger can run it from the README alone |

Each phase ended with tests run, imports verified and integration checked.

### What the build actually found

Five defects that only surfaced because each phase was verified against running
infrastructure rather than declared done. They are documented in the README under
"Things that went wrong", and each has a regression test:

| Phase | Defect | Why it was invisible |
|---|---|---|
| 5 | Prophet 66% *worse* than predicting the mean | Every structural test passed — shapes, bounds, serialisation |
| 6 | `merge_asof` row misalignment flattened the booking curve | The column kept a plausible range and mean; 10 of 11 tests in the class still passed |
| 8 | Custom-validator 422s became 500s | Pydantic v2 puts the original exception in `ctx`, which is unserialisable |
| 8 | Forecast returned dates two months old | Prophet continues from the *training* window, which excludes the holdout |
| 1 | Plaintext password in `repr(settings.database)` | `computed_field` is included in a model's repr by default |

The lesson encoded in the test suite: structural tests prove a component
*behaves*, and only quality assertions prove it is *worth having*. Both models
now have a test that fails if they stop beating a predict-the-mean baseline.

---

## 17. Known risks

| Risk | Impact | Mitigation |
|---|---|---|
| Synthetic data too simplistic → models learn nothing | Entire ML layer becomes decorative | Explicit realism requirements in Phase 2; verify learned seasonality and non-trivial feature importances |
| Prophet build slowness | Painful iteration | Multi-stage build, layer caching, local venv for dev |
| Kafka startup race | `api` crashes on boot | Compose healthchecks + consumer retry/backoff |
| Train/serve feature skew | Silently wrong prices | `feature_list.json` validated at load |
| Guardrails bypassed by a future code path | Unsafe price emitted | Type-level gate (`RawPrice` → `FinalPrice`) + dedicated tests |
| Timezone drift | Off-by-one-day pricing | `TIMESTAMPTZ` + UTC everywhere, asserted in tests |

---

## 18. Related documents

- `docs/ml_pipeline.md` — feature definitions, model hyperparameters, evaluation protocol
- `docs/api.md` — full endpoint reference with examples
- `docs/deployment.md` — Compose operations, env reference, troubleshooting
- `README.md` — project overview and quickstart
