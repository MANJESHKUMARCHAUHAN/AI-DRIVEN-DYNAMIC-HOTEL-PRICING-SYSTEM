# Technical Overview

The single document for *"explain your system"*. It runs the full arc — the
problem, the data, the models, the backend, deployment and scaling — and points
at the deeper documents rather than repeating them.

Every number here was measured on the running stack. Where something has not
been measured, it says so.

**Contents**

1. [The problem](#1-the-problem)
2. [What the system does](#2-what-the-system-does)
3. [Data flow](#3-data-flow)
4. [The model / AI layer](#4-the-model--ai-layer)
5. [Backend architecture](#5-backend-architecture)
6. [Deployment](#6-deployment)
7. [Scaling](#7-scaling)
8. [Security, observability, testing](#8-security-observability-testing)
9. [Limitations and what comes next](#9-limitations-and-what-comes-next)

---

## 1. The problem

### 1.1 Why hotel pricing is hard

A hotel room is the most perishable product there is. An unsold room-night is
worth exactly **zero** at midnight, and there is no inventory to carry forward —
you cannot sell tonight's empty room tomorrow. Every night, for every room type,
for every future date, somebody has to decide what to charge.

Four things make that decision genuinely difficult:

| | |
|---|---|
| **Demand is uncertain** | You are pricing a night that has not happened yet, against bookings that have not arrived |
| **The answer changes with time** | The same night is a different problem 60 days out and 2 days out. On check-in day the rooms already on the books essentially *are* the answer |
| **Competitors move** | A rate that was correct this morning can be conspicuous by this afternoon |
| **Mistakes are asymmetric** | Price too high and the room goes empty — you lose 100% of that night's revenue. Price too low and you sold a room you could have sold anyway, at a discount |

### 1.2 The constraint that shaped everything

There is a fifth requirement, and it is the one that drove the architecture:

> **A price a revenue manager cannot interrogate is a price they will override —
> and an overridden pricing system is a switched-off pricing system.**

This is not a nice-to-have. Revenue managers are accountable for RevPAR and they
have a manual override. A system that says *"the model decided ₹9,400"* gets
turned off within a month. A system that says *"base ₹7,936, +18% occupancy,
+14% event, then clipped by the 15% daily-change cap"* gets trusted, argued
with, and used.

Everything downstream follows from that: named adjustments instead of a learned
price, a deterministic engine instead of an end-to-end model, guardrails that
cannot be bypassed, and a stored breakdown for every decision ever made.

### 1.3 Problem statement

> Given a hotel, a room type and a future date, recommend a nightly rate that
> maximises expected RevPAR — using observed competitor rates, the booking curve
> to date, and calendar and event structure — such that:
>
> 1. every recommendation is reproducible and fully explained,
> 2. no recommendation can violate business limits, whatever the model says,
> 3. the recommendation returns fast enough to price a full grid interactively,
> 4. the system degrades rather than fails when any input is missing.

### 1.4 Explicit non-goals

Stating these matters, because each one is a defensible scope decision rather
than an omission:

- **Not price elasticity.** The engine prices *to* forecast demand; it does not
  model how demand responds to the price it sets. Closing that loop needs
  experimentation or an instrumental-variables design ([§9](#9-limitations-and-what-comes-next)).
- **Not multi-tenant SaaS.** Single-operator deployment.
- **Not length-of-stay or channel pricing.** Everything is a room-night through
  one channel.
- **Not live third-party scraping as the default.** Booking.com and Expedia both
  disallow the paths their scrapers would need; see [§3.1](#31-competitor-ingestion).

---

## 2. What the system does

Given `(hotel, room type, date)` it returns a rate in **~28 ms**, together with
the five adjustments that produced it, every guardrail that changed it, the
demand forecast behind it, and a confidence score.

**Measured on the shipped dataset** — 8 hotels, 6 Indian cities, 4 room types,
365 days:

| | Value |
|---|---|
| Gradient Boosting MAE | **0.0644** vs baseline **0.1430** — 55% better |
| Gradient Boosting R² | 0.765 (baseline 0.000 by construction) |
| Prophet MAE | 0.0879 — 39% better than baseline |
| Prophet interval coverage | 78% against a nominal 80% — calibrated |
| Pricing latency | p50 **28.4 ms**, p95 31.9, p99 33.0 |
| Cold first request | ~2,000 ms (lazy model load — see [§7.4](#74-cold-start)) |

Every metric is quoted against a **predict-the-mean baseline** on a
**chronological holdout** — the most recent 60 days, never shuffled. A random
split on time-series data leaks the future into training.

---

## 3. Data flow

```
 demo OTA          scraper          Kafka          consumer        PostgreSQL
 (real site)  ──►  robots.txt  ──►  4 topics  ──►  idempotent  ──►  9 tables
                   CSS parse        at-least-once   writes
                                                                        │
                                                                        ▼
                                                              feature pipeline
                                                              30 features
                                                              leakage-free
                                                                        │
                                            ┌───────────────────────────┤
                                            ▼                           ▼
                                     Prophet                     Gradient Boosting
                                (calendar structure)          (current situation)
                                            └─────────────┬─────────────┘
                                                          ▼
                                             blended demand + confidence
                                                          ▼
                                                  pricing engine
                                             5 clamped adjustments
                                                          ▼
                                                    guardrails
                                                          ▼
                                          FastAPI  ──►  dashboard / audit trail
```

### 3.1 Competitor ingestion

One interface, `CompetitorScraper`, four implementations. Everything downstream
depends only on the interface, so swapping in a licensed rate feed is one
subclass and one environment variable.

| Source | Network | Default | Notes |
|---|---|---|---|
| `synthetic` | no | native | Deterministic offline generator |
| `demo_ota` | **yes** | **Docker** | Real scraping against a bundled site |
| `booking` / `expedia` | yes | disabled | Third-party, opt-in twice over |

**Why the default scraping target is a site we ship.** Booking.com and Expedia
both disallow the search paths their scrapers need in `robots.txt`, and
`HttpCompetitorScraper` reads and obeys `robots.txt` — so pointing at them
correctly produces `ScraperBlocked` and collects nothing. Two options: delete the
robots check, or scrape something that genuinely grants permission. `demo_ota/`
is a real HTTP server with real HTML and its own permissive `robots.txt`. The
scraper opens a socket, fetches `/robots.txt`, parses it, respects the rate
limiter, requests a page, gets a real status code, parses real markup with CSS
selectors, and raises `ScraperParseError` when the markup stops matching. Nothing
about the scraping is simulated — only the operator of the far end differs.

`parse()` **raises** rather than returning an empty list on a markup change,
because to a pricing engine *"the competitive set published no rates"* and
*"our parser broke"* must never look the same. `INGESTION_DEMO_OTA_LAYOUT=v2`
serves redesigned markup on demand so that failure is reproducible.

### 3.2 Streaming

Four Kafka topics, one envelope, **at-least-once** delivery made safe by
idempotent writes:

1. Producer wraps each observation in an envelope carrying a `uuid4` `event_id`
2. Consumer reads a batch — **auto-commit is off**, which is the whole design
3. Handler writes with `INSERT … ON CONFLICT DO NOTHING` on the unique `event_id`
4. Offset is committed **after** the database commit

Committing the offset first would be at-most-once: a crash silently loses
competitor observations and the engine keeps quoting a market it can no longer
see. After a consumer rebalance the same message legitimately arrives twice;
without the unique key that is a duplicated observation quietly dragging the
competitor average around — a data-quality bug with no error and no log line.

### 3.3 Storage grain — the decision everything rests on

Bookings are stored at **room-night grain on the pickup grid**: one row per
`(booking date × stay date)`.

That single choice is what makes leakage-free features possible. Because the
booking curve is preserved rather than aggregated away, demand for any night can
be reconstructed **as of any earlier point in time** — which is exactly what a
feature computed at pricing time must do.

Current volumes:

| Table | Rows |
|---|---|
| `bookings` | 224,182 |
| `competitor_prices` | 98,632 |
| `demand_features` | 11,680 |
| `rooms` / `hotels` | 32 / 8 |

Whole database: **76 MB** (37 MB bookings, 25 MB competitor prices, including
indexes).

### 3.4 Feature engineering

30 features in four groups:

| Group | Examples |
|---|---|
| Booking curve | On-the-books occupancy at horizon, pickup pace, days to check-in |
| Calendar | Day of week, month, season one-hots, holiday and local-event scores |
| Market | Competitor avg/min/max, price-to-competitor ratio, availability pressure |
| Interaction | `occupancy × lead_time` — the third most important feature |

**Two guarantees, both enforced by code:**

- **Leakage** — a counterfactual test changes only *future* bookings for a stay
  date, rebuilds the features, and asserts nothing as-of-today moved. It tests
  the property, not the implementation.
- **Train/serve skew** — the training matrix builder and the serving-row builder
  both call the same `FeatureBuilder._add_derived_features`. Two implementations
  of "what are the features" is how skew happens.

Full detail: [`ml_pipeline.md`](ml_pipeline.md).

---

## 4. The model / AI layer

There are **two** distinct AI stories here, and conflating them is the most
common misreading of this project.

### 4.1 The demand models — these make decisions

Two models, because they answer different questions.

**Prophet — calendar structure.** Weekly shape, trend, holiday effects, fitted
per `(hotel, room type)` — 32 series. Produces an 80% uncertainty interval.
Knows nothing about today: no occupancy, no competitor rates.

**Gradient Boosting — the current situation.** 30 features, 247 trees selected by
early stopping. Knows the state right now; has no notion of a seasonal cycle.

They are **blended**, both scored on the *same* chronological holdout — never
Prophet's internal backtest, because otherwise the blend weight between them has
no basis.

**A measured decision that must not be tidied away:** yearly seasonality and
holiday regressors are **disabled below 730 days of history**. With ~365 days the
Fourier term is unidentifiable, so Prophet latches onto noise and extrapolates it
confidently — measured MAE 0.246 against a 0.148 baseline, R² −6.5. *Worse than
predicting the mean, while passing every structural test.* Disabling them:
MAE 0.088, interval coverage 0.82 against a nominal 0.80.

**What the model learned** (permutation importance on the holdout, not
scikit-learn's impurity measure):

| # | Feature | Importance |
|---|---|---|
| 1 | `occupancy_rate` | 0.0298 |
| 2 | `days_to_checkin` | 0.0147 |
| 3 | `occupancy_x_lead` | 0.0080 |

On-the-books occupancy first, lead time second, their interaction third — the
revenue-management story the whole system is built around, found in the data
rather than imposed.

**Error by horizon has the shape a correct model should have:** MAE 0.045 at 0
days (R² 0.884), 0.081 at 60 days. Most accurate near check-in because the
booking curve has already resolved most of the answer. A model equally accurate
at every horizon would be suspicious, not impressive.

### 4.2 The pricing engine — deterministic, not learned

The models produce a **demand estimate**; they do not produce a price. The price
comes from a deterministic engine:

```
base rate
  × (1 + Σ five individually-clamped adjustments)      demand, occupancy,
  × confidence (floor 0.5)                             competitor, event, season
  → RawPrice
  → guardrails.apply()  → FinalPrice
```

Adjustments are **additive inside one multiplier** and each is individually
clamped — never chained multiplicatively, because chaining lets three moderate
signals compound into an extreme move nobody intended.

**Confidence scaling is the subtle part.** Prophet's interval width becomes a
confidence score, and the entire multiplier is scaled by it with a floor of 0.5.
An unsure model moves the price only half as far from base as a certain one — it
degrades towards doing nothing rather than making a confident mistake.

**Guardrails run in a fixed order:** low-occupancy block → daily-change cap →
competitor bands → per-room floor/ceiling → global MIN/MAX. Relative rules before
absolute ones, because a floor a percentage rule can undercut afterwards is not a
floor.

**The type gate.** The engine returns `RawPrice`. Only `guardrails.apply()` can
construct a `FinalPrice` — it holds a module-private token — and the API will
only serialise a `FinalPrice`. So *"guardrails always run"* is a property of the
type system, not a convention somebody has to remember.

### 4.3 The LLM agent — this one explains, it never decides

Separate, optional, and deliberately bounded. `anthropic` is not a runtime
dependency; delete `ai_agent/` and the pricing system is unchanged.

**Where it does not go:** not the pricing calculation (reproducibility,
auditability, a 28 ms budget — an LLM gives up all three), not the demand
forecast (Prophet and the booster are 39% and 55% better than baseline and run in
milliseconds), not anywhere in the request path.

**The trap worth naming:** *let the model pick the multipliers and keep the
guardrails* is worse than either extreme. Guardrails clamp the **output**; they
cannot detect that the reasoning was wrong. A model that hallucinates "Diwali is
next week" produces a price inside every band, confidently justified, and
completely false. **Guardrails bound damage. They do not create correctness.**

**Where it earns its keep:** every decision is already stored with its full
arithmetic — what was missing is *language*. Explaining a decision from the
stored breakdown, triaging correlated monitoring alerts, and (not yet built)
extracting events from unstructured text, which is the only use that would
improve the *model* rather than the operator experience.

**How "it cannot write" is enforced:** a fixed `(method, path)` allowlist checked
before any request leaves, `persist=False` hardcoded rather than exposed as a
parameter, **and** the agent is issued the read-scoped API key — so the
restriction is a fact about the network, not a claim about the code.

Full argument: [`ai_agent_design.md`](ai_agent_design.md).

---

## 5. Backend architecture

### 5.1 Modular monolith, enforced

One codebase, six processes from one image. **Deliberately not microservices:**
at this scale splitting would put network hops inside a 28 ms pricing path, turn
the `pricing_decisions` audit trail into a distributed transaction, and multiply
the operational surface by six for no benefit anybody could name.

But "modular" is a claim, and an unenforced claim decays.
`tests/test_architecture.py` reads the import graph and asserts **24 boundaries**:

- `pricing/` imports **no framework** — no FastAPI, SQLAlchemy, Kafka, HTTP client
- `dashboard/` never opens a database connection
- `ai_agent/` cannot reach the database or the pricing engine; nothing imports it
- `anthropic` appears in exactly one package
- `demo_ota/` imports nothing from the application
- dependencies point inward: `domain` ← `config` ← `database` ← `features`/`models` ← `pricing` ← `api`

> Writing these found a real defect: `pricing/` imported `database.models` for two
> enums — harmless at runtime, but the graph said pricing depended on
> persistence. The fix was to move the shared vocabulary down into `domain/`
> rather than weaken the rule.

### 5.2 Layers

| Layer | Packages | Rule |
|---|---|---|
| Vocabulary | `domain/` | Imports nothing |
| Configuration | `config/` | One settings tree, entirely from the environment |
| Persistence | `database/` | 9 tables, portable enums so tests run on SQLite |
| Acquisition | `ingestion/`, `streaming/`, `demo_ota/` | One interface, four sources |
| Computation | `features/`, `models/`, `training/` | Leakage-free, versioned artifacts |
| **Business logic** | `pricing/` | **Framework-free.** Numbers in, numbers out |
| Edge | `api/`, `dashboard/` | 12 endpoints; the dashboard is a pure API consumer |
| Cross-cutting | `monitoring/`, `ai_agent/` | Logging, metrics, health; the agent is a leaf |

### 5.3 API

12 endpoints, OpenAPI at `/docs`. Two properties worth calling out:

**The registry loads lazily and never fatally.** A missing model artifact is a
degraded service on the historical fallback, not a failed boot — *an API that
refuses to start before a training job has run cannot be deployed before it is
trained.*

**The dashboard holds no business logic and no database connection.** Every
figure it renders comes from an HTTP call, so there is one implementation of
"what is the right price" rather than two that drift apart. It also makes a
rendering page a genuine integration test of the endpoint behind it.

Reference: [`api.md`](api.md).

### 5.4 Design principles

1. **Transparent over clever** — every adjustment is named, logged and
   individually inspectable.
2. **Guardrails are not advisory** — exactly one function produces a final price,
   and it always runs them.
3. **Degrade, don't die** — Kafka down → the API still serves. Competitor data
   missing → fall back to the hotel's own recent rate *and lower the confidence
   score*. Model artifact missing → historical fallback, reported on `/health`.
4. **Configuration comes from the environment** — no literals for hosts, ports,
   credentials or business thresholds anywhere in the source tree.
5. **Nothing non-deterministic is load-bearing** — the LLM agent is optional by
   construction, and a test pins the SDK-absent path.

---

## 6. Deployment

### 6.1 Topology

Eight Compose services; `api`, `dashboard`, `consumer`, `producer`, `init` and
`demo-ota` all run **one image** with different commands — they share every
dependency, so separate images would mean six copies of Prophet, NumPy and
scikit-learn.

| Service | Port | Role |
|---|---|---|
| `postgres` | 55432 | Non-standard host port; 5432 is usually taken |
| `kafka` | 29092 | KRaft mode, no ZooKeeper |
| `demo-ota` | 8900 | The site the scraper scrapes |
| `init` | — | Runs once, exits 0. **Idempotent** |
| `api` | 8000 | FastAPI |
| `producer` / `consumer` | — | Scrape → Kafka → Postgres |
| `dashboard` | 8501 | Streamlit, 9 pages |

### 6.2 Image

Multi-stage. Prophet and psycopg2 need a C toolchain, so they compile in a
builder stage and only the finished virtualenv is copied forward — `gcc`, `g++`
and the header packages never reach the runtime image. Runs as a non-root user.

Dependencies come from `pyproject.toml`; only `[project.dependencies]` is
installed, so the runtime image contains **no** pytest, ruff or `anthropic` — and
CI asserts that rather than assuming it.

### 6.3 Bootstrap is idempotent, and that was learned the hard way

`init` runs schema → topics → data → seed → features → train, and **skips
whatever is already done**. The first version was a shell chain that re-ran
everything on every `up`, silently destroying all pricing history and every
streamed competitor rate. `docker compose up` must converge, not reset.

Two more Compose lessons in the same category:

- **`bash -c`, never `bash -lc`** — a login shell rebuilds `PATH` from
  `/etc/profile` and drops `/opt/venv/bin`, so everything fails on
  `import sqlalchemy`.
- **Workers set `healthcheck: disable: true`** — they inherit the image's API
  healthcheck and would otherwise report "unhealthy" forever while working
  perfectly. A false unhealthy trains operators to ignore the column.

Operations detail: [`deployment.md`](deployment.md).

---

## 7. Scaling

The honest version. The shipped scale is 8 hotels; this is what actually breaks
as that grows, in the order it breaks.

### 7.1 What "scale" means here

The unit of work is a **priced room-night**. A full daily repricing run is:

```
hotels × room types × pricing horizon
```

| Estate | Room-nights per full run |
|---|---|
| 8 (shipped) | 8 × 4 × 365 = **11,680** |
| 100 | 146,000 |
| 1,000 | 1,460,000 |

### 7.2 Measured throughput and the first bottleneck

Single API process, warm: **p50 28.4 ms**, p95 31.9, p99 33.0 → **~35 priced
room-nights per second**.

| Estate | Full run, single process | Verdict |
|---|---|---|
| 8 | ~5.5 min | Fine |
| 100 | ~70 min | Needs parallelism |
| 1,000 | **~11.6 hours** | Broken |

**The first thing to break is batch repricing, not interactive requests.**
Interactive load is a revenue manager clicking a dashboard — a handful of
requests per second, which one process serves comfortably at any estate size.

The fix is not a faster model. It is a **batch path** that scores a whole
`(hotel × room type × date)` grid in one vectorised call instead of one HTTP
request per room-night. The engine is already framework-free and takes plain
numbers, so this is a new caller rather than a rewrite — which is one of the
things that boundary buys.

### 7.3 Training

| Step | Measured | Scaling |
|---|---|---|
| Gradient Boosting | 4.5 s (11,648 rows, 247 trees) | Sub-linear in rows |
| Prophet | 15.9 s for 32 series → **496 ms/series** | **Linear in series** |
| Total | 20.7 s | |

Prophet dominates and scales linearly with `hotels × room types`:

| Estate | Series | Prophet fit |
|---|---|---|
| 8 | 32 | 16 s |
| 100 | 400 | ~3.3 min |
| 1,000 | 4,000 | **~33 min** |

Every series is independent, so this is **embarrassingly parallel** — the ceiling
is cores, not architecture. Beyond a few thousand series the real answer is a
hierarchical or pooled model rather than one Prophet per series.

### 7.4 Cold start

The first pricing request after a boot takes **~2,000 ms** against ~28 ms warm —
lazy model loading (a 32-series Prophet bundle plus the booster).

This is deliberate: the registry loads lazily and never fatally, which is what
lets the API boot before it has ever been trained. But it means the first request
after every deploy pays for it. If that becomes user-visible, the fix is a
readiness probe that prices one throwaway room-night at startup, so the load
happens before traffic arrives rather than during it. **Not currently
implemented.**

### 7.5 Ingestion

Deliberately the slowest thing in the system:

```
requests = hotels × room types × horizons
wall clock ≈ requests × INGESTION_RATE_LIMIT_SECONDS
```

At the shipped 8 hotels, 4 types, 5 horizons and 1 s spacing: 160 requests ≈
**2.7 minutes** per sweep. At 100 hotels: 2,000 requests ≈ 33 minutes.

Politeness is the constraint, not throughput, and it should stay that way. The
scaling answer is not a shorter delay — it is a **licensed rate feed**, which
returns bulk JSON and drops in as one subclass behind the existing interface.
That is precisely what the `CompetitorScraper` abstraction exists for.

`POST /ingestion/run` guards on **estimated duration**, not request count: the
same 96 requests are instant with the limiter off and eight minutes at five
seconds apart, and it is the second that hangs a browser.

### 7.6 Database

76 MB at 8 hotels, dominated by `bookings` (224k rows, 37 MB). Bookings grow as
`hotels × room types × stay dates × booking dates` — the pickup grid is quadratic
in the horizon, which is the price paid for leakage-free features.

Extrapolating linearly in hotels: ~950 MB at 100 hotels, ~9.5 GB at 1,000.
Comfortable for a single managed Postgres. The lookup indexes
(`hotel_id, check_in_date` and the composite `hotel_id, room_type, check_in_date,
collected_at`) are the ones that matter; beyond ~10 GB, partitioning
`competitor_prices` by month is the obvious next step. **Not measured beyond the
current dataset** — the figures above are extrapolation, not benchmark.

### 7.7 Horizontal scaling

The API is **stateless** — session per request, models loaded read-only from a
shared volume — so replicas scale linearly behind a load balancer. Two caveats,
both real:

- **The rate limiter is in-process.** Each replica holds its own counter, so the
  effective limit multiplies by replica count. Enforce the real quota at the load
  balancer.
- **Each replica loads its own model bundle** — memory scales with replicas, and
  each pays the ~2 s cold start.

Kafka scales by partition; the consumer group rebalances automatically, and the
`event_id` idempotency key is what makes rebalancing safe.

### 7.8 Summary

| Component | Breaks at | Fix |
|---|---|---|
| **Batch repricing** | ~100 hotels | Vectorised batch path (engine already supports it) |
| Prophet training | ~1,000 hotels | Parallelise; then a hierarchical model |
| Ingestion | ~100 hotels | Licensed feed behind the existing interface |
| Database | ~1,000 hotels | Partition `competitor_prices` by month |
| Interactive API | not the bottleneck | Stateless replicas |

---

## 8. Security, observability, testing

### 8.1 Security

**Two scopes.** `read` reaches every `GET` plus simulations (`persist=false`);
`write` additionally reaches persisted decisions, competitor submissions,
ingestion runs and retraining.

The split is not decoration: **the AI agent is issued the read key**, so "the
agent cannot write" becomes a fact about the network rather than a claim about
its allowlist.

`ENVIRONMENT=production` **refuses to boot** without auth enabled, with either
key still at its shipped value, with identical read and write keys, with
`DEBUG=true`, with a default database password, or with `CORS=*`. Every check
fails at startup — a misconfiguration that only appears under traffic is a
misconfiguration that ships.

Also: rate limiting, security headers on every response including error paths,
and `SecretStr` throughout with a logging redaction filter.

### 8.2 Observability

`GET /metrics` in Prometheus format:

| Metric | Use |
|---|---|
| `http_request_duration_seconds` | Latency, bucketed for a ~28 ms budget |
| `http_requests_total` | Error rate by route and status |
| **`pricing_guardrail_hits_total`** | **The one to alert on** |
| `ingestion_observations_total` | Feed liveness by source |
| `model_version_info` | Which artifacts are serving |

Guardrail pressure is the interesting series. One firing occasionally is the
system working; one firing on most decisions means the model wants prices the
business will not allow — a retuning signal invisible unless somebody counts.

Latency is labelled by **route template**, never resolved path, so
`/hotels/{hotel_id}` is one series rather than one per hotel.

Separately, `make monitor` runs 9 data-quality checks plus PSI drift nightly. It
emits its own caveat: with under two years of history, comparing a 30-day window
against a longer reference **cannot separate drift from season**. A system that
cries wolf every autumn is a system people mute.

### 8.3 Testing

**893 tests, ~55 seconds, and none of them need a running service.**

| Layer | What |
|---|---|
| Unit | Pricing rules, guardrails, metrics — pure functions, one line each |
| Contract | Event envelopes, API schemas, OpenAPI surface asserted as an exact set |
| Architecture | 24 import-graph boundary tests |
| Integration | In-memory SQLite; the scraper against a **real server** on a free port |
| End-to-end | Every dashboard page executed with Streamlit's `AppTest` |

The scraper tests start a real HTTP server rather than mocking `httpx`, because
mocking would exercise `parse()` and skip robots.txt, status-code mapping,
connection reuse and the rate limiter — between them most of what a scraper is.

CI runs boundaries (~1 s, no dependencies), lint, tests on Python 3.10 and 3.11,
and a Docker build that asserts the runtime image contains no dev or agent
extras.

---

## 9. Limitations and what comes next

### Known limitations

- **One year of synthetic history.** Not enough to identify yearly seasonality,
  which is why Prophet's yearly term is off.
- **One snapshot per stay date.** The feature store holds a single horizon per
  night; a full panel would give more rows but heavily correlated ones.
- **`POST /models/train` is synchronous.** Honest about it in the docstring; at
  scale it becomes an enqueue returning 202.
- **Auth is off by default** — not absent. Production refuses to boot without it.
- **The rate limiter is in-process.**
- **The demo OTA is not market data.** The site and the scraping are real; the
  rates are generated. The claim is "the ingestion path works end to end", not
  "these are the prices in Goa".
- **Single Kafka broker, replication factor 1.**
- **No batch pricing path** — see [§7.2](#72-measured-throughput-and-the-first-bottleneck).

### Next, in order of value

1. **Two years of history.** The single change that most improves the forecast —
   it re-enables yearly seasonality.
2. **A vectorised batch pricing path.** The first real scaling ceiling.
3. **Price elasticity.** The engine prices *to* forecast demand; it does not model
   how demand responds to the price it sets. Needs experimentation or an
   instrumental-variables approach — saying so is more honest than pretending the
   current design closes the loop.
4. **The event-extraction agent.** The event calendar is 19 hand-curated records;
   this is the one LLM use that improves the model rather than the operator
   experience.
5. **Async training** with a task queue, triggered by the drift monitor rather
   than a human.

---

## Where to go deeper

| Topic | Document |
|---|---|
| Design decisions and ADRs | [`architecture.md`](architecture.md) |
| Flow-by-flow walkthrough | [`technical_deep_dive.md`](technical_deep_dive.md) |
| Features, models, evaluation | [`ml_pipeline.md`](ml_pipeline.md) |
| Endpoint reference | [`api.md`](api.md) |
| Compose operations | [`deployment.md`](deployment.md) |
| The LLM agent | [`ai_agent_design.md`](ai_agent_design.md) |
| Explaining this in an interview | [`interview_guide.md`](interview_guide.md) |
| Running it | [README → Setup](../README.md#setup) |
