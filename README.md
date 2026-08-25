# AI-Driven Dynamic Hotel Pricing System

An end-to-end revenue-management platform: it collects competitor rates through
Kafka, engineers leakage-free features from the booking curve, forecasts demand
with Prophet and Gradient Boosting, converts that into a room rate through a
transparent pricing engine, enforces business guardrails that cannot be
bypassed, serves it over FastAPI and explains it in a Streamlit dashboard.

```
competitor data ──► Kafka ──► consumer ──► PostgreSQL ──► features
                                                             │
                                     ┌───────────────────────┴──────────┐
                                     ▼                                  ▼
                              Prophet forecast              Gradient Boosting
                            (calendar structure)          (current situation)
                                     └───────────┬──────────────────────┘
                                                 ▼
                                        blended demand + confidence
                                                 ▼
                                          pricing engine
                                    (5 clamped adjustments)
                                                 ▼
                                       business guardrails
                                                 ▼
                              FastAPI  ──►  Streamlit dashboard
```

**893 tests. ~33,600 lines across 89 modules. Every number below was measured on
this machine, not estimated.**

---

## Table of contents

**Getting it running**

- [Setup](#setup) — pick a path
- [**Docker — everything in one command**](#docker--everything-in-one-command)
- [**Without Docker — running natively**](#without-docker--running-natively)
- [Verify it actually works](#verify-it-actually-works)
- [Optional: the AI Agent page](#optional-the-ai-agent-page)
- [Troubleshooting](#troubleshooting)

**How it works**

- [What it does](#what-it-does)
- [Results](#results)
- [Architecture](#architecture)
- [The data](#the-data)
- [Competitor ingestion](#competitor-ingestion) — the scraping pipeline
- [The Kafka pipeline](#the-kafka-pipeline)
- [Feature engineering](#feature-engineering)
- [The models](#the-models)
- [The pricing algorithm](#the-pricing-algorithm)
- [Guardrails](#guardrails)
- [The API](#the-api)
- [**The dashboard — page by page**](#the-dashboard--page-by-page)
- [**Where the AI agent fits**](#where-the-ai-agent-fits)
- [MLOps](#mlops)
- [Monitoring](#monitoring)

**Running it for real**

- [**Production**](#production) — auth, scopes, metrics, enforced boundaries

**Everything else**

- [**Technical overview**](docs/technical_overview.md) — problem → data flow → AI → architecture → scaling, in one read
- [Testing](#testing)
- [Configuration](#configuration)
- [Things that went wrong](#things-that-went-wrong)
- [Known limitations](#known-limitations)
- [Future improvements](#future-improvements)
- [**Explaining this project**](docs/interview_guide.md) — the interview walkthrough

---

## What it does

A hotel room is the most perishable product there is: an unsold room-night is
worth exactly zero at midnight, and there is no inventory to carry forward. The
job of a pricing system is to decide, every day, for every room type and every
future night, what to charge — knowing that demand is uncertain, competitors are
moving, and the answer changes as check-in approaches.

This system does that. Given a hotel, a room type and a date it returns a rate,
the five adjustments that produced it, every guardrail that changed it, the
demand forecast behind it and a confidence score — in about 30 milliseconds.

It is built to be **explainable**, because a price a revenue manager cannot
interrogate is a price they will override, and an overridden pricing system is a
switched-off pricing system.

---

## Results

Both models are evaluated on the **same chronological holdout** — the most
recent 60 days, never shuffled — against a predict-the-mean baseline.

| Model | MAE | RMSE | MAPE | R² | vs baseline |
|---|---|---|---|---|---|
| Baseline (predict the mean) | 0.1430 | 0.1729 | 31.6% | 0.000 | — |
| **Gradient Boosting** | **0.0644** | **0.0838** | **13.6%** | **0.765** | **55% better** |
| Prophet | 0.0879 | 0.1144 | 17.5% | 0.562 | 39% better |

Prophet's 80% uncertainty interval covers **78%** of outcomes — calibrated, not
decorative, which matters because the pricing engine turns interval width into
its confidence score.

Accuracy by lead time — the shape is the story. The model is confident near
check-in, where the booking curve has already resolved most of the answer, and
appropriately uncertain far out:

| Days to check-in | 0 | 3 | 7 | 14 | 30 | 60 |
|---|---|---|---|---|---|---|
| **MAE** | 0.045 | 0.054 | 0.064 | 0.085 | 0.077 | 0.081 |
| **R²** | 0.884 | 0.841 | 0.742 | 0.629 | 0.692 | 0.647 |
| Baseline MAE | 0.144 | 0.149 | 0.131 | 0.144 | 0.151 | 0.144 |

Top features by **permutation importance on the holdout** (not scikit-learn's
impurity measure, which favours high-cardinality columns):

| # | Feature | Importance |
|---|---|---|
| 1 | `occupancy_rate` | 0.0298 |
| 2 | `days_to_checkin` | 0.0147 |
| 3 | `occupancy_x_lead` | 0.0080 |
| 4 | `search_demand` | 0.0073 |
| 5 | `lead_time` | 0.0071 |

On-the-books occupancy first, lead time second, and their interaction third.
That is the revenue-management story the whole system is built around, and it is
what the model actually learned.

---

## Setup

Two paths, both complete and copy-pasteable.

- **[Docker](#docker--everything-in-one-command)** — nothing to install but Docker.
  Brings up all 8 services including Postgres and Kafka. **Use this on a new machine.**
- **[Without Docker](#without-docker--running-natively)** — you supply Postgres and
  Kafka. Use this to develop on the code.

Both end at the same place: dashboard on 8501, API on 8000, competitor site on 8900.

---

## Docker — everything in one command

### Prerequisites

Only Docker. Verify:

```bash
docker --version           # 20.10+
docker compose version     # v2.x  (note: "compose", not "docker-compose")
```

Needs roughly **3 GB** of disk for images and **4 GB** of free RAM.

### The complete sequence

```bash
# ---------------------------------------------------------------------------
# 1. Get the code
# ---------------------------------------------------------------------------
git clone <your-repo-url> "AI-DRIVEN DYNAMIC HOTEL PRICING SYSTEM"
cd "AI-DRIVEN DYNAMIC HOTEL PRICING SYSTEM"

# ---------------------------------------------------------------------------
# 2. Configure. Copy and DO NOT EDIT -- every default works under Docker.
# ---------------------------------------------------------------------------
cp .env.example .env

# ---------------------------------------------------------------------------
# 3. ONLY if `docker compose up` cannot pull images (see Troubleshooting).
#    Builds Kafka locally from the Apache tarball, tagged as Compose expects.
# ---------------------------------------------------------------------------
# make kafka-image

# ---------------------------------------------------------------------------
# 4. Build and start everything. First run: 5-10 minutes.
# ---------------------------------------------------------------------------
docker compose up --build -d

# ---------------------------------------------------------------------------
# 5. Watch the one-shot initialiser:
#    schema -> topics -> data -> seed -> features -> train
# ---------------------------------------------------------------------------
docker compose logs -f init          # Ctrl-C once it prints BOOTSTRAP COMPLETE

# ---------------------------------------------------------------------------
# 6. Confirm every service is up
# ---------------------------------------------------------------------------
docker compose ps
```

`docker compose ps` should show:

```
NAME                STATUS
pricing-postgres    Up (healthy)
pricing-kafka       Up (healthy)
pricing-demo-ota    Up (healthy)
pricing-init        Exited (0)        <- correct: it runs once and stops
pricing-api         Up (healthy)
pricing-consumer    Up
pricing-producer    Up
pricing-dashboard   Up (healthy)
```

### What each service is

| Service | Port | Job |
|---|---|---|
| `postgres` | 55432 | Database. Non-standard host port because 5432 is usually taken. |
| `kafka` | 29092 | Broker, KRaft mode, no ZooKeeper |
| `demo-ota` | 8900 | The competitor website the scraper scrapes |
| `init` | — | Runs once and exits 0. **Idempotent** — safe on every `up`. |
| `api` | 8000 | FastAPI + Swagger |
| `producer` | — | Scrapes demo-ota into Kafka, sweep after sweep |
| `consumer` | — | Kafka → Postgres |
| `dashboard` | 8501 | Streamlit, 9 pages |

### Open it

- **Dashboard** — <http://localhost:8501>
- **API + Swagger** — <http://localhost:8000/docs>
- **The competitor site** — <http://localhost:8900> ← what the scraper sees

### Everyday commands

```bash
docker compose logs -f producer      # watch it scrape, sweep by sweep
docker compose logs -f api
docker compose restart api
docker compose down                  # stop, keep the data
docker compose down -v               # stop and DELETE volumes (loses history)
docker compose up --build -d         # rebuild after a code change
```

Or through the Makefile: `make up`, `make logs`, `make ps`, `make down`,
`make clean`, `make rebuild`, `make shell`.

---

## Without Docker — running natively

You supply PostgreSQL and Kafka; everything else is Python.

### Prerequisites

| | Needed | Note |
|---|---|---|
| Python | 3.11 recommended | 3.10 works |
| PostgreSQL | 14+ | running and reachable |
| Apache Kafka | 4.x | optional — see step 4 |
| Disk | ~1.5 GB | mostly Prophet and scikit-learn wheels |

### The complete sequence

```bash
# ---------------------------------------------------------------------------
# 1. Code and virtual environment
# ---------------------------------------------------------------------------
git clone <your-repo-url> "AI-DRIVEN DYNAMIC HOTEL PRICING SYSTEM"
cd "AI-DRIVEN DYNAMIC HOTEL PRICING SYSTEM"

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# ---------------------------------------------------------------------------
# 2. Dependencies. Install FROM THE FILE -- cmdstanpy==1.2.4 is a load-bearing
#    pin, and an unpinned install breaks Prophet at runtime.
# ---------------------------------------------------------------------------
python -m pip install --upgrade pip
pip install -e ".[dev]"

# ---------------------------------------------------------------------------
# 3. Create the database
# ---------------------------------------------------------------------------
createdb hotel_pricing                              # or via psql / pgAdmin
psql -c "CREATE USER pricing WITH PASSWORD 'your_password';"
psql -c "GRANT ALL PRIVILEGES ON DATABASE hotel_pricing TO pricing;"

# ---------------------------------------------------------------------------
# 4. Configure
# ---------------------------------------------------------------------------
cp .env.example .env
```

Now edit `.env` — these are the only lines that matter:

```ini
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=hotel_pricing
POSTGRES_USER=pricing
POSTGRES_PASSWORD=your_password

KAFKA_ENABLED=true
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# synthetic = offline generator, no network.
# demo_ota  = real scraping, but you must run `make demo-ota` in another terminal.
INGESTION_SOURCE=synthetic
INGESTION_DEMO_OTA_BASE_URL=http://localhost:8900
```

> **No Kafka?** Set `KAFKA_ENABLED=false`. Everything except the streaming
> pipeline works — competitor rates are persisted directly instead of streamed,
> and `/health` reports Kafka as disabled rather than broken.

```bash
# ---------------------------------------------------------------------------
# 5. Validate the configuration BEFORE building anything
# ---------------------------------------------------------------------------
make config          # == python scripts/check_config.py

# ---------------------------------------------------------------------------
# 6. Build everything: schema -> topics -> data -> seed -> features -> train
#    Idempotent: re-running skips whatever is already done.
# ---------------------------------------------------------------------------
make bootstrap       # 3-8 minutes, mostly Prophet
```

Then run the pieces, **one terminal each**:

```bash
make api             # terminal 1  -> http://localhost:8000/docs
make dashboard       # terminal 2  -> http://localhost:8501
make demo-ota        # terminal 3  -> http://localhost:8900  (only for scraping)
make consumer        # terminal 4  -> Kafka -> Postgres
make producer        # terminal 5  -> collect competitor rates
```

Minimum to see the system work: **terminals 1 and 2.** Add 3–5 for the live
ingestion pipeline.

<details>
<summary>The same thing without <code>make</code></summary>

```bash
python scripts/check_config.py                      # validate .env
python -m database.init_db                          # create the schema
python scripts/create_topics.py                     # create the Kafka topics
python scripts/generate_data.py                     # a year of synthetic data
python scripts/seed_database.py                     # load it into PostgreSQL
python scripts/build_features.py                    # build the feature matrix
python scripts/train_models.py                      # train, write artifacts

python -m uvicorn api.main:app --port 8000                     # terminal 1
python -m streamlit run dashboard/app.py --server.port 8501    # terminal 2
python -m uvicorn demo_ota.app:app --port 8900                 # terminal 3
python scripts/run_consumer.py                                 # terminal 4
python scripts/run_producer.py                                 # terminal 5
```

</details>

### Turning on real scraping natively

```bash
make demo-ota              # terminal 3 -- the target must be up FIRST

# then set INGESTION_SOURCE=demo_ota in .env, and restart the producer:
make producer
```

---

## Verify it actually works

Same for both paths.

```bash
# 1. The API is up and its dependencies are reachable
curl -s http://localhost:8000/health | python -m json.tool

# 2. It can price a room-night
curl -s -X POST http://localhost:8000/api/v1/pricing/predict \
  -H 'Content-Type: application/json' \
  -d '{"hotel_id":"H001","room_type":"deluxe","check_in_date":"2026-12-24"}' \
  | python -m json.tool

# 3. Competitor data is arriving, and from where
curl -s http://localhost:8000/api/v1/ingestion/status | python -m json.tool
```

| Call | Look for |
|---|---|
| `/health` | `"status": "ok"`, and `postgres` / `kafka` / `models` all `up` |
| `/pricing/predict` | a `final_recommended_price`, a named `adjustments` list, `guardrails_applied`, `latency_ms` around 20–30 |
| `/ingestion/status` | `robots.allowed: true`, and `by_source` with a growing `demo_ota` count (Docker) |

That last one is the proof the scraping path is live: `demo_ota` rows are
observations the scraper fetched over HTTP, published to Kafka, and the consumer
wrote to Postgres.

Then run the suite:

```bash
make test            # 893 tests, about 55s
make lint            # unused imports and undefined names
make check           # both
```

The tests need **nothing running** — in-memory SQLite, and they start their own
demo OTA on a free port.

---

## Optional: the AI Agent page

Off by default. Nothing else depends on it.

```bash
pip install -e ".[agent]"
export ANTHROPIC_API_KEY=sk-ant-...     # or add it to .env
```

Then restart the dashboard (or `docker compose up -d --build dashboard`).

Without it, that one page renders these instructions and every other page is
unaffected — see [Where the AI agent fits](#where-the-ai-agent-fits).

---

## Troubleshooting

Every row here is something that actually happened while building this.

| Symptom | Cause | Fix |
|---|---|---|
| `docker compose` fails pulling images, `tls: protocol version not supported` | Docker Hub unreachable on some networks | Images are already pinned to `public.ecr.aws/docker/library/*` where possible. For Kafka: `make kafka-image` builds it locally from the Apache tarball and tags it exactly as Compose expects. |
| Port 5432 already in use | A native PostgreSQL service on the host | Already handled — Compose publishes on **55432**. To change it, set `POSTGRES_HOST_PORT` in `.env`. |
| Ports 8000 / 8501 / 8900 in use | Something else is on them | Set `API_HOST_PORT`, `DASHBOARD_HOST_PORT`, `DEMO_OTA_HOST_PORT` in `.env`. |
| `init` container exits | **It is meant to.** It runs once and stops. | Only a non-zero exit is a problem: `docker compose logs init`. |
| `producer` / `consumer` show "unhealthy" | They inherit the image's API healthcheck and have no endpoint to probe | Already disabled in `docker-compose.yml`. A false unhealthy trains you to ignore the column. |
| `'Prophet' object has no attribute 'stan_backend'` | `cmdstanpy` resolved to 1.3.x, which rejects the CmdStan directory Prophet ships | `cmdstanpy==1.2.4` is pinned in `pyproject.toml` — install from it, not ad hoc. |
| Consumer connects but reads nothing | `kafka-python` 2.0.2 cannot speak to Kafka 4.x | Pinned to `kafka-python==2.3.2`. |
| `ModuleNotFoundError: sqlalchemy` inside a container | A login shell rebuilt `PATH` and dropped the venv | Use `bash -c`, never `bash -lc`, in a container command. |
| Pricing returns 200 but says it used a fallback | No model artifacts yet | `make train`, or `POST /api/v1/models/train`. The API is designed to boot before it has been trained. |
| `/ingestion/status` shows `robots.allowed: false` | The target's robots.txt refuses us | Correct behaviour, not a fault. Booking.com and Expedia both disallow the paths their scrapers need. Use `INGESTION_SOURCE=demo_ota`. |
| Scraping returns 0 rates and "the markup has changed" | `INGESTION_DEMO_OTA_LAYOUT=v2` | That setting exists to *cause* this, so the parse-error path is demonstrable. Set it back to `v1`. |
| AI Agent page says the SDK is missing | It is optional | `pip install -e ".[agent]"` and set `ANTHROPIC_API_KEY`. |

---

## Architecture

```
config/       one settings tree, read from the environment, nothing hardcoded
database/     9 tables, SQLAlchemy 2.0, composite foreign keys
ingestion/    the CompetitorScraper interface + synthetic generator + 3 scrapers
demo_ota/     a stand-in OTA, served locally, for the scraper to scrape
domain/       shared vocabulary (the enums). Imports nothing.
streaming/    event contracts, topics, producer, consumer, handlers
features/     the calendar, the feature pipeline, the feature store
models/       Prophet, Gradient Boosting, metrics, the model registry
training/     the training pipeline
pricing/      rules, demand blending, the pricing engine, guardrails
api/          FastAPI: 12 endpoints, schemas, dependencies, auth
dashboard/    Streamlit: 8 pages, Plotly, an HTTP client and nothing else
agent/        optional LLM analyst over the audit trail. Read-only.
monitoring/   structured logging, data quality, drift, Prometheus metrics
scripts/      every operation as a CLI
tests/        893 tests, incl. 24 module-boundary tests
docs/         architecture, deep dive, ML pipeline, API, deployment, AI agent
```

Three rules hold the design together:

**`pricing/` imports no framework.** No FastAPI, no SQLAlchemy, no Kafka — the
business logic takes numbers and returns numbers. That is what lets every
pricing rule be tested in one line, and it stops HTTP and ORM concerns from
leaking into the part an auditor cares about.

**The dashboard has no database connection.** Every figure it renders comes from
an HTTP call, so there is one implementation of "what is the right price" rather
than two that drift apart. It also means a rendering page is a genuine
integration test of the endpoint behind it.

**Nothing non-deterministic is load-bearing.** `ai_agent/` can be deleted and the
pricing system is exactly what it was; `anthropic` is not even in
a runtime dependency. An LLM explains decisions here, it never makes them — see
[Where the AI agent fits](#where-the-ai-agent-fits).

See [`docs/architecture.md`](docs/architecture.md) for the full design and the
architecture decision records, and
[`docs/technical_deep_dive.md`](docs/technical_deep_dive.md) for the end-to-end
flow diagrams.

---

## The data

The system ships with a **simulator**, not a random number generator. Every
relationship the pricing engine later claims to exploit is put there
deliberately and asserted in the tests.

8 hotels, 6 Indian cities, 4 room types, 365 days — **312,586 rows**.

Measured from the generated data:

| Relationship | Evidence |
|---|---|
| Weekend effect is **segment-specific** | Business hotels: 0.72 weekday vs 0.46 weekend. Leisure: 0.54 vs 0.69 — the mirror image. |
| Seasonality is real | Goa: 0.79 occupancy in winter, 0.34 in the monsoon. |
| Events spike demand | 0.62 mean occupancy on quiet nights, 0.83 during a city event. |
| Demand drives price | corr(occupancy, ADR) = 0.55–0.92 within each room type. |
| Competitors track the market | corr(occupancy, competitor rate) = 0.48–0.87. |
| Cancellations grow with lead time | 6% same-day rising to 21% at 90 days. |
| Business books later than leisure | Mean lead time 24 vs 25 days, with very different curves. |

> **A trap worth knowing about.** Pooled across room types, the
> occupancy-to-price correlation comes out *negative* — suites are the least
> occupied and the most expensive category, so the between-group effect swamps
> the within-group one. Textbook Simpson's paradox. The tests correlate within
> `(hotel, room_type)` and say so in the docstring.

The booking data is stored at **room-night grain on the pickup grid**: one row
per (booking date × stay date), which preserves the booking curve — how demand
for a given night accumulates as that night approaches. That single choice is
what makes leakage-free features possible.

---

## Competitor ingestion

One interface, `CompetitorScraper`, and four implementations. Everything
downstream — producer, Kafka, consumer, features, pricing — depends only on the
interface, so swapping in a licensed rate feed is one subclass and one
environment variable.

| Source | Network? | Default | What it is |
|---|---|---|---|
| `synthetic` | no | offline | Deterministic generator. Always available. |
| `demo_ota` | **yes** | **Compose** | Real scraping against the bundled site |
| `booking` | yes | disabled | Third-party. Opt-in twice over. |
| `expedia` | yes | disabled | Third-party. Opt-in twice over. |

### Why the default scraping target is a site we ship

The obvious demo — point `BookingScraper` at Booking.com — does not work, and it
is important that it does not. Both Booking.com and Expedia **disallow the
search paths those scrapers need** in their `robots.txt`, and
`HttpCompetitorScraper` reads robots.txt and obeys it. So enabling them produces
`ScraperBlocked`, correctly, and you can watch it happen on the Data Ingestion
page.

There were two ways forward. Delete the robots check and get a screenshot, or
scrape something that genuinely grants permission. This project does the second:
`demo_ota/` is a real web server with real HTML and its own `robots.txt` that
allows `/search`.

Nothing about the scraping is faked. The scraper opens a TCP connection, fetches
`/robots.txt`, parses it, respects the rate limiter, requests a search page, gets
a real HTTP status, parses real markup with CSS selectors, and raises
`ScraperParseError` when the markup stops matching. The only thing that is not
third-party is who runs the server.

```bash
# Watch a redesign break the parser, on demand
INGESTION_DEMO_OTA_LAYOUT=v2 docker compose up -d producer
docker compose logs producer     # ScraperParseError: no property cards matched
```

That switch exists because "the site redesigned overnight and every selector now
matches nothing" is the single most common way a scraper dies in production, and
it is the failure this system is built to make loud. `parse()` raises rather than
returning an empty list, because to a pricing engine *"the competitive set
published no rates"* and *"our parser broke"* must never look the same — the
second one would silently widen every competitor band on no data at all.

### Two flags, two different risks

`INGESTION_ENABLE_REAL_SCRAPERS` gates **third-party** sites only. It is about
terms of service and legal exposure, not about whether HTTP is involved — so
gating `demo_ota` behind it would conflate two unrelated risks and put the safe
option behind the frightening switch. `CompetitorSource.is_third_party` is where
that distinction lives.

### Ingestion is observable

```bash
curl -s localhost:8000/api/v1/ingestion/status | python -m json.tool
```

reports the configured source, **what robots.txt actually said**, and how many
observations landed in the last hour. `robots.allowed: false` is a healthy
state, not a fault to clear: it means the scraper asked and was told no. Zero
recent observations, on the other hand, means the feed is dead — which explains a
stale competitor band far better than any story about the market.

The **Data Ingestion** dashboard page shows all of it and can run a pass on
demand. It never scrapes for itself: it calls `POST /api/v1/ingestion/run`, so a
rate on screen is the row that landed in the table.

---

## The Kafka pipeline

Four topics, one envelope, at-least-once delivery.

```
SyntheticCompetitorGenerator ──► producer ──► hotel.competitor_prices
                                                        │
                                                        ▼
                                     consumer ──► validate ──► PostgreSQL
                                                        │
                                                   commit offsets
```

Every message shares an envelope carrying `event_id`, `event_type`, `version`,
`timestamp` and a typed `payload`. The version is present from day one: adding
it later means every consumer has to guess whether a missing field is an old
producer or a bad message.

**Delivery is at-least-once, and every design choice follows from it.**
`enable.auto.commit` is off — offsets are committed only *after* the database
transaction commits, because auto-commit acknowledges messages the database
never saw. Because offsets lag the read position, a crash replays messages, so
every handler is idempotent: `event_id` is unique, and a redelivery becomes a
rejected insert rather than a duplicated observation dragging the competitor
average around.

A **poison message** — bytes that are not a decodable event — is counted, logged
and skipped, and its offset *is* committed. The alternative is one malformed
record blocking its partition permanently, which is far worse than losing a
message that was never valid.

Verified live: 120 events published → 120 written. Replaying the same topic →
**120 duplicates, 0 written**.

---

## Feature engineering

30 features. Every row answers one question: **what did we know about this
night, at the moment we had to price it?**

Getting that wrong is the most common way an ML pricing system produces
excellent offline metrics and useless online prices, so the leakage rules are
explicit:

| Feature | Rule |
|---|---|
| `occupancy_rate`, `available_rooms` | **Gross** rooms on the books at the snapshot. Cancellations are excluded — the schema records which booking a cancellation came from but not *when*, so netting them off would use the future to describe the present. |
| `cancellation_count` | Recent cancellation *pressure*: the trailing 28 days over stay dates already completed at the snapshot. What a revenue manager actually has — a forecast, never the actuals. |
| `historical_demand` | Trailing 28-day mean, over stay dates on or before the snapshot. The window end moves with the snapshot, so a 30-day-out row genuinely sees a month less history. |
| `competitor_*` | Only observations collected by the snapshot, freshest per competitor. A long horizon legitimately finds nothing, which is why `competitor_missing` is a feature rather than an exception. |
| `target_demand` | Final net rooms sold over inventory. Known only after the stay. **Never a feature.** |

The tests prove this by counterfactual: build the features, mutate only data
that arrived *after* the snapshot, rebuild, and assert every feature is
byte-identical. If one moves, it was reading the future.

The resulting booking curve, measured:

| Horizon | 60d | 30d | 14d | 7d | 0d |
|---|---|---|---|---|---|
| Mean occupancy on the books | 0.033 | 0.183 | 0.411 | 0.529 | 0.702 |
| corr(occupancy, final demand) | 0.32 | 0.66 | 0.67 | 0.78 | **0.97** |

See [`docs/ml_pipeline.md`](docs/ml_pipeline.md) for every feature definition.

---

## The models

Two models answering different questions. Neither subsumes the other.

| | Prophet | Gradient Boosting |
|---|---|---|
| Sees | the date | 30 features about the night |
| Knows | *"mid-September Tuesdays in Goa trend like this"* | *"given 72% on the books, a competitor at ₹6,500 and 14 days to go…"* |
| Works | for dates nobody has booked yet | inside the booking window |
| Gives | an uncertainty band | a point estimate + residual spread |

Prophet cannot see today's competitor rate. The GBR cannot see that next
Thursday is Diwali. Blending them is not hedging — it is combining complementary
signal:

```
blended = w · prophet + (1 − w) · gbr
```

If one model is missing, unfitted for that series, or throws, the weight
collapses onto the other and confidence drops. If both are gone the engine falls
back to stored historical demand and says so. **A pricing API that returns 500
because a model file is missing has turned a degraded feature into an outage.**

---

## The pricing algorithm

```
base_price × (1 + demand + occupancy + competitor + season + event) = raw price
guardrails(raw price, context)                                      = final price
```

Adjustments are **additive inside one multiplier, never chained**.
`1.12 × 1.08 × 1.05` is 27%, not the 25% a reader adds up in their head, and the
gap widens with every factor. Additive terms are readable — *"+12 demand, +8
occupancy, so +20"* — and a revenue manager can check them without a calculator.
Each term is clamped before summing, so a broken competitor feed moves the price
by at most its own cap.

The whole multiplier is then scaled by demand confidence (floored at 0.5), so
*"the models are unsure"* degrades towards the base rate rather than towards an
arbitrary number.

### The occupancy × lead-time interaction

The one genuinely non-obvious rule. Occupancy alone does not justify a price
move — it depends entirely on the time left to sell:

| | far out (>21 days) | near (<5 days) |
|---|---|---|
| **high occupancy** | raise hard — demand is real and it arrived early | raise a little — nearly sold out anyway |
| **low occupancy** | hold — there is still time to sell | discount — the room perishes tonight |

### A worked example

```
H001 / deluxe / 2026-09-15
==============================================================
Base price:            INR      7,936   (the rate adjustments apply to)
Current price:         INR      6,000   (what we charge today)

Adjustments
  Demand          +1.9%   forecast demand 67% is about normal
  Occupancy       +0.0%   72% sold at 22 day(s) out is on pace
  Competitor      -9.0%   the market is -18% below our base rate, so we are exposed
  Season          -8.0%   monsoon weakens rates in this market
  Event           +0.0%   no event, holiday or weekend pressure

  Total          -12.2%

Raw price:             INR      6,965   (base -12.2%)

Guardrails
  MAX_DAILY_RISE           INR     6,965 -> INR     6,900
                           rise capped at 15% per day from 6,000

Final price:           INR      6,900   +15.0% vs current
--------------------------------------------------------------
Demand 67% (prophet 61%, gbr 74%), confidence 62%
```

---

## Guardrails

The last gate before a price is allowed out, and **structurally unbypassable**.

`pricing_engine` returns a `RawPrice`. The only function that can construct a
`FinalPrice` is `guardrails.apply()` — constructing one anywhere else raises
`TypeError`. The API can only serialise a `FinalPrice`. There is no code path
that reaches a customer without passing through every rule. *A guardrail a
future refactor can route around is a comment, not a control.*

| Rule | Setting | Behaviour |
|---|---|---|
| Sanity | — | NaN, infinite, negative, or >5× base → fall back to base rate |
| Low-occupancy block | `LOW_OCCUPANCY_THRESHOLD` | Below it, no increase is allowed at all |
| Max daily rise/fall | `MAX_DAILY_CHANGE_PERCENT` | Clamp to ±15% of yesterday |
| Competitor band | `COMPETITOR_UPPER/LOWER_BOUND_PERCENT` | Stay within reach of the market |
| Room floor/ceiling | per room | Per-category limits, inside the global ones |
| Absolute limits | `MIN_PRICE` / `MAX_PRICE` | ₹2,500 – ₹25,000 |

**Order is fixed: relative rules first, absolute rules last.** A floor that a
relative rule can undercut is not a floor.

Every rule that fires is recorded with before/after values and logged at
WARNING. A guardrail firing occasionally is the system working; one firing on
most decisions means the model wants prices the business will not allow — which
is a retuning signal, and only visible if somebody counts.

44 tests cover the guardrails alone, including one that sweeps NaN, infinity,
negatives and ₹1e9 through every combination and asserts the result always lands
inside `[MIN_PRICE, MAX_PRICE]`.

---

## The API

Twelve endpoints. Swagger at `/docs`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness, dependency and model status |
| `GET` | `/api/v1/hotels` | List hotels |
| `GET` | `/api/v1/hotels/{id}` | Hotel, rooms, and 30-day occupancy/ADR/RevPAR |
| `POST` | `/api/v1/pricing/predict` | **Recommend a price** |
| `GET` | `/api/v1/pricing/{id}` | The audit trail |
| `GET` | `/api/v1/forecast/{id}` | Demand forecast with its interval |
| `GET` | `/api/v1/competitors/{id}` | Competitor rates and per-night summaries |
| `POST` | `/api/v1/competitors/events` | Submit an observed rate |
| `GET` | `/api/v1/ingestion/status` | Feed configuration, robots.txt, row counts |
| `POST` | `/api/v1/ingestion/run` | Run one collection pass now |
| `GET` | `/api/v1/models` | Trained versions and their metrics |
| `POST` | `/api/v1/models/train` | Retrain |

### Example request

```bash
curl -X POST http://localhost:8000/api/v1/pricing/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hotel_id": "H001",
    "room_type": "deluxe",
    "check_in_date": "2026-09-15",
    "current_price": 6000,
    "occupancy_rate": 0.72,
    "competitor_rate": 6500,
    "available_rooms": 28
  }'
```

### Example response (abridged)

```json
{
  "hotel_id": "H001",
  "room_type": "deluxe",
  "check_in_date": "2026-09-15",
  "forecasted_demand": 0.6060,
  "predicted_demand": 0.6010,
  "blended_demand": 0.6035,
  "base_price": 7936.0,
  "raw_recommended_price": 6449.19,
  "final_recommended_price": 6449.0,
  "price_change_percent": 7.48,
  "competitor_rate": 6500.0,
  "confidence": 0.756,
  "adjustments": [
    {"name": "demand", "percent": 1.9, "clamped": false,
     "reason": "forecast demand 67% is about normal"},
    {"name": "competitor", "percent": -9.0, "clamped": false,
     "reason": "the market is -18% below our base rate, so we are exposed"}
  ],
  "guardrails_applied": [],
  "model_version": "v1",
  "feature_version": "v1",
  "explanation": "H001 / deluxe / 2026-09-15\n...",
  "latency_ms": 29.68
}
```

Only `hotel_id`, `room_type` and `check_in_date` are required — everything else
is looked up from the feature store when omitted, so a caller who knows only
those three still gets a usable, honest price.

---

## The dashboard — page by page

Nine Streamlit pages at <http://localhost:8501>. None of them holds any business
logic or opens a database connection: every figure comes from an HTTP call, so
there is one implementation of "what is the right price" rather than two that
drift apart.

| # | Page | Endpoint behind it |
|---|---|---|
| — | Overview | `/hotels`, `/hotels/{id}`, `/models` |
| 1 | Hotel Performance | `/hotels/{id}`, `/pricing/{id}` |
| 2 | Demand Forecast | `/forecast/{id}` |
| 3 | Competitor Pricing | `/competitors/{id}`, `/competitors/events` |
| 4 | Dynamic Pricing | `/pricing/predict` |
| 5 | Model Performance | `/models`, `/models/train` |
| 6 | Monitoring | reads `data/monitoring_report.json` |
| 7 | Data Ingestion | `/ingestion/status`, `/ingestion/run` |
| 8 | AI Agent | all of the above, through the agent |

---

### Overview — is the estate healthy?

Portfolio KPIs across all 8 hotels: rooms, cities, and 30-day occupancy, ADR and
RevPAR. A RevPAR-by-property bar chart coloured by market segment, plus which
model versions are currently in service.

**Why RevPAR is the headline number:** occupancy alone rewards giving rooms away,
ADR alone rewards an empty hotel with one expensive suite sold. RevPAR
(occupancy × ADR) is the only one you can't game by sacrificing the other.

**What to look at:** if "Models in service" says *historical fallback*, no
training run has happened yet — pricing still works, it is just using stored
history instead of a model.

### 1. Hotel Performance — one property in detail

Inventory by room type, the rate structure (base, floor, ceiling per category),
recent occupancy and ADR, and the pricing decisions this hotel has actually had.

**What to look at:** the gap between `raw_recommended_price` and
`final_recommended_price` in the history. That gap *is* the guardrails working.
A hotel where they never fire is priced too conservatively; one where they fire
on most nights means the model wants prices the business will not allow, which
is a retuning signal.

### 2. Demand Forecast — what Prophet expects

Forecast for the next N nights with its **80% uncertainty interval**, the weekly
seasonality shape Prophet extracted, and how the interval widens with horizon.

**What to look at:** the band, not the line. Interval width is what the pricing
engine converts into its confidence score, and confidence scales the entire
adjustment multiplier (floor 0.5). A wide band on a night is usually the whole
explanation for a price that looks unresponsive to obvious demand.

> Yearly seasonality is **off** below 730 days of history. With one year it is
> unidentifiable and Prophet extrapolates noise — measured MAE 0.097 with it on
> versus 0.066 with it off. See [Things that went wrong](#things-that-went-wrong).

### 3. Competitor Pricing — where the market is

The competitive band per night (min / average / max) against our own base rate,
a per-source breakdown, and a form to submit an observed rate by hand.

**What to look at:** the *spread*, not just the average. A tight band means the
market has price discipline and moving outside it is conspicuous; a wide one
means there is room to move without anyone noticing. That is exactly why
`competitor_min_rate` and `competitor_max_rate` are separate model features.

A rate submitted through the form goes through the same validation and the same
persistence handler as the scrapers and the Kafka consumer — downstream it is
indistinguishable from one collected automatically, which is the point.

### 4. Dynamic Pricing — the page the whole project exists for

Pick a hotel, room type and night; get a rate. Then:

- a **waterfall chart** building the price from base rate through each of the
  five adjustments to the final number
- every **guardrail** that changed it, named
- an **occupancy sensitivity** curve — the same night repriced across a range of
  occupancies, which is what makes the occupancy × lead-time interaction visible

**What to look at:** the waterfall is the answer to "why this price". Each bar is
a named, individually-clamped adjustment. Nothing is a black box; there is no
step that says "the model decided".

### 5. Model Performance — is the model earning its keep?

Every trained version, metrics against the **predict-the-mean baseline**,
permutation importances, accuracy by lead time, and a button to retrain.

**What to look at:** always the baseline column. A model with R² 0.7 sounds good
until you learn the baseline gets 0.65. Here the baseline is explicit on every
metric — GBR MAE 0.0644 against baseline 0.1430.

Importances are **permutation on the holdout**, not scikit-learn's impurity
measure, which is biased towards high-cardinality columns.

### 6. Monitoring — what is quietly going wrong

Nine data-quality checks, PSI drift per feature, prediction and price
distributions, and guardrail pressure.

**What to look at:** the seasonality caveat. With under two years of history,
comparing a 30-day window against a longer reference **cannot separate drift from
season** — August in Goa genuinely does not look like January in Goa. The monitor
says so explicitly rather than crying wolf every autumn.

Also watch prediction *variance*: a collapsed spread means the model is serving
something close to a constant, which usually means the feed died and every row is
being scored on identical stale features.

### 7. Data Ingestion — where competitor data comes from

Configured source, **what the target's robots.txt actually said**, how many
observations landed in the last hour, and a form to run a collection pass on
demand and watch the rates arrive.

**What to look at:** `robots.allowed`. `false` is a *healthy* state — it means
the scraper asked the site for permission and was told no. And if
`observations_last_hour` is zero while the producer should be running, the feed
is dead, which explains a stale competitor band far better than any story about
the market.

The page never scrapes for itself. It calls `POST /ingestion/run`, so a rate on
screen is the row that landed in the table.

### 8. AI Agent — ask why, in plain language

A chat over the pricing audit trail. Every answer shows the **tool trace** — which
tools were called with which arguments — next to it, because an answer whose
citations cannot be checked is an answer nobody should act on.

Optional. Without the SDK this page renders install instructions and every other
page is unaffected.

---

Every page is executed in the test suite with Streamlit's `AppTest` against a
live API, which makes them genuine end-to-end tests: a page that renders proves
the endpoint behind it works and returns the shape the dashboard expects. **AI
Agent** is the exception — it depends on an optional SDK, so it is *also* tested
in the state where that SDK is absent, which is how most people will first open
it.

---

## Where the AI agent fits

The short version: **the agent reads, explains and simulates. The deterministic
engine decides.**

### Where it deliberately does not go

Not the pricing calculation. Pricing needs reproducibility, auditability and a
28 ms budget, and a language model gives up all three. "The model decided
₹9,400" is not an answer a revenue director or a regulator accepts; `base 7000 ×
1.18 occupancy × 1.14 event, capped at +15%` is.

The tempting middle position — *let the model pick the multipliers and keep the
guardrails* — is **worse than either extreme**. Guardrails clamp the output; they
cannot detect that the reasoning was wrong. A model that hallucinates "Diwali is
next week" produces a price inside every band, with a fluent and completely false
justification recorded in the audit trail. Guardrails bound damage; they do not
create correctness.

Nor the demand forecast — Prophet and the GBR beat the baseline by 39% and 55%
MAE, run in milliseconds, and expose permutation importances. Language models are
not regressors.

### Where it earns its keep

Every pricing decision is already stored with its full arithmetic. What is
missing is *language*, and that is the gap:

- **Pricing agent** — "why did we drop the price on H004 suites last Tuesday?"
  It retrieves the decision, notices the competitor band fired, pulls the rates
  for that night, checks the forecast, and answers with the numbers it read.
- **Monitoring triage** (`make triage`) — turns six correlated alerts into one
  root cause and a "safe to ignore tonight" list.
- **Event extraction** *(not built)* — the event calendar is 19 hand-curated
  records. Reading unstructured sources into that schema is the one use that
  improves the *model* rather than the operator experience.

### How it actually works

```
ai_agent/
  __init__.py   public surface; documents that the whole package is optional
  tools.py      CAPABILITY layer — 6 tools + the allowlist that bounds them
  agent.py      REASONING layer — PricingAgent, the system prompt, the loop
  triage.py     a single LLM call, no tools. Deliberately not an agent.
```

**The two layers do not know about each other.** `tools.py` contains plain
functions and has no idea an LLM exists — they are directly callable and directly
testable. `agent.py` has no idea the tools speak HTTP. That separation is why 26
tests run with no SDK installed.

One turn, end to end:

```
question ──► PricingAgent.ask()
               │  system prompt (cached) + tools + conversation history
               ▼
          Claude Opus 4.8, adaptive thinking, effort=medium
               │  "I need the decision for H004 suite last Tuesday"
               ▼
          SDK tool runner ──► get_pricing_history(...)
               │                    │
               │              _ALLOWED_CALLS gate ── denied? raise
               │                    │ allowed
               │              GET /api/v1/pricing/H004  ──► FastAPI
               │                    │
               │              compact JSON back into context
               ▼
          model decides: need competitor rates too → loop (max 12 iterations)
               ▼
          final text + tool trace + token usage ──► AgentTurn
```

**Two tiers, on purpose.** `agent.py` is a true *agent*: the questions cannot be
enumerated in advance and each tool call depends on what the last one returned.
`triage.py` is a *single call* — one JSON in, one ops note out, no tools. Reaching
for the agent tier where a single call suffices is the most common way to overpay
for an LLM feature.

**Cost control that is actually in the code:** the system prompt and tool list are
byte-identical every turn and carry a `cache_control` breakpoint, so repeat input
bills at roughly a tenth of list price. `MAX_ITERATIONS = 12` stops one badly
formed question from walking the whole competitor table. Every turn returns its
own token counts, and the page shows the running session cost.

**What it is not:** no RAG and no vector database — the data is structured, so
retrieval here is SQL and typed REST, not cosine distance; embedding an exact
query to search it approximately would be a downgrade. No fine-tuning — the
domain knowledge fits in a system prompt, where changing behaviour is a one-line
edit rather than a training run.

### How "it cannot write" is enforced

Not by asking it nicely. `ai_agent/tools.py` holds a fixed allowlist of
`(method, path)` pairs and refuses anything else with `ToolCallDenied`; adding a
write tool fails until somebody deliberately edits that list, which is a diff a
reviewer sees. `persist=False` is hardcoded rather than exposed as a parameter
the model could set.

This is the same reasoning as `pricing/guardrails.py`'s module-private
construction token: both times the answer to "how do we guarantee X?" was to make
the violating state unrepresentable rather than merely discouraged. A constraint
that lives only in a system prompt is advice to a probabilistic system — and
prompt injection is a real path into an agent that reads scraped text.

> **Documented gap:** the allowlist is in-process. The complete control is an API
> credential with no write scope, which this project does not have because it
> ships no auth layer at all. That is stated rather than glossed over.

Full argument, cost model and build order:
[`docs/ai_agent_design.md`](docs/ai_agent_design.md).

---

## MLOps

**Model registry.** A training run writes `gbr_<version>.joblib`,
`prophet_<version>.joblib`, `feature_list.json` and
`training_report_<version>.json`. The registry discovers versions, resolves
which to serve (explicit → `MODEL_ACTIVE_VERSION` → newest), and loads them.

**Loading is lazy and never fatal.** A missing artifact is a degraded service,
not a failed boot. *An API that refuses to start before a training job has run
cannot be deployed before it is trained.*

**The feature contract is checked at load, not at first request.** Every
artifact is saved beside the ordered feature list it was trained on. If the
running code produces a different list — added, removed *or reordered* — loading
fails loudly. A positionally-indexed model reading a shifted matrix produces
plausible, wrong numbers with no error anywhere.

**Reproducibility.** Every run records the dataset SHA-256, the feature version,
the hyperparameters and the exact train/test windows.

**Reload is atomic.** The new bundle is built fully before it is swapped in, so
a failed reload leaves the previous models serving rather than none.

MLflow is deliberately *not* used. ADR-005 explains why: the file-plus-database
registry does what this system needs in ~400 lines with no extra service to run,
and the metadata it records is MLflow-shaped, so adopting MLflow later is an
adapter rather than a rewrite.

---

## Monitoring

`python scripts/monitor.py` runs data quality first and model health second,
deliberately: most production ML failures are data failures wearing a model's
clothes, and a drift number computed over a broken feature build is a
distraction.

**Data quality** — 9 checks: reference integrity, booking recency, competitor
freshness and coverage, feature freshness, feature version, nulls in required
columns, target range, grain duplication.

**Model health** — PSI drift per feature, prediction distribution (a collapsed
spread means a model serving a constant), price distribution, guardrail
pressure, and realised accuracy against completed nights.

> **The most important caveat in the whole layer**, and it is emitted as a check
> rather than buried in a comment: with under two years of history, comparing a
> 30-day window against a longer reference **cannot separate drift from
> season**. August in Goa genuinely does not look like January in Goa. The
> monitor says so explicitly, because a system that cries wolf every autumn is a
> system people mute.

---

## Testing

```bash
make test            # 893 tests, ~55 seconds
make test-cov        # with coverage
make lint            # pyflakes across every package
make check           # both
```

| Module | Tests | Covers |
|---|---|---|
| `test_config` | 40 | Settings, secrets, validation |
| `test_database` | 57 | Schema, constraints, relationships, sessions |
| `test_synthetic_data` | 58 | Determinism and every realism relationship |
| `test_streaming` + `test_streaming_topics` | 66 | Event contracts, producer, consumer, idempotency |
| `test_ingestion` | 55 | Generator, scraper interlocks, HTML parsing |
| `test_demo_ota` | 18 | The scraped site: rates, markup, robots ordering |
| `test_demo_ota_scraper` | 31 | Live HTTP scraping, parse errors, robots, retries |
| `test_features` | 63 | Leakage counterfactuals, parity, the feature store |
| `test_models` | 104 | Metrics, Prophet, Gradient Boosting, artifacts |
| `test_pricing` | 65 | Every rule's sign, blending, degradation |
| `test_guardrails` | 44 | Each rule, ordering, the type gate |
| `test_api` + `test_api_endpoints` | 68 | All 12 endpoints, error codes, the contract |
| `test_api_ingestion` | 16 | Feed status, live collection passes, the run guards |
| `test_ai_agent` | 26 | The read-only allowlist, graceful degradation |
| `test_training` | 26 | Pipeline, versioning, partial failure |
| `test_monitoring` | 44 | PSI arithmetic, every unhealthy state, the registry |
| `test_dashboard` | 39 | Charts, API client, every page rendering |
| `test_logging` | 20 | Formatting, correlation ids, secret redaction |

Tests run on in-memory SQLite where possible, which is only sound because
`database/models.py` deliberately avoids PostgreSQL-only constructs. The
PostgreSQL path is exercised separately by the seeder against a real server.

**The scraper tests start a real server.** A session fixture runs the demo OTA on
an OS-assigned free port and the scraper fetches it over the loopback. Mocking
`httpx` would exercise `parse()` and skip robots.txt, status-code mapping,
connection reuse and the rate limiter — between them most of what a scraper *is*,
and all of what tends to break. It costs about a second, once.

---

## Configuration

Everything is environment-driven; nothing is hardcoded. `cp .env.example .env`
and edit. `python scripts/check_config.py` validates it and prints a redacted
summary.

| Variable | Default | Purpose |
|---|---|---|
| `POSTGRES_HOST/PORT/DB/USER/PASSWORD` | localhost:5432 | Database |
| `KAFKA_BOOTSTRAP_SERVERS` | localhost:9092 | Broker |
| `KAFKA_ENABLED` | true | Set false to run without Kafka |
| `MIN_PRICE` / `MAX_PRICE` | 2500 / 25000 | Absolute guardrails |
| `MAX_DAILY_CHANGE_PERCENT` | 0.15 | Day-over-day cap |
| `MODEL_PROPHET_BLEND_WEIGHT` | 0.5 | Prophet's share of the blend |
| `MODEL_ACTIVE_VERSION` | (newest) | Pin a model version |
| `INGESTION_SOURCE` | demo_ota (Compose) | `synthetic` \| `demo_ota` \| `booking` \| `expedia` |
| `INGESTION_ENABLE_REAL_SCRAPERS` | false | Second lock, third-party sites only |
| `INGESTION_DEMO_OTA_BASE_URL` | http://localhost:8900 | Where the demo OTA is served |
| `INGESTION_DEMO_OTA_LAYOUT` | v1 | `v2` simulates a site redesign |
| `INGESTION_RATE_LIMIT_SECONDS` | 2.0 | Pause between outbound requests |
| `INGESTION_SCRAPE_HORIZONS` | 1,3,7,14,30 | Days ahead each sweep collects |
| `ANTHROPIC_API_KEY` | (unset) | Optional, AI analyst only |

Secrets are `SecretStr`, never logged, and the logging layer has a redaction
filter that scrubs credential-shaped patterns from every record — enforced even
when a future caller forgets.

---

## Things that went wrong

Kept because the debugging is the interesting part.

**Prophet was 66% worse than predicting the mean.** It passed every structural
test — correct shapes, bounded intervals, clean serialisation — while scoring
MAE 0.246 against a 0.148 baseline, R² −6.5. The cause was yearly seasonality
fitted on less than one full cycle: the Fourier term latched onto noise and
extrapolated it confidently. Yearly seasonality and holiday regressors are now
**disabled below 730 days of history**, with the measurements in the docstring.
Result: 47% better than baseline, interval coverage 0.82 against a nominal 0.80.

**The Gradient Boosting model ranked occupancy below `weather_score`.** The
booking curve was flat at ~0.48 occupancy at *every* horizon — impossible, since
on check-in day the rooms on the books essentially are the answer. The cause:
`pd.merge_asof` returns a fresh `RangeIndex`, so restoring row order with
`sort_index()` silently re-sorted by *position* and handed every row some other
row's on-the-books total. The column still looked plausible in aggregate — right
range, right mean — which is why 10 of the 11 tests in that class passed with
the bug in place. Fixed, and the regression test needs *varying* horizons to
catch it; the fixed-horizon companion passes either way.

**Every custom-validator rejection was a 500.** Pydantic v2 puts the original
exception object in `ctx`, so echoing `exc.errors()` into the response body hit
an unserialisable `ValueError`. The handler now emits only `{field, message,
type}` — which also stops the rejected input being echoed back into logs.

**The robots.txt check passed a path robots.txt forbade.** The demo site's
policy listed `Allow: /` above `Disallow: /admin`, and `urllib.robotparser`
implements the original 1994 standard where the **first** matching rule wins —
not Google's longest-match. So `/admin` matched the blanket allow and the
disallow underneath it was dead text. The scraper cheerfully fetched a path it
had been told not to, and every "we honour robots.txt" claim in the codebase was
false for that path. The blanket `Allow: /` bought nothing in the first place —
unmatched paths are permitted by default — so it was deleted and the disallow
moved to the top. Caught only because a test asserted the *refusal*, not just
the permission: a suite that checks the happy path here proves nothing at all.

**The producer crashed on every scraper source.** `run_producer.py` called
`source.requests(...)`, which only `SyntheticCompetitorGenerator` implements —
the base `CompetitorScraper` has no such method, so the branch was an
`AttributeError` waiting to happen. It had never fired because until the demo OTA
existed, no scraper could get past `robots.txt` to reach that code. Making one
path work exposed the other. Now `scraper_stream()` builds the request set from
the hotel catalogue and rebuilds it each sweep, so a long-running producer does
not keep scraping the dates it started with.

**A request guard that could never fire.** `POST /ingestion/run` capped a pass at
400 requests. The schema caps horizons at 12, and the catalogue has 8 hotels
across 4 categories — a maximum of 384. The limit was unreachable, so the only
real protection was that nobody tried. The actual risk was never request *count*
but wall-clock: the same 96 requests are instant with the rate limiter off and
eight minutes at five seconds apart, and it is the second that hangs a browser.
The guard is now on estimated duration, and there is a test asserting the same
pass is *allowed* when it would be fast — otherwise the rejection test would pass
for the wrong reason.

**The forecast endpoint returned dates two months in the past.** Prophet's
`forecast()` continues from the end of the *training* window, and the pipeline
deliberately holds out the last 60 days. "The next 7 nights" has to mean the
next 7 nights.

**A password leak in the settings object.** `DatabaseSettings.url` was a
`computed_field`, and pydantic includes those in `repr` — so any traceback,
debugger frame or pytest assertion dump printed the plaintext DSN. `repr=False`
is load-bearing, not cosmetic.

---

## Production

The stack ships configured for a laptop. This is what changes for anything else,
and what the code refuses to let you forget.

### Startup refuses a half-configured production

`ENVIRONMENT=production` will not boot unless every one of these holds. Each
check fails at **startup**, not on the first request that happens to touch it —
a misconfiguration that only shows up under traffic is a misconfiguration that
ships.

| Refused | Why |
|---|---|
| A default `POSTGRES_PASSWORD` | The shipped value is deliberately unusable-looking rather than plausible |
| `DEBUG=true` | Tracebacks in responses |
| `SECURITY_ENABLED=false` | An open pricing API exposes the commercial audit trail and lets anyone trigger retraining |
| Either key still `dev-*-key` | A key in a public README is not a key |
| `SECURITY_READ_KEY == SECURITY_WRITE_KEY` | Identical keys collapse the two scopes and hand every reader write access |
| `API_CORS_ORIGINS=*` | — |

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"   # run twice
```

```ini
ENVIRONMENT=production
DEBUG=false
POSTGRES_PASSWORD=<generated>
SECURITY_ENABLED=true
SECURITY_READ_KEY=<generated>
SECURITY_WRITE_KEY=<a different one>
API_CORS_ORIGINS=https://your-dashboard.example.com
```

### Two scopes, and why it is two

| Scope | Reaches |
|---|---|
| `read` | Every `GET`, plus `POST /pricing/predict` with `persist=false` |
| `write` | All of the above, plus persisted decisions, competitor submissions, ingestion runs, retraining |

```bash
curl -H "X-API-Key: $READ_KEY"  localhost:8000/api/v1/hotels           # 200
curl -H "X-API-Key: $READ_KEY"  -X POST .../pricing/predict \
     -d '{"...":"...","persist":true}'                                 # 403
```

A single key would be simpler and would give away the most useful property here.
**The AI agent is issued the read key**, so "the agent cannot write" stops being
a promise made by `ai_agent/tools.py`'s allowlist and becomes a fact about the
network. The allowlist still fails fast and still documents intent — it is now
defence in depth rather than the only thing between a prompt injection and a
written price. That was the documented gap in the agent design; this closes it.

`/health` and `/metrics` stay open: a container probe and a Prometheus scrape
cannot carry a credential, and an API key checked into a Prometheus config is
not a secret. Both are protected at the network layer in any real deployment.

### Observability

`GET /metrics` exposes Prometheus text format:

| Metric | Use |
|---|---|
| `http_request_duration_seconds` | Latency, bucketed for a ~28 ms budget rather than the default coarse edges |
| `http_requests_total` | Error rate by route and status |
| `pricing_decisions_total` | Throughput, split by persisted vs simulated |
| **`pricing_guardrail_hits_total`** | **The one to alert on** |
| `ingestion_observations_total` | Competitor feed liveness by source |
| `model_version_info` | Which artifacts are actually serving |

Guardrail pressure is the interesting series. One firing now and then is the
system working. One firing on most decisions means the model wants prices the
business will not allow — a retuning signal that is invisible unless somebody
counts. Latency is labelled by **route template**, never resolved path, so
`/hotels/{hotel_id}` is one series rather than one per hotel.

### Module boundaries are enforced, not described

This is a **modular monolith on purpose**. At eight hotels and one operator,
splitting it into services would put network hops inside a 25 ms pricing path,
turn the `pricing_decisions` audit trail into a distributed transaction, and
multiply the operational surface by six for no benefit anybody could name.

But "modular" is a claim, and an unenforced claim decays. `tests/test_architecture.py`
reads the import graph and asserts 24 boundaries:

- `pricing/` imports **no framework** — no FastAPI, SQLAlchemy, Kafka, HTTP client
- `dashboard/` never opens a database connection
- `ai_agent/` cannot reach the database or the pricing engine, and nothing imports it
- `anthropic` appears in exactly one package
- `demo_ota/` imports nothing from the application — it stands in for a third party
- dependencies point inward: `domain` ← `config` ← `database` ← `features`/`models` ← `pricing` ← `api`
- `FinalPrice`'s construction token exists in exactly one module
- no module reads the clock at import time

> Writing these found a real one. `pricing/` imported `database.models` for two
> enums — plain `str` subclasses, harmless at runtime, but the graph said pricing
> depended on persistence. The fix was to move the shared vocabulary down into
> `domain/` rather than soften the rule into something that no longer meant
> anything.

### Deployment checklist

- [ ] Generate both API keys; set `ENVIRONMENT=production` and let the startup checks run
- [ ] Terminate TLS in front of the API — the keys are static, not bearer tokens
- [ ] Enforce the real rate limit at the load balancer; the in-process one is per replica
- [ ] Point Prometheus at `/metrics` and alert on `pricing_guardrail_hits_total`
- [ ] Kafka: more than one broker, replication factor > 1
- [ ] Managed Postgres with backups; tune the pool for your replica count
- [ ] Schedule `make monitor` and read the seasonality caveat before acting on drift
- [ ] Give the AI agent the **read** key, never the write one

---

## Known limitations

- **One year of synthetic data.** Enough for weekly seasonality and a trend, not
  enough to identify yearly seasonality — which is why Prophet's yearly term is
  off by default. Two years would change that, at the cost of a longer generate
  and seed step.
- **One snapshot per stay date.** The feature store holds a single horizon per
  night, sampled deterministically from the row key. A full panel of every night
  at every horizon would give more rows, but heavily correlated ones.
- **`POST /models/train` is synchronous.** Fine at this scale and honest about
  it in the docstring; at real scale it becomes an enqueue returning 202 with a
  job id. The `TrainingResult` the pipeline already returns is exactly what such
  a job would store.
- **Auth is off by default** — not absent, see [Production](#production). A demo
  that refuses to answer without a header is a demo nobody runs, so
  `SECURITY_ENABLED=false` ships as the default. `ENVIRONMENT=production` refuses
  to boot in that state, so the convenient default cannot reach production by
  accident.
- **The rate limiter is in-process.** With more than one replica each holds its
  own counter, so the effective limit multiplies by the replica count. Behind a
  load balancer, enforce the real quota there.
- **The demo OTA is not market data.** It is a real site and the scraping against
  it is real, but the rates it serves are generated. The honest claim is "the
  ingestion path works end to end", not "these are the prices in Goa". A licensed
  rate feed drops in as one subclass.
- **Single Kafka broker, replication factor 1.** Correct for local development,
  not for production.
- **PSI cannot separate drift from seasonality** on under two years of history.
  The monitor says so rather than pretending otherwise.

---

## Future improvements

- **Multi-horizon feature panel** — every stay date at every horizon, with
  grouped cross-validation to handle the correlation.
- **Price elasticity** — the current engine prices *to* forecast demand; it does
  not model how demand responds to the price it sets. That needs either
  experimentation or an instrumental-variables approach.
- **Length-of-stay and channel pricing** — currently everything is a room-night
  through one channel.
- **Async training** with a task queue, and scheduled retraining triggered by
  the drift monitor rather than by a human.
- **MLflow** for experiment tracking, once there are enough experiments to
  justify the extra service. The registry metadata is already MLflow-shaped.
- **Group-aware cross-validation** — the current chronological split is correct
  but single-fold; a rolling-origin scheme across the whole panel would give
  tighter error bars on the reported metrics.

---

## Documentation

- [`AI_Driven_Hotel_Pricing_System.pptx`](AI_Driven_Hotel_Pricing_System.pptx) — **23-slide technical deck** with speaker notes on every slide. Regenerate with `cd _deck && npm install && node build_deck.js`
- [`docs/technical_overview.md`](docs/technical_overview.md) — **the single technical document**: problem statement, data flow, the model/AI layer, backend architecture, deployment and scaling
- [`docs/interview_guide.md`](docs/interview_guide.md) — **how to explain this project**: the 60-second version, the 5-minute demo order, the numbers to quote, the five questions you will get
- [`docs/architecture.md`](docs/architecture.md) — design, data flows, ADRs
- [`docs/technical_deep_dive.md`](docs/technical_deep_dive.md) — end-to-end flow diagrams, the bugs that shaped the design, interview Q&A
- [`docs/ml_pipeline.md`](docs/ml_pipeline.md) — every feature, both models, the evaluation protocol
- [`docs/api.md`](docs/api.md) — full endpoint reference
- [`docs/deployment.md`](docs/deployment.md) — Compose operations and troubleshooting
- [`docs/ai_agent_design.md`](docs/ai_agent_design.md) — where an LLM agent belongs in this system, and where it must not go

## Screenshots

<!-- Add screenshots of the running dashboard here:
     - dashboard Overview page
     - Dynamic Pricing waterfall
     - Demand Forecast with the uncertainty band
     - Monitoring page
     - Swagger UI at /docs
-->

_To capture: run `docker compose up --build`, then screenshot
<http://localhost:8501> and <http://localhost:8000/docs>._
