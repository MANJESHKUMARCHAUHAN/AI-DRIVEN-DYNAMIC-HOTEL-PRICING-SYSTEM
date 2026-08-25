# Technical Deep Dive

Everything that happens in this system, where it happens, and why it was built
that way — organised so it doubles as interview preparation.

Read this alongside [`architecture.md`](architecture.md) (design decisions),
[`ml_pipeline.md`](ml_pipeline.md) (features and models) and
[`api.md`](api.md) (endpoints).

---

> Looking for the problem statement, the architecture summary, or scaling?
> Those are in [`technical_overview.md`](technical_overview.md). This document
> is the flow-by-flow walkthrough underneath it.

## Table of contents

1. [The system in one picture](#1-the-system-in-one-picture)
2. [Flow A — a competitor rate becomes a database row](#2-flow-a--a-competitor-rate-becomes-a-database-row)
3. [Flow B — raw rows become a feature matrix](#3-flow-b--raw-rows-become-a-feature-matrix)
4. [Flow C — a feature matrix becomes two models](#4-flow-c--a-feature-matrix-becomes-two-models)
5. [Flow D — a request becomes a price](#5-flow-d--a-request-becomes-a-price)
6. [Flow E — the dashboard renders](#6-flow-e--the-dashboard-renders)
7. [Flow F — monitoring runs](#7-flow-f--monitoring-runs)
8. [Where every decision lives](#8-where-every-decision-lives)
9. [Interview questions, with answers](#9-interview-questions-with-answers)
10. [The five bugs, and what each teaches](#10-the-five-bugs-and-what-each-teaches)
11. [A five-minute demo script](#11-a-five-minute-demo-script)

---

## 1. The system in one picture

```
                         ┌───────────────────────────────────────────────┐
                         │              DATA ACQUISITION                 │
                         │                                               │
   ┌──────────────┐      │  SyntheticCompetitorGenerator  (default)      │
   │ Booking.com  │─ ─ ─▶│  BookingScraper    (opt-in, off by default)   │
   │ Expedia      │─ ─ ─▶│  ExpediaScraper    (opt-in, off by default)   │
   └──────────────┘      │         └── all implement CompetitorScraper   │
     (never used by      └───────────────────────┬───────────────────────┘
      default: ADR-004)                          │  CompetitorPricePayload
                                                 ▼
                    ┌────────────────────────────────────────────┐
                    │           streaming/producer.py            │
                    │   EventEnvelope.wrap() → JSON → key=hotel  │
                    └────────────────────┬───────────────────────┘
                                         ▼
              ╔══════════════════════════════════════════════════╗
              ║                  APACHE KAFKA                    ║
              ║  hotel.competitor_prices   3 partitions, 7d      ║
              ║  hotel.booking_events      3 partitions, 7d      ║
              ║  hotel.demand_events       3 partitions, 7d      ║
              ║  hotel.price_predictions   3 partitions, 3d      ║
              ╚════════════════════┬═════════════════════════════╝
                                   ▼
                    ┌──────────────────────────────────────┐
                    │       streaming/consumer.py          │
                    │  poll → decode → validate → handle   │
                    │  → DB commit → THEN offset commit    │
                    └──────────────────┬───────────────────┘
                                       ▼
        ┌──────────────────────────────────────────────────────────┐
        │                      POSTGRESQL                          │
        │  hotels  rooms  bookings  competitor_prices              │
        │  demand_features  predictions  pricing_decisions         │
        │  model_versions  training_runs                           │
        └──────────┬─────────────────────────────────┬─────────────┘
                   │                                 ▲
                   ▼                                 │
    ┌──────────────────────────────┐                 │
    │  features/feature_store.py   │                 │
    │  + feature_engineering.py    │                 │
    │  30 features, snapshot-safe  │                 │
    └──────────────┬───────────────┘                 │
                   ▼                                 │
    ┌──────────────────────────────┐                 │
    │    training/pipeline.py      │                 │
    │  chronological split → fit   │                 │
    └──────┬───────────────┬───────┘                 │
           ▼               ▼                         │
    ┌────────────┐  ┌──────────────┐                 │
    │  Prophet   │  │   Gradient   │   artifacts     │
    │  32 series │  │   Boosting   │──────────┐      │
    └──────┬─────┘  └──────┬───────┘          │      │
           │               │                  ▼      │
           └───────┬───────┘        models/artifacts/│
                   ▼                 gbr_v1.joblib   │
      ┌────────────────────────┐     prophet_v1.joblib
      │ pricing/demand_engine  │     feature_list.json
      │  blended + confidence  │     training_report_v1.json
      └────────────┬───────────┘                     │
                   ▼                                 │
      ┌────────────────────────┐                     │
      │ pricing/pricing_engine │  5 clamped adjustments
      │   → RawPrice           │                     │
      └────────────┬───────────┘                     │
                   ▼                                 │
      ┌────────────────────────┐                     │
      │  pricing/guardrails    │  the ONLY producer  │
      │   → FinalPrice         │  of a FinalPrice    │
      └────────────┬───────────┘                     │
                   ▼                                 │
      ┌────────────────────────┐                     │
      │      api/ (FastAPI)    │─────────────────────┘
      │      10 endpoints      │   persists prediction + decision
      └────────────┬───────────┘
                   │ HTTP only
                   ▼
      ┌────────────────────────┐
      │  dashboard/ Streamlit  │  7 pages, no DB connection
      └────────────────────────┘
```

**The two rules that hold it together:**

- `pricing/` imports no framework. Numbers in, numbers out.
- `dashboard/` has no database connection. HTTP only.

---

## 2. Flow A — a competitor rate becomes a database row

Continuous, asynchronous, at-least-once.

```
 STEP 1  scripts/run_producer.py
         │
         ├─ get_scraper(settings)  ─────────────► ingestion/scraper_base.py
         │    reads INGESTION_SOURCE                 factory; enforces ADR-004
         │    default → SyntheticCompetitorGenerator
         │
         ├─ load_catalog_from_database()
         │    SELECT hotel_id, room_type, base_price FROM rooms
         │    (falls back to the shipped catalogue if the DB is down)
         │
         └─ source.stream(horizons=[1,3,7,14,30,60])
              │
 STEP 2       └─► ingestion/synthetic_generator.py :: fetch()
                   │
                   ├─ demand_index_for(profile, stay_date)   ← shared with the
                   │    city season × day-of-week × holiday    historical generator,
                   │    × event × trend × shock                so streamed and stored
                   │                                          data share one model
                   ├─ _market_rate(...)  competitors' noisy read of the same demand
                   ├─ per competitor: 15% chance of NO rate  (missing data is real)
                   └─ returns List[CompetitorPricePayload]     ← already validated
                                                                by Pydantic
 STEP 3  streaming/producer.py :: EventProducer.send()
         │
         ├─ if not KAFKA_ENABLED → count, log, return False  (never raise)
         ├─ EventEnvelope.wrap(payload)
         │     {event_id: uuid4, event_type, version: 1, timestamp, source, payload}
         ├─ key = hotel_id.encode()   ← one hotel's events stay ordered
         └─ client.send(topic, value, key)
              │
              │  acks=all, retries=5, linger_ms=50
              ▼
 STEP 4  ╔═══════════════════════════════════════╗
         ║   Kafka: hotel.competitor_prices      ║
         ║   partition = hash(hotel_id) % 3      ║
         ╚═══════════════════┬═══════════════════╝
                             ▼
 STEP 5  scripts/run_consumer.py → streaming/consumer.py :: run()
         │
         │  loop:
         ├─ client.poll(timeout_ms=1000, max_records=500)
         │
         ├─ process_records(batch)  ◄── ONE database transaction per batch
         │    │
         │    for each message:
         │    ├─ EventEnvelope.from_bytes(raw)
         │    │     ✗ not JSON / not an object / bad envelope
         │    │       → EventDecodeError → count as POISON, log, SKIP
         │    │         (offset still commits: one bad byte must not
         │    │          block a partition forever)
         │    │
         │    ├─ envelope.decode_payload()
         │    │     ✗ version > SCHEMA_VERSION → refuse
         │    │     ✗ price <= 0, bad room type, missing field → refuse
         │    │
         │    ├─ validator.validate(payload, session)   ← ingestion/validator.py
         │    │     ✗ unknown hotel  → REJECTED (counted by reason)
         │    │     ✗ room the hotel does not sell → REJECTED
         │    │     (reference data cached 300s — a per-message SELECT
         │    │      would be the bottleneck and would only ever learn
         │    │      "still the same eight hotels")
         │    │
         │    └─ handler(envelope, payload, session)    ← streaming/handlers.py
         │          INSERT ... ON CONFLICT (event_id) DO NOTHING
         │          rowcount 0 → DUPLICATE (redelivery: harmless)
         │
         ├─ session.commit()          ◄── database FIRST
         └─ client.commit()           ◄── offsets SECOND
                                          this ordering is the whole
                                          at-least-once guarantee
```

### What happens when things go wrong

| Failure | Behaviour | Why |
|---|---|---|
| Kafka down at publish | `send()` returns `False`, counted, logged | A pricing API must not 500 because an analytics topic is unreachable |
| Kafka down, one probe | Connection is not retried per-event | Otherwise a degraded dependency becomes a latency incident |
| Poison message | Counted, logged, skipped, **offset committed** | One malformed record must not block its partition permanently |
| Unknown hotel | Rejected with a reason code, offset committed | Replaying it produces the same result |
| Database error | Rollback, **rewind to batch start**, retry with backoff ×5 | Nothing is skipped; offsets are not committed |
| Consumer crash | Uncommitted messages redelivered on restart | Handlers are idempotent, so replay is harmless |

**Proven live:** 120 events published → 120 written. Replaying the same topic →
**120 duplicates, 0 written.**

---

## 3. Flow B — raw rows become a feature matrix

Batch. `python scripts/build_features.py`.

```
 INPUT   bookings          224,527 rows   (booking_date × stay_date pickup grid)
         competitor_prices  76,339 rows   (with collected_at)
         demand_features    11,680 rows   (exogenous signals only)
         rooms                  32 rows

 STEP 1  features/feature_store.py :: FeatureStore.load_raw()
         four whole-table reads into pandas

 STEP 2  features/feature_engineering.py :: FeatureBuilder.build()
         │
         ├─ _skeleton()
         │    one row per (hotel, room_type, stay_date)
         │    target = (Σ booking_count − Σ cancellation_count) / room_count
         │    drop the incomplete tail (curve still open)
         │    days_to_checkin = blake2b(seed|hotel|room|date) → weighted choice
         │    snapshot_date   = stay_date − days_to_checkin
         │
         ├─ _add_booking_curve_features()      ◄── the clever bit
         │    │
         │    │  bookings are stored one row per lead time, so a REVERSE
         │    │  cumulative sum turns "taken at exactly this lead" into
         │    │  "on the books at this lead or earlier" in one vectorised pass
         │    │
         │    ├─ curves sorted by (hotel, room, stay, lead ASC)
         │    ├─ reverse cumsum of booking_count, lead×count, adr×count
         │    └─ merge_asof(direction="forward") at each row's own horizon
         │         ⚠ row order restored via an explicit _row column —
         │           merge_asof returns a FRESH RangeIndex, and sort_index()
         │           silently re-sorts by POSITION (see §10, bug 2)
         │    ↓
         │    occupancy_rate, available_rooms, booking_count (7d pickup),
         │    lead_time, current_room_price
         │
         ├─ _add_history_features()
         │    daily series → reindex to a complete calendar → rolling(28)
         │    joined at snapshot_date, NOT stay_date
         │    ↓  historical_demand, cancellation_count
         │
         ├─ _add_competitor_features()
         │    keep only observation_lead >= days_to_checkin  (visible at snapshot)
         │    freshest per competitor → mean / min / max / count
         │    absent → impute base_price, set competitor_missing = 1
         │    ↓  (19.9% of rows have no competitor data — realistic)
         │
         ├─ _add_signal_features()   search_demand, weather_score, event_score
         ├─ _add_calendar_features() recomputed from features/calendars.py,
         │                           never trusted from the database
         ├─ _add_derived_features()  the 7 ratios and interactions
         └─ _finalise()              every column numeric, finite, non-null
                                     or raise — a NaN reaching a model is a
                                     silent failure

 STEP 3  FeatureStore.write()
         UPSERT onto demand_features
         ONLY the derived columns are in the UPDATE clause, so a rebuild
         cannot clobber a signal the streaming consumer wrote a moment ago

 OUTPUT  11,648 rows × 30 features, built in 1.4 s
```

### The leakage guarantee, tested

```python
baseline = build(bookings)                       # normal curve
surge    = build(bookings_with_huge_late_pickup) # only POST-snapshot changed

for column in FEATURE_COLUMNS:
    assert baseline[column] == approx(surge[column])   # nothing moved
assert surge[TARGET] > baseline[TARGET]                # or it proved nothing
```

### The booking curve that results

| Horizon | 60d | 30d | 14d | 7d | 3d | 0d |
|---|---|---|---|---|---|---|
| Mean occupancy | 0.033 | 0.183 | 0.411 | 0.529 | 0.613 | **0.702** |
| corr with target | 0.32 | 0.66 | 0.67 | 0.78 | 0.91 | **0.97** |

---

## 4. Flow C — a feature matrix becomes two models

Batch. `python scripts/train_models.py`.

```
 STEP 1  training/pipeline.py :: TrainingPipeline.run()
         │
         ├─ next_version()   scans training_report_v*.json → "v2"
         │
         ├─ load_features(session)
         │    FeatureStore.load_model_matrix()
         │      = read demand_features  →  to_model_matrix()
         │        re-derives season one-hots + the 7 derived features
         │        using the SAME FeatureBuilder._add_derived_features
         │        that build_serving_row() calls  ◄── parity seam
         │    refuse below 500 labelled rows
         │
         └─ time_based_split(test_days=60)
              train: stay_date <= cutoff   9,728 rows
              test : stay_date >  cutoff   1,920 rows
              ⚠ NEVER shuffled. A random split puts next Tuesday in
                training and reports a score production can never reach.

 STEP 2  _train_gradient_boosting(split)         guarded — see below
         │
         ├─ GradientBoostingRegressor(n=500, lr=0.05, depth=4, subsample=0.8)
         │    early stopping on an internal slice → used 247 trees
         ├─ evaluate(split.test) + baseline_metrics(split.test)
         ├─ residual_std = 0.0833  ← served as the uncertainty band
         ├─ permutation_importance(split.test)   ← holdout, unbiased
         ├─ evaluate_by("days_to_checkin")       ← per-horizon breakdown
         └─ save → models/artifacts/gbr_v2.joblib
                   {format, version, model, features, config, metadata}

 STEP 3  _train_prophet(split)                   guarded
         │
         ├─ daily_demand(split.train) → 32 series of (ds, y)
         ├─ per series: ProphetDemandForecaster.fit()
         │    growth=linear, multiplicative seasonality, cps=0.05
         │    yearly seasonality OFF  (< 730 days of history)
         │    holidays          OFF  (< 730 days of history)
         │    a series that fails is RECORDED and SKIPPED, not fatal
         ├─ evaluate_on(split.test)   ◄── the SAME holdout as the GBR,
         │                                so the two numbers are comparable
         │                                and the blend weight has a basis
         └─ save → models/artifacts/prophet_v2.joblib
                   each Prophet serialised via model_to_json (not pickle)

 STEP 4  save_feature_list()  → feature_list.json   ◄── the train/serve contract
         write training_report_v2.json

 GUARDED STEPS: a Prophet failure does not lose a perfectly good GBR, and vice
 versa. result.succeeded is true if EITHER worked; only both failing raises.
```

### Results, both on the same holdout

| | MAE | RMSE | MAPE | R² | Bias |
|---|---|---|---|---|---|
| Baseline (mean) | 0.1430 | 0.1729 | 31.6% | 0.000 | 0.0000 |
| **Gradient Boosting** | **0.0644** | **0.0838** | 13.6% | **0.765** | −0.0087 |
| Prophet | 0.0879 | 0.1144 | 17.5% | 0.562 | — |

---

## 5. Flow D — a request becomes a price

Synchronous. **~28 ms.** This is the flow to be able to narrate end to end.

```
 POST /api/v1/pricing/predict
 {"hotel_id":"H001","room_type":"deluxe","check_in_date":"2026-09-15",
  "current_price":6000,"occupancy_rate":0.72,"competitor_rate":6500}

 ┌──────────────────────────────────────────────────────────────────────┐
 │ 1. api/main.py  middleware                                           │
 │    X-Correlation-ID bound to a ContextVar → every log line downstream │
 │    carries it                                                        │
 └───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 2. api/schemas.py :: PricingRequestSchema                            │
 │    field validation → 422 with {field, message, type} on failure     │
 │    horizon check: ±365 days                                          │
 │    competitor_min <= competitor_max                                  │
 └───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 3. api/dependencies.py                                               │
 │    require_hotel()  → 404 naming the id                              │
 │    require_room()   → 404 LISTING what the hotel does sell           │
 └───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 4. api/routes/pricing.py :: _build_request()                         │
 │    merge caller input with the database (CALLER WINS — they are       │
 │    describing now, which is fresher than anything stored)             │
 │                                                                       │
 │    _competitor_context()  freshest rate per competitor, ≤30d old      │
 │    _stored_features()     whatever the feature store knows            │
 │    calendars.season_of / is_weekend / is_holiday / event_score        │
 └───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 5. features/feature_engineering.py :: build_serving_row()            │
 │    ★ THE TRAIN/SERVE PARITY SEAM ★                                   │
 │    produces exactly FEATURE_COLUMNS, in order, calling the SAME       │
 │    _add_derived_features() the training matrix uses                   │
 └───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 6. pricing/demand_engine.py :: DemandEngine.estimate()               │
 │                                                                       │
 │    prophet_bundle.demand_on(...)    → 0.5898  [0.52, 0.66]           │
 │    gbr_model.predict_one(features)  → 0.5979  [0.51, 0.68]           │
 │                                                                       │
 │    blended = 0.5·prophet + 0.5·gbr  = 0.5939                         │
 │    confidence = 0.6·f(width) + 0.4·g(disagreement) = 0.748           │
 │                                                                       │
 │    DEGRADATION (never raises):                                        │
 │      one model missing/throwing → weight collapses, conf ≤ 0.70      │
 │      both missing → stored historical demand, conf = 0.25            │
 └───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 7. pricing/pricing_engine.py :: PricingEngine.price()                │
 │                                                                       │
 │    five pure functions in pricing/rules.py, each CLAMPED:             │
 │      demand_adjustment(0.5939, baseline=0.65)      →  −5.2%          │
 │      occupancy_adjustment(0.72, lead=22)           →   0.0%          │
 │      competitor_adjustment(7936, 6500)             →  −9.1%          │
 │      season_adjustment(MONSOON)                    →  −8.0%          │
 │      event_adjustment(0.0, weekend=F, holiday=F)   →   0.0%          │
 │                                              total = −22.3%          │
 │                                                                       │
 │    × confidence_scale(0.748) = 0.874     → −19.4%                    │
 │                                                                       │
 │    raw = 7936 × (1 − 0.194) = 6394.06                                │
 │                                                                       │
 │    returns RawPrice  ◄── a DISTINCT TYPE                             │
 └───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 8. pricing/guardrails.py :: apply(raw, context)                      │
 │                                                                       │
 │    0. sanity        NaN/inf/≤0/>5× base → fall back to base rate     │
 │    ── RELATIVE (while the number can still move) ──                   │
 │    1. low-occupancy block   occ < 40% → no increase at all           │
 │    2. max daily rise/fall   ±15% of current_price                    │
 │    3. competitor band       within +20% of max, −20% of min          │
 │    ── ABSOLUTE (these must win) ──                                    │
 │    4. room floor / ceiling                                            │
 │    5. MIN_PRICE 2500 / MAX_PRICE 25000                               │
 │                                                                       │
 │    returns FinalPrice ◄── constructible ONLY here (private token)    │
 └───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │ 9. persist + publish   (both BEST-EFFORT, never fatal)               │
 │    INSERT predictions      (feature vector as JSONB, for replay)     │
 │    INSERT pricing_decisions (every adjustment, every guardrail)      │
 │    producer.send(PricePredictionPayload, hotel.price_predictions)    │
 │                                                                       │
 │    a correctly computed price is returned even if the audit write     │
 │    or the publish fails — those are warnings, not a 500 for someone   │
 │    who just wanted a rate                                             │
 └───────────────────────────────┬──────────────────────────────────────┘
                                 ▼
                    200  final_recommended_price: 6394.0
                         + 5 adjustments with reasons
                         + every guardrail with before/after
                         + demand breakdown + confidence
                         + a plain-text explanation
                         + latency_ms: 27.85
```

### Why `RawPrice` → `FinalPrice` is the key design decision

```python
_CONSTRUCTION_TOKEN = object()          # module-private

class FinalPrice:
    def __init__(self, *, amount, raw, applied, _token=None):
        if _token is not _CONSTRUCTION_TOKEN:
            raise TypeError("FinalPrice can only be created by guardrails.apply()")
```

The engine returns `RawPrice`. The API serialises `FinalPrice`. There is **no
code path** that reaches a customer without passing every rule. A guardrail a
future refactor can route around is a comment, not a control.

---

## 6. Flow E — the dashboard renders

```
 Browser ──► Streamlit (8501)
              │
              │  dashboard/pages/4_Dynamic_Pricing.py
              │
              ├─ ui.hotel_selector()  ──► api_client.list_hotels()
              │                              GET /api/v1/hotels
              │
              ├─ st.button("Recommend a price")
              │     └─► api_client.predict_price(persist=False)
              │            POST /api/v1/pricing/predict
              │            (persist=False: exploring must not fill the
              │             audit trail with prices nobody charged)
              │
              ├─ charts.adjustment_waterfall(body["adjustments"], base)
              │     a waterfall because that is what the arithmetic IS:
              │     start at base, add five signed percentages, arrive
              │
              └─ occupancy sweep: 10 more calls at 0.1 … 0.98
                    shows the occupancy × lead-time interaction as a curve

 ⚠ NO DATABASE CONNECTION ANYWHERE IN dashboard/
   Every number comes over HTTP, so a rendering page is a genuine
   integration test of the endpoint behind it.
```

The test suite executes all 7 pages with Streamlit's `AppTest` against a live
API and asserts no exception — the closest thing this project has to a
full-stack end-to-end test.

---

## 7. Flow F — monitoring runs

```
 python scripts/monitor.py

 ┌── DATA QUALITY FIRST ────────────────────────────────────────────┐
 │  monitoring/data_monitor.py — 9 checks                           │
 │                                                                  │
 │  reference_data      hotels exist, every hotel has rooms         │
 │  booking_recency     bookings still arriving                     │
 │  competitor_freshness newest rate < 48h        ← fires most often│
 │  competitor_coverage  ≥50% of the next 30 nights                 │
 │  feature_freshness    features rebuilt < 36h ago                 │
 │  feature_version      all rows match FEATURE_VERSION             │
 │  feature_nulls        required columns populated                 │
 │  target_range         realised demand ≤ 1.6                      │
 │  duplicate_grain      the unique constraint still holds          │
 │                                                                  │
 │  WHY FIRST: most production ML failures are data failures        │
 │  wearing a model's clothes, and a drift number computed over a   │
 │  broken feature build is a distraction.                          │
 └──────────────────────────────┬───────────────────────────────────┘
                                ▼
 ┌── MODEL HEALTH ──────────────────────────────────────────────────┐
 │  monitoring/model_monitor.py                                     │
 │                                                                  │
 │  seasonality_caveat  ◄ emitted BEFORE any PSI number             │
 │      "only 364 days of history, so the 30-day window is compared │
 │       against a reference covering different seasons"            │
 │                                                                  │
 │  feature_drift       PSI per feature, bins from the REFERENCE    │
 │                      quantiles (<0.10 none, <0.25 watch, ≥ retrain)│
 │  prediction_variance a collapsed spread = a model serving a      │
 │                      constant  (guarded below 50 predictions)    │
 │  prediction_outliers beyond 3σ                                   │
 │  guardrail_pressure  >50% clamped = the model wants prices the   │
 │                      business will not allow                     │
 │  absolute_limits     >10% pinned to MIN/MAX = miscalibrated      │
 │  realised_accuracy   served predictions vs completed nights      │
 └──────────────────────────────┬───────────────────────────────────┘
                                ▼
              data/monitoring_report.json
                     │
                     └──► dashboard/pages/6_Monitoring.py
              exit code honours --fail-on for cron / CI
```

---

## 8. Where every decision lives

A cheat sheet for "where would I change X?"

| Question | File | Symbol |
|---|---|---|
| How is demand simulated? | `ingestion/synthetic_dataset.py` | `demand_index_for()` |
| What is a holiday? An event? | `features/calendars.py` | `_MOVABLE_HOLIDAYS`, `_CITY_EVENTS` |
| What features exist? | `features/feature_engineering.py` | `FEATURE_COLUMNS` |
| How is leakage prevented? | `features/feature_engineering.py` | `_add_booking_curve_features()` |
| Prophet hyperparameters? | `models/prophet_model.py` | `ProphetConfig` |
| GBR hyperparameters? | `models/gradient_boosting_model.py` | `GradientBoostingConfig` |
| How do the models blend? | `pricing/demand_engine.py` | `DemandEngine.estimate()` |
| What moves the price? | `pricing/rules.py` | 5 `*_adjustment()` functions |
| What limits the price? | `pricing/guardrails.py` | `apply()` |
| The event envelope? | `streaming/events.py` | `EventEnvelope` |
| Idempotent writes? | `streaming/handlers.py` | `_insert_ignoring_conflicts()` |
| Which model is served? | `models/model_registry.py` | `ModelRegistry.resolve_version()` |
| Any threshold or limit | `config/settings.py` | the nine settings groups |

---

## 9. Interview questions, with answers

### "Walk me through what happens when I call the pricing endpoint."

Use [Flow D](#5-flow-d--a-request-becomes-a-price). The nine numbered steps, in
order, in about ninety seconds. The parts worth emphasising: `build_serving_row`
is the same code the training matrix uses, and the `RawPrice → FinalPrice` type
gate means no price can skip the guardrails.

### "How do you prevent data leakage?"

Every feature row is computed as of a **snapshot** N days before the stay.
Three concrete examples:

- `occupancy_rate` uses **gross** rooms on the books, not net — the schema
  records which booking a cancellation came from but not *when*, so netting them
  off would use the future.
- `historical_demand`'s 28-day window ends at the **snapshot**, not the stay
  date, so a 30-day-out row genuinely sees a month less history.
- `competitor_rate` only sees observations with `collected_at <= snapshot`.

And it is tested by counterfactual: mutate only post-snapshot data, rebuild,
assert every feature is byte-identical.

### "Why two models instead of one?"

They answer different questions. Prophet sees only the date and knows
*"mid-September Tuesdays in Goa trend like this"* — it works for nights nobody
has booked. The GBR sees 30 features and knows *"given 72% on the books and a
competitor at ₹6,500…"*. Prophet cannot see today's competitor rate; the GBR
cannot see that next Thursday is Diwali. It is combining complementary signal,
not hedging.

### "How do you know the models are any good?"

Every number is reported next to a predict-the-mean baseline. GBR MAE 0.0644
against 0.1430 — 55% better, R² 0.765. Both models are scored on the **same**
chronological holdout so they are comparable, which is what gives the blend
weight a basis. And there is a *test* that fails if either model stops beating
the baseline — which caught a Prophet configuration that was 66% worse while
passing every structural test.

### "Why not just use a random train/test split?"

This is a time-series panel. A random split puts next Tuesday in the training
set and last Tuesday in the test set, and reports a score the model can never
achieve in production. The split holds out the most recent 60 days.

### "What happens if the model file is missing or corrupt?"

The registry loads lazily and never fatally. The API starts, `/health` reports
models as `unavailable` (not `down` — the service works without them), and
pricing falls back to stored historical demand with confidence 0.25, which the
engine then uses to scale the whole adjustment down towards the base rate. *An
API that refuses to start before a training job has run cannot be deployed
before it is trained.*

### "How do you prevent train/serve skew?"

Three mechanisms. One derivation — `to_model_matrix` (training) and
`build_serving_row` (serving) both call the same `_add_derived_features`.
`feature_list.json` is written beside every artifact and validated at load, and
a reorder fails as loudly as an addition, because a positionally-indexed model
reading a shifted matrix produces plausible wrong numbers. And
`FEATURE_VERSION` is stamped on every stored row, with a monitor check for
mixed versions.

### "Why are the price adjustments additive rather than multiplicative?"

`1.12 × 1.08 × 1.05` is 27%, not the 25% a reader adds up in their head, and the
gap widens with every factor. Additive terms are readable — "+12 demand, +8
occupancy, so +20" — and a revenue manager can check them without a calculator.
Each term is clamped before summing, so a broken competitor feed moves the price
by at most its own cap.

### "How do you guarantee the guardrails cannot be bypassed?"

A type gate, not a convention. `pricing_engine` returns `RawPrice`;
`FinalPrice.__init__` requires a module-private token that only
`guardrails.apply()` holds; the API can only serialise a `FinalPrice`.
Constructing one anywhere else raises `TypeError`, and there is a test for it.

### "Your consumer is at-least-once. How do you avoid duplicates?"

Offsets are committed only after the database transaction, so a crash replays
messages. Every handler is idempotent: `event_id` is unique and inserts use
`ON CONFLICT DO NOTHING`, so a redelivery inserts zero rows and is counted as a
duplicate. Proven by replaying a topic: 120 polled, 120 duplicates, 0 written.

### "What does your monitoring actually catch?"

Data quality first — the most common real failure is a competitor feed that
quietly stops, which no model metric reveals. Then drift by PSI, prediction
distribution (a collapsed spread means a model serving a constant), guardrail
pressure, and realised accuracy against completed nights. And the monitor emits
an explicit caveat that on under two years of history PSI cannot separate drift
from season — because a system that cries wolf every autumn gets muted.

### "What would you do differently at 10× the scale?"

`POST /models/train` becomes an enqueue returning 202 with a job id — the
`TrainingResult` object is already exactly what such a job would store. The
whole-table pandas reads in the feature build become incremental. The overview
page's N+1 hotel calls become one aggregate endpoint. And Kafka gets three
brokers with replication factor 3.

### "What is the weakest part of this system?"

It prices *to* forecast demand; it does not model how demand responds to the
price it sets. Real price elasticity needs either experimentation (A/B on rate)
or an instrumental-variables approach, and without it the engine is optimising a
proxy. Second weakest: one year of synthetic data, which is why Prophet's yearly
seasonality is disabled — so the demand model cannot know December is peak in
Goa, and the event rule has to carry that.

---

## 10. The five bugs, and what each teaches

Good interview material, because debugging is where the thinking shows.

### 1. Prophet was 66% worse than predicting the mean

**Symptom:** MAE 0.246 against a 0.148 baseline, R² −6.5, 80% interval covering
27% of outcomes. Every structural test passed.

**Cause:** yearly seasonality fitted on less than one full cycle. The Fourier
term latched onto noise and extrapolated it confidently.

**Fix:** yearly seasonality and holiday regressors disabled below 730 days, with
the measurement in the docstring. Result: 47% better than baseline, coverage
0.82 against nominal 0.80.

**Lesson:** structural tests prove a component *behaves*; only a quality
assertion proves it is *worth having*. Both models now have a test that fails if
they stop beating the baseline.

### 2. The booking curve was flat at every horizon

**Symptom:** occupancy ~0.48 at 60 days out *and* on check-in day. The GBR
ranked `occupancy_rate` below `weather_score`.

**Cause:** `pd.merge_asof` returns a fresh `RangeIndex`, so restoring row order
with `sort_index()` re-sorted by *position* — every row got some other row's
on-the-books total.

**Fix:** an explicit `_row` column carried through the merge.

**Lesson:** the column kept a plausible range and mean, so 10 of the 11 tests in
that class passed with the bug in place. The regression test needs *varying*
horizons to catch it — the fixed-horizon companion passes either way, because a
stable sort does not reorder. Aggregate assertions cannot catch misalignment.

### 3. Every custom-validator 422 became a 500

**Cause:** pydantic v2 puts the original exception object in `ctx`, and the
handler echoed `exc.errors()` wholesale into the response body.

**Fix:** emit only `{field, message, type}`.

**Lesson:** this also stopped the rejected input being echoed back into logs —
one fix, two problems.

### 4. The forecast endpoint returned dates two months old

**Cause:** Prophet's `forecast()` continues from the end of the *training*
window, and the pipeline deliberately holds out the last 60 days.

**Lesson:** a method that is correct in the training context can be wrong in the
serving context. `forecast_range(start=today)` now exists for the serving one.

### 5. A plaintext password in the settings repr

**Cause:** `DatabaseSettings.url` was a `computed_field`, and pydantic includes
computed fields in `repr` — so any traceback, debugger frame or pytest assertion
dump printed the DSN with the password in it.

**Fix:** `@computed_field(repr=False)`.

**Lesson:** `SecretStr` protects the field, not everything derived from it.

---

## 11. A five-minute demo script

```bash
# 0. Everything, from nothing
docker compose up --build          # ~2 min on a warm image

# 1. The stack is healthy and knows what it is serving
curl -s localhost:8000/health | jq '{status, models}'

# 2. THE headline: price a room, with the full reasoning
curl -s -X POST localhost:8000/api/v1/pricing/predict \
  -H 'Content-Type: application/json' \
  -d '{"hotel_id":"H001","room_type":"deluxe","check_in_date":"2026-09-15",
       "current_price":6000,"occupancy_rate":0.72,"competitor_rate":6500}' \
  | jq -r '.explanation'

# 3. Show a guardrail firing — strong demand on a hotel that is not selling
curl -s -X POST localhost:8000/api/v1/pricing/predict \
  -H 'Content-Type: application/json' \
  -d '{"hotel_id":"H001","room_type":"deluxe","check_in_date":"2026-09-15",
       "current_price":4500,"occupancy_rate":0.10}' \
  | jq '{final_recommended_price, guardrails_applied}'

# 4. The models, next to the baseline they beat
curl -s localhost:8000/api/v1/models | jq '.versions[0].metrics'

# 5. The dashboard — Dynamic Pricing page, waterfall + occupancy sweep
open http://localhost:8501

# 6. Monitoring finding something real
docker compose exec api python scripts/monitor.py
```

**Talking points, in order:** the type gate → the leakage counterfactual → the
two models answering different questions → the guardrail that just fired → the
baseline every number is reported against.
