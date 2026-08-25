# API Reference

Ten endpoints. Interactive documentation at `/docs`, OpenAPI schema at
`/openapi.json`.

Base URL: `http://localhost:8000`. Business endpoints sit behind `/api/v1`;
`/health` is at the root because container healthchecks and load balancers look
for it there.

---

## Conventions

### Error envelope

Every non-2xx response has the same shape:

```json
{
  "error": "http_404",
  "detail": "No hotel with id 'H999'",
  "correlation_id": "bfa6d197986d41c8",
  "timestamp": "2026-08-24T11:28:37.986512Z",
  "context": null
}
```

`correlation_id` ties the response to the server logs. It is echoed from the
`X-Correlation-ID` request header when one is supplied, and generated otherwise;
either way it comes back in the `X-Correlation-ID` response header.

### Status codes

| Code | Meaning |
|---|---|
| 200 | Success |
| 202 | Accepted for processing (competitor events) |
| 404 | The hotel, room type or series does not exist |
| 409 | The request is fine but the *state* is wrong — not enough data to train |
| 422 | The request failed schema validation |
| 503 | A capability is missing — no model is loaded |

409 versus 422 is a deliberate distinction: 422 means "fix your request", 409
means "fix the system's state" (seed the database, build features).

503 versus 500 likewise: a missing model is a capability that resolves by
training, not a fault that resolves by debugging.

### Validation errors

`context.errors` carries only what helps a caller fix the request:

```json
{
  "error": "validation_error",
  "detail": "Request failed schema validation.",
  "context": {
    "errors": [
      {
        "field": "check_in_date",
        "message": "Value error, check_in_date is 26428 days ahead; the models cannot say anything useful beyond 365 days",
        "type": "value_error"
      }
    ]
  }
}
```

The rejected input is deliberately **not** echoed back.

---

## `GET /health`

Liveness, dependency reachability and model availability.

Always returns 200 while the process is alive. A dependency being down
downgrades `status` to `degraded` but does not fail the request — a healthcheck
that kills the API for being unable to reach Postgres turns a database blip into
an outage.

```json
{
  "status": "ok",
  "app": "dynamic-hotel-pricing",
  "version": "0.1.0",
  "environment": "docker",
  "dependencies": [
    {"name": "postgres", "state": "up", "target": "postgres:5432", "latency_ms": 49.84},
    {"name": "kafka",    "state": "up", "target": "kafka:9092",    "latency_ms": 109.52},
    {"name": "models",   "state": "up", "target": "/app/models/artifacts",
     "detail": "serving gradient_boosting, prophet"}
  ],
  "models": {"active_version": "v1", "available": ["gradient_boosting", "prophet"]}
}
```

Probes are protocol-level, not socket-level: `SELECT 1` for Postgres and a
metadata request for Kafka. A TCP connect to 5432 only proves something is
listening; in KRaft mode a Kafka broker accepts connections well before it can
serve metadata. Each probe does a fast TCP check first so an unreachable
dependency fails in half a second rather than on the pool's ten-second timeout.

`unavailable` (models) is distinct from `down`: the service works without models,
on the historical fallback.

---

## `GET /api/v1/hotels`

| Parameter | Type | Default | |
|---|---|---|---|
| `city` | string | — | Case-insensitive |
| `star_rating` | int 1–5 | — | |
| `active_only` | bool | true | |
| `limit` | int 1–500 | 100 | |

```bash
curl "http://localhost:8000/api/v1/hotels?city=mumbai"
```

---

## `GET /api/v1/hotels/{hotel_id}`

A hotel, its room inventory, and the last 30 nights of trading.

```json
{
  "hotel_id": "H001",
  "hotel_name": "Sanchay Grand Mumbai",
  "city": "Mumbai",
  "star_rating": 5,
  "total_rooms": 240,
  "segment": "business",
  "rooms": [
    {"room_id": "H001-DEL", "room_type": "deluxe", "capacity": 2,
     "room_count": 72, "base_price": 7936.0,
     "floor_price": 5158.4, "ceiling_price": 17459.2}
  ],
  "occupancy_last_30_days": 0.6035,
  "adr_last_30_days": 7767.34,
  "revpar_last_30_days": 4687.37
}
```

RevPAR — occupancy × ADR — is included because it is the number hotels actually
manage against. Occupancy alone rewards giving rooms away; ADR alone rewards an
empty hotel with one expensive suite sold.

**404** when the hotel does not exist.

---

## `POST /api/v1/pricing/predict`

The endpoint the system exists for.

### Request

Only `hotel_id`, `room_type` and `check_in_date` are required. Everything else
refines the answer; anything omitted is looked up from the feature store or
contributes nothing.

| Field | Type | Notes |
|---|---|---|
| `hotel_id` | string | required |
| `room_type` | enum | `standard` \| `deluxe` \| `premium` \| `suite` |
| `check_in_date` | date | required, within ±365 days |
| `current_price` | float > 0 | Enables the day-over-day change cap |
| `occupancy_rate` | float 0–1 | Rooms already on the books |
| `available_rooms` | int ≥ 0 | Used to derive occupancy when it is absent |
| `competitor_rate` | float > 0 | Omit and the market is looked up |
| `competitor_min_rate` | float > 0 | Must not exceed `competitor_max_rate` |
| `competitor_max_rate` | float > 0 | |
| `base_price` | float > 0 | Overrides the room's rack rate |
| `as_of` | date | Pricing date, for lead time. Defaults to today |
| `persist` | bool | Default true. Set false for what-if queries |

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

### Response

```json
{
  "hotel_id": "H001",
  "room_type": "deluxe",
  "check_in_date": "2026-09-15",
  "currency": "INR",

  "forecasted_demand": 0.5898,
  "predicted_demand": 0.5979,
  "blended_demand": 0.5939,

  "base_price": 7936.0,
  "current_price": 6000.0,
  "raw_recommended_price": 6394.06,
  "final_recommended_price": 6394.0,
  "price_change_percent": 6.57,

  "competitor_rate": 6500.0,
  "confidence": 0.7482,

  "adjustments": [
    {"name": "demand", "value": -0.0524, "percent": -5.24, "clamped": false,
     "reason": "forecast demand 59% is below the 65% baseline, so price has to work harder",
     "inputs": {"blended_demand": 0.5939, "baseline_demand": 0.65}},
    {"name": "occupancy", "value": 0.0, "percent": 0.0, "clamped": false,
     "reason": "72% sold at 22 day(s) out is on pace", "inputs": {}},
    {"name": "competitor", "value": -0.0905, "percent": -9.05, "clamped": false,
     "reason": "the market is -18% below our base rate, so we are exposed", "inputs": {}},
    {"name": "season", "value": -0.08, "percent": -8.0, "clamped": false,
     "reason": "monsoon weakens rates in this market", "inputs": {"season": "monsoon"}},
    {"name": "event", "value": 0.0, "percent": 0.0, "clamped": false,
     "reason": "no event, holiday or weekend pressure", "inputs": {}}
  ],
  "total_adjustment": -0.1943,

  "guardrails_applied": [],
  "guardrail_detail": [],

  "demand": {
    "blended_demand": 0.5939, "forecasted_demand": 0.5898,
    "predicted_demand": 0.5979, "prophet_weight": 0.5,
    "confidence": 0.7482, "lower": 0.5239, "upper": 0.6639,
    "disagreement": 0.0081,
    "sources": ["prophet", "gradient_boosting"],
    "degraded": false, "notes": []
  },

  "model_version": "v1",
  "feature_version": "v1",
  "prediction_id": "3f9a2c1e-...",
  "explanation": "H001 / deluxe / 2026-09-15\n===...",
  "latency_ms": 27.85,
  "timestamp": "2026-08-24T12:16:03.129Z"
}
```

`explanation` is the whole calculation as readable text — the thing you paste
into a ticket when somebody queries a rate.

### Guardrails in the response

When a rule fires:

```json
"guardrails_applied": ["MAX_DAILY_RISE"],
"guardrail_detail": [{
  "rule": "MAX_DAILY_RISE",
  "before": 6965.0, "after": 6900.0, "delta": -65.0,
  "reason": "rise capped at 15% per day from 6,000"
}]
```

### Errors

| Code | When |
|---|---|
| 404 | Unknown hotel, or a room type the hotel does not sell (the message lists what it does) |
| 422 | Schema violation, a date beyond ±365 days, or an inverted competitor band |

---

## `GET /api/v1/pricing/{hotel_id}`

The audit trail, newest first.

| Parameter | Type | Default |
|---|---|---|
| `room_type` | enum | — |
| `limit` | int 1–500 | 50 |

```json
{
  "hotel_id": "H001",
  "count": 2,
  "items": [{
    "prediction_id": "3f9a2c1e-...",
    "check_in_date": "2026-09-15",
    "base_price": 7936.0,
    "raw_recommended_price": 6394.06,
    "final_recommended_price": 6394.0,
    "price_change_percent": 6.57,
    "guardrails_applied": [],
    "blended_demand": 0.5939,
    "confidence": 0.7482,
    "model_version": "v1",
    "created_at": "2026-08-24T12:16:03Z"
  }]
}
```

---

## `GET /api/v1/forecast/{hotel_id}`

Prophet's demand forecast, unblended.

| Parameter | Type | Default | |
|---|---|---|---|
| `room_type` | enum | `deluxe` | |
| `horizon_days` | int 1–90 | 30 | The spec's headline horizons are 7, 14, 30 |
| `start_date` | date | today | First night to forecast |

```json
{
  "hotel_id": "H001",
  "room_type": "deluxe",
  "horizon_days": 7,
  "model_version": "v1",
  "points": [
    {"date": "2026-08-24", "forecast": 0.6042, "lower": 0.4899, "upper": 0.7258, "trend": 0.6561},
    {"date": "2026-08-29", "forecast": 0.3521, "lower": 0.2394, "upper": 0.4713, "trend": 0.6543}
  ]
}
```

Demand is a fraction of inventory: 0.60 means 60% of rooms are expected to sell.
The Saturday dip above is H001's business-hotel weekly profile.

**Note:** the forecast starts from `start_date` (default today), *not* from the
end of the training window — the training pipeline deliberately holds out the
last 60 days, so "the next 7 nights" would otherwise return dates two months
old.

| Code | When |
|---|---|
| 404 | Unknown hotel/room, or no series was fitted for it |
| 503 | No Prophet model is loaded |

---

## `GET /api/v1/competitors/{hotel_id}`

| Parameter | Type | Default |
|---|---|---|
| `room_type` | enum | all |
| `start_date` | date | today |
| `end_date` | date | today + 30 |
| `limit` | int 1–500 | 200 |

```json
{
  "hotel_id": "H001",
  "count": 8,
  "summaries": [{
    "check_in_date": "2026-08-24",
    "room_type": "deluxe",
    "competitor_rate": 7600.64,
    "competitor_min_rate": 7325.32,
    "competitor_max_rate": 7863.16,
    "competitor_count": 4,
    "spread_percent": 7.08
  }],
  "observations": [{
    "competitor": "booking", "room_type": "deluxe",
    "check_in_date": "2026-08-24", "price": 7863.16,
    "currency": "INR", "is_available": true,
    "source": "synthetic", "collected_at": "2026-08-24T09:41:00Z"
  }]
}
```

Summaries use only the **freshest observation per competitor**; an average over
three weeks of history is not a description of today's market.

`spread_percent` is `(max − min) / mean`. A wide spread means weak price
discipline in the market and more room to move.

---

## `POST /api/v1/competitors/events`

Submit an observed competitor rate. Returns **202**.

```bash
curl -X POST http://localhost:8000/api/v1/competitors/events \
  -H "Content-Type: application/json" \
  -d '{"hotel_id":"H001","competitor":"booking","room_type":"deluxe",
       "check_in_date":"2026-09-15","price":6200,"currency":"INR"}'
```

```json
{
  "accepted": true,
  "event_id": "8f2c1e4a-...",
  "published_to_kafka": true,
  "persisted": true,
  "detail": "accepted"
}
```

The event goes through the **same** Pydantic payload the Kafka consumer
validates and the **same** persistence handler, so a rate submitted here is
indistinguishable downstream from one collected by a scraper. It is stored even
when Kafka is unavailable — a broker outage delays the stream rather than losing
the data — and `detail` says so.

Re-submitting an identical event is idempotent: `detail` becomes
`"already recorded"`.

---

## `GET /api/v1/models`

```json
{
  "active_version": "v1",
  "available": ["gradient_boosting", "prophet"],
  "loaded_at": "2026-08-24T12:14:58Z",
  "errors": {},
  "feature_version": "v1",
  "versions": [{
    "version": "v1",
    "is_active": true,
    "gradient_boosting": "gbr_v1.joblib",
    "prophet": "prophet_v1.joblib",
    "trained_at": "2026-08-24T12:14:31Z",
    "feature_version": "v1",
    "dataset_hash": "a3f1...",
    "n_train": 9728,
    "n_test": 1920,
    "metrics": {
      "gradient_boosting": {"mae": 0.0644, "rmse": 0.0838, "r2": 0.7653},
      "prophet": {"mae": 0.0879, "rmse": 0.1144, "r2": 0.562}
    }
  }]
}
```

`errors` is the field to watch. A `feature_list` entry mentioning a contract
mismatch means the running code no longer produces the features a model was
trained on — the model is **not** served, because doing so would produce
silently wrong prices.

---

## `POST /api/v1/models/train`

Runs the full training pipeline. **Synchronous, and takes tens of seconds.**

| Field | Type | Default |
|---|---|---|
| `test_days` | int 7–365 | 60 |
| `train_prophet` | bool | true |
| `train_gradient_boosting` | bool | true |
| `backtest_folds` | int 0–5 | 0 (above zero roughly doubles the time) |
| `reload_after` | bool | true |

```json
{
  "version": "v2",
  "succeeded": true,
  "duration_seconds": 18.4,
  "n_train": 9728,
  "n_test": 1920,
  "steps": [
    {"name": "gradient_boosting", "succeeded": true,
     "metrics": {"mae": 0.0644, "r2": 0.7653},
     "baseline": {"mae": 0.1430}},
    {"name": "prophet", "succeeded": true,
     "metrics": {"mae": 0.0879}, "baseline": {"mae": 0.1430}}
  ],
  "artifacts": {"gradient_boosting": "...", "prophet": "...", "report": "..."},
  "reloaded": true,
  "summary": "version v2\n..."
}
```

Each model trains in its own guarded step: a Prophet failure does not lose a
perfectly good Gradient Boosting model, and `steps[].succeeded` says which
happened.

**409** when there is not enough labelled data — seed and build features first.

At real scale this becomes an enqueue returning 202 with a job id; the
`TrainingResult` the pipeline already returns is exactly what such a job would
store.

---

## CORS

Restricted to `API_CORS_ORIGINS` (default `http://localhost:8501` and
`http://dashboard:8501`). Never `*`. Methods are limited to `GET` and `POST`.
