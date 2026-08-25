# ML Pipeline

Every feature definition, both models, and the evaluation protocol. This is the
document to read before changing anything in `features/`, `models/` or
`training/`.

---

## 1. The target

`target_demand` — final **net** rooms sold for a night, divided by that room
type's inventory.

```
target_demand = (Σ booking_count − Σ cancellation_count) / room_count
```

Clipped at 1.5, not 1.0. Overbooking is real and "we sold 108% of the rooms" is
information about demand, not an error to clip away. A value above ~1.6 means
the inventory denominator is wrong, and the data monitor treats it as critical.

Known only after the stay date. **Never a feature.**

---

## 2. Snapshot discipline

A feature row answers exactly one question:

> What did we know about this night, at the moment we had to price it?

Every row is computed as of a **snapshot** taken `days_to_checkin` days before
the stay. Nothing that happened after that snapshot may reach the model.

### Horizon selection

One row per `(hotel, room type, stay date)`. The horizon is drawn
deterministically from a BLAKE2b hash of the row's own key, weighted towards the
lead times where pricing decisions actually get made:

| Horizon (days) | 0 | 1 | 2 | 3 | 5 | 7 | 10 | 14 | 21 | 30 | 45 | 60 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Weight | .06 | .09 | .09 | .09 | .10 | .11 | .10 | .10 | .09 | .08 | .05 | .04 |

A hash rather than an RNG draw, so the choice is reproducible without threading
generator state through the pipeline, and stable if rows are reordered or the
catalogue grows.

**The alternative considered:** a full panel of every stay date at every
horizon. More rows, but heavily correlated ones, and it changes the feature
store's grain. Since `days_to_checkin` is itself a feature, the model still
learns how the picture differs at 30 days out versus same-day.

### The resulting booking curve

Measured on the shipped dataset:

| Horizon | 60d | 45d | 30d | 21d | 14d | 7d | 3d | 0d |
|---|---|---|---|---|---|---|---|---|
| Mean occupancy on books | 0.033 | 0.071 | 0.183 | 0.287 | 0.411 | 0.529 | 0.613 | 0.702 |
| corr(occupancy, target) | 0.32 | 0.54 | 0.66 | 0.67 | 0.67 | 0.78 | 0.91 | **0.97** |

---

## 3. Leakage rules

The rules, and why each one is what it is:

### `occupancy_rate`, `available_rooms`

**Gross** rooms on the books at the snapshot — every booking whose
`booking_date` is on or before it.

Cancellations are *excluded*. The schema records which booking a cancellation
came from but not *when* it happened, so netting them off at snapshot time uses
information from after the snapshot. Using gross is conservative and leak-free;
using net would be leaky and only slightly more accurate.

### `cancellation_count`

Recent cancellation **pressure**, not this night's cancellations: the trailing 28
days of cancellations on stay dates that had already completed by the snapshot.

This is what a revenue manager actually has — a cancellation forecast, never the
actuals for a night that has not happened.

### `historical_demand`

Trailing 28-day mean realised demand for the same hotel and room type, over stay
dates on or **before the snapshot date**.

The window end moves with the snapshot. A row priced 30 days out genuinely sees
a month less history than a same-day row — exactly as it would in production.

### `competitor_*`

Only observations with `collected_at <= snapshot`, and the **freshest per
competitor**. A rate published three weeks ago is not the market today.

A long horizon can legitimately find nothing, which is why `competitor_missing`
is a feature rather than an exception. On the shipped dataset, 19.9% of rows
have no competitor data at their snapshot.

### `booking_count`, `pickup_velocity`

Rooms taken in the 7 days *before* the snapshot — the slope of the booking curve
at the moment of decision.

### How this is tested

By counterfactual. Build the features, mutate only data that arrived **after**
the snapshot, rebuild, and assert every feature is byte-identical:

```python
baseline = build(make_bookings())
surge    = build(make_bookings(counts=[2, 3, 4, 3, 4, 40, 40, 40, 40]))

for column in FEATURE_COLUMNS:
    assert baseline.iloc[0][column] == approx(surge.iloc[0][column])
assert surge.iloc[0][TARGET] > baseline.iloc[0][TARGET]   # or it proved nothing
```

---

## 4. The feature contract

30 columns, in a fixed order. The order is part of the contract: a
positionally-indexed model reading a shifted matrix produces plausible, wrong
numbers with no error anywhere.

### Base features (19)

| Feature | Definition |
|---|---|
| `occupancy_rate` | Gross rooms on books ÷ inventory, clipped to [0, 1] |
| `available_rooms` | Inventory − rooms on books |
| `total_rooms` | Inventory for this room type |
| `booking_count` | Rooms taken in the 7 days before the snapshot |
| `cancellation_count` | Trailing 28-day completed cancellations |
| `competitor_rate` | Mean of the freshest visible competitor rates |
| `competitor_min_rate` | Lowest of them |
| `competitor_max_rate` | Highest of them |
| `competitor_count` | How many sources were visible |
| `days_to_checkin` | The snapshot horizon |
| `lead_time` | Booking-weighted mean lead time of rooms on the books |
| `search_demand` | Exogenous search interest, 0–1 |
| `historical_demand` | Trailing 28-day mean realised demand |
| `current_room_price` | Revenue-weighted ADR achieved so far |
| `is_weekend` | Saturday or Sunday |
| `day_of_week` | 0–6, Monday first |
| `holiday_flag` | Public holiday or major festival |
| `local_event_score` | City event pressure, 0–1 |
| `weather_score` | Seasonal pleasantness, 0–1 |

### Season (4, one-hot)

`season_winter`, `season_summer`, `season_monsoon`, `season_autumn`.

One-hot rather than an integer code: an ordinal encoding invites a tree to split
on "summer > winter", which is meaningless.

### Derived (7)

| Feature | Definition | Why it earns a place |
|---|---|---|
| `price_to_competitor` | `current_room_price / competitor_rate` | Where we sit matters more than either level |
| `competitor_spread` | `(max − min) / rate` | Wide spread = weak market discipline = room to move |
| `pickup_velocity` | `booking_count / 7` | The slope of the curve, not its level |
| `occupancy_x_lead` | `occupancy × (1 − h/60)` | 80% full at 30 days out ≠ 80% on the day |
| `demand_pressure` | `search_demand × (1 + event_score)` | Forward interest, amplified by the city |
| `competitor_missing` | 1 when no rate was visible | Lets trees learn "when the market is invisible, trust our own signals" |
| `holiday_proximity` | Decayed holiday significance ±2 days | Travel happens around the holiday, not only on it |

---

## 5. Train/serve parity

The single most dangerous failure mode in this system is train/serve skew,
because it produces confidently wrong prices with no error anywhere. Three
mechanisms guard against it:

1. **One derivation.** `demand_features` stores only *base* columns. The season
   one-hots and the derived features are rebuilt by
   `FeatureStore.to_model_matrix` (training) and `build_serving_row` (serving),
   and both call the same `FeatureBuilder._add_derived_features`. A change to a
   ratio cannot land on one path and not the other.

2. **`feature_list.json`** is written beside every artifact and validated at
   load. Added, removed *or reordered* columns all raise
   `FeatureVersionMismatch`, and the API refuses to serve a Gradient Boosting
   model whose contract does not match.

3. **`FEATURE_VERSION`** is stored on every feature row. The data monitor warns
   when the store holds rows from a different pipeline version.

The tests assert that `build_serving_row` produces exactly `FEATURE_COLUMNS`, in
order, and that given identical inputs its derived values match the training
path's to floating-point tolerance.

---

## 6. Prophet

Forecasts the **shape of demand over time** — trend, weekly rhythm, yearly
season, holiday spikes — for dates nobody has booked yet.

One model per `(hotel, room type)`: 32 series, fitted in about 14 seconds. A
single pooled model cannot represent a business hotel that empties at the
weekend and a resort that fills, at the same time.

### Configuration, and why

| Parameter | Value | Reason |
|---|---|---|
| `growth` | `linear` | Logistic measured no better (MAE 0.067 vs 0.066) while adding a capacity parameter that has to be guessed. Forecasts are clipped to `[0, cap]` anyway. |
| `seasonality_mode` | `multiplicative` | Demand swings scale with the level |
| `changepoint_prior_scale` | 0.05 | 0.02 was too stiff to follow the drift; 0.30 chased noise |
| `interval_width` | 0.80 | The pricing engine turns width into confidence; a 95% band makes every night look equally uncertain |
| `yearly_seasonality` | **off below 730 days** | See below |
| `holidays` | **off below 730 days** | Each holiday occurs once in a year of data |
| `yearly_fourier_order` | 6 | Ten harmonics fit holiday-week wiggles the holiday regressors already explain |

### The yearly-seasonality finding

The first configuration shipped was **66% worse than predicting the mean**,
while passing every structural test. Measured on a 60-day holdout across four
series with ~300 training days available:

| Configuration | MAE | Interval coverage (nominal 0.80) |
|---|---|---|
| yearly seasonality **on** | 0.097 | 0.53 |
| yearly seasonality **off** | **0.066** | **0.87** |
| baseline (predict the mean) | 0.113 | — |

With less than one full cycle in the training window the yearly Fourier term is
unidentifiable: Prophet fits it to noise and extrapolates that noise with full
confidence. Two cycles is the point at which the term becomes separable from the
trend, so `MIN_DAYS_FOR_YEARLY = 730`.

Enabling holidays on one year of data cost a further 0.066 → 0.068, for the same
reason: each holiday's coefficient comes from a single observation.

### Serialisation

Prophet objects hold a compiled Stan backend and do not pickle reliably across
versions. Each fitted model is serialised with Prophet's own `model_to_json`,
and joblib stores the *strings*. The artifact survives a Prophet upgrade.

Note: the interval bounds are Monte-Carlo sampled, so they move ~0.3% between
calls on the same model. Only `yhat` and `trend` are deterministic.

---

## 7. Gradient Boosting

Predicts the **level of demand for a specific night** given everything currently
known about it.

`sklearn.ensemble.GradientBoostingRegressor`, one model for the whole estate.

| Parameter | Value | Reason |
|---|---|---|
| `n_estimators` | 500 | With early stopping, overshooting costs time not accuracy. 247 were used. |
| `learning_rate` | 0.05 | The other half of the capacity knob |
| `max_depth` | 4 | Not the default 3 — the signal is interactional and depth 3 cannot express a three-way interaction |
| `min_samples_leaf` | 20 | ~0.2% of the training set |
| `subsample` | 0.8 | Stochastic gradient boosting; decorrelates the trees |
| `max_features` | `sqrt` | ~5 of 30 features per split |
| `n_iter_no_change` | 25 | Early stopping on an internal validation slice |

`validation_fraction` uses scikit-learn's *random* slice, which is acceptable
only because it decides when to stop rather than reports a score. Every reported
number comes from the chronological holdout.

### Uncertainty

The holdout residual standard deviation (0.0833) is stored with the artifact and
served as a band around every prediction. A point estimate with no error bar is
not usable by a pricing engine that has to decide how far to move.

---

## 8. Evaluation protocol

### The split is chronological, never random

```python
cutoff = max(stay_date) − 60 days
train  = rows with stay_date <= cutoff        # 9,728
test   = rows with stay_date >  cutoff        # 1,920
```

`train_test_split(shuffle=True)` would put next Tuesday in training and last
Tuesday in test, and report a score the model can never achieve in production.

**Both models are scored on the same holdout.** Prophet is fitted on
`split.train` and evaluated with `ProphetBundle.evaluate_on(split.test)` — not
on its own internal backtest, which measures a different thing on a different
window. Comparable numbers are what give the blend weight a basis.

### Metrics

Defined once in `models/metrics.py` and shared, so "MAPE" means one thing across
the project.

MAPE gets special handling because the naive formula divides by the actual
value, which explodes near zero — and a hotel with two rooms sold on a wet
Tuesday in the monsoon is exactly the kind of row this dataset contains:

- `mape` masks actuals below 0.02 and reports `mape_coverage` alongside, so a
  headline MAPE can never quietly be computed over 60% of the data
- `smape` is symmetric and bounded at 200%
- `weighted_mape` weights by volume — the business metric, since being wrong
  about a full hotel matters more than about an empty one

Every result is reported next to `baseline_metrics` (predict the mean). A model
that beats nothing is not a model.

### Results

| | MAE | RMSE | MAPE | wMAPE | R² | Bias |
|---|---|---|---|---|---|---|
| Baseline | 0.1430 | 0.1729 | 31.60% | 26.63% | 0.000 | 0.0000 |
| **Gradient Boosting** | **0.0644** | **0.0838** | 13.56% | 11.99% | **0.765** | −0.0087 |
| Prophet | 0.0879 | 0.1144 | 17.46% | — | 0.562 | — |

Bias near zero matters: a model with excellent MAE and a large bias is
systematically wrong in one direction, which for pricing means consistently
over- or under-charging.

---

## 9. Blending

```
blended = w · prophet + (1 − w) · gbr        # w = MODEL_PROPHET_BLEND_WEIGHT
```

Confidence combines two things, and **both must be good** to earn a high number:

```
confidence = 0.6 · f(interval width) + 0.4 · g(model disagreement)
```

Two confident models that disagree is precisely when the blend is least
trustworthy, and averaging them hides that — so disagreement is penalised
separately rather than being allowed to cancel out.

Degradation:

| Situation | Behaviour |
|---|---|
| Both available | Blend at `w`, full confidence range |
| One missing or throwing | Weight collapses to the other, confidence capped at 0.70 |
| Neither | Stored historical demand for that hotel/room/weekday, confidence 0.25 |

---

## 10. Reproducibility

Every training run records:

- **`dataset_hash`** — SHA-256 over the feature matrix bytes plus the column
  names. Same data and same hyperparameters must give the same model.
- **`feature_version`**, and the ordered feature list on disk
- **Hyperparameters**, in full
- **Train and test windows**, as dates
- **Metrics and the baseline**, side by side
- **Permutation importances** and per-horizon accuracy

Written to `models/artifacts/training_report_<version>.json`.
