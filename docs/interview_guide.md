# Explaining This Project

How to talk about this system: what to lead with, what to demo, which numbers to
quote, and how to answer the questions that actually get asked.

---

## The one thing to get right

Most candidates describe a pricing project as *"I used Prophet and XGBoost to
predict demand and set prices."* That sentence is forgettable, and it invites the
only follow-up that matters — "how do you know it works?" — with nothing prepared.

Lead with the **problem shape** instead, because it is what makes every technical
decision downstream make sense:

> A hotel room is the most perishable product there is. An unsold room-night is
> worth exactly zero at midnight and there is no inventory to carry forward. So
> the system has to decide, every day, for every room type and every future
> night, what to charge — knowing demand is uncertain, competitors are moving,
> and the answer changes as check-in approaches.

Then the constraint that shaped the architecture:

> And it has to be **explainable**, because a price a revenue manager cannot
> interrogate is a price they will override — and an overridden pricing system is
> a switched-off pricing system.

That framing earns you the rest of the conversation. Everything else in this
document is downstream of those two paragraphs.

---

## The 60-second version

Use this when someone says "tell me about a project".

> It is an end-to-end dynamic pricing system for hotels. Competitor rates are
> scraped and streamed through Kafka into Postgres, a feature pipeline builds 30
> leakage-free features off the booking curve, two models — Prophet for calendar
> structure and a gradient booster for the current situation — produce a blended
> demand estimate with a confidence score, and a deterministic pricing engine
> turns that into a rate through five named adjustments and a fixed guardrail
> chain. FastAPI serves it, Streamlit explains it.
>
> The gradient booster is 55% better than a predict-the-mean baseline on a
> chronological holdout, and a request returns in about 28 milliseconds with the
> full arithmetic that produced the number.
>
> The part I'd actually want to talk about is the bugs — a couple of them looked
> completely fine and were badly wrong, and finding them is most of what I learned.

That last line is deliberate. It hands the interviewer a thread to pull, and the
thread leads somewhere you are strong.

---

## The 5-minute walkthrough

If you can share a screen, run `docker compose up -d` beforehand and drive the
dashboard. This order builds the story:

**1. Overview (20s)** — "Eight hotels, six Indian cities. RevPAR is the headline
number because occupancy alone rewards giving rooms away and ADR alone rewards an
empty hotel with one expensive suite sold."

**2. Data Ingestion (60s)** — the differentiator, so spend real time here. "This
is where competitor data comes from. It is a genuine scraper — it reads
robots.txt, parses HTML with CSS selectors, respects a rate limiter." Then the
honest part: *"Booking.com and Expedia both disallow the paths their scrapers
would need. My scraper honours robots.txt, so pointing at them correctly gets
blocked. Rather than delete the check to make a demo work, I ship a site that
genuinely grants permission."* Hit **Fetch now**, watch rates arrive.

**3. Dynamic Pricing (90s)** — the core. Price a night. Then the waterfall: "base
rate, then five individually-clamped adjustments, then the guardrails. Every bar
is named. There is no step where a model just decides." Point at any guardrail
that fired.

**4. Demand Forecast (45s)** — "the band matters more than the line. Interval
width becomes the confidence score, and confidence scales the whole multiplier
with a floor of 0.5. So an unsure model degrades towards the base rate instead of
making a confident mistake."

**5. Model Performance (45s)** — "always read the baseline column. R² of 0.77
means nothing until you know predicting the mean gets 0.00 here." Show
permutation importances: occupancy first, lead time second, their interaction
third — "which is the revenue-management story the whole system is built on, and
the model found it."

**6. AI Agent (60s)** — "every decision is already stored with its arithmetic;
what was missing was language." Ask it something. Then, unprompted, give the
boundary: *"it reads and simulates, it never prices. And that is enforced by an
allowlist in code, not by asking it nicely in a prompt."*

Leave Monitoring and Competitor Pricing for questions. Six minutes is already
longer than most people's attention for a demo.

---

## Numbers worth memorising

Only these. Quoting five accurate numbers beats gesturing at twenty.

| Number | What it is |
|---|---|
| **0.0644 vs 0.1430** | GBR MAE vs baseline MAE — **55% better** |
| **0.765** | GBR R² on the holdout (baseline: 0.000) |
| **78%** | Prophet's 80% interval coverage — calibrated, not decorative |
| **~28 ms** | p50 latency for a full priced room-night (p99 33 ms) |
| **893** | tests, ~55 seconds |

Two framings that make those numbers land:

- **"Against a baseline"** — always. A model is only as good as what it beats.
- **"Chronological holdout, never shuffled"** — say this before you are asked. A
  random split on time-series data leaks the future into training, and mentioning
  it first signals you know that.

---

## The five questions you will get

### "How do you know the model is actually good?"

Predict-the-mean baseline on every metric, and a chronological split — the most
recent 60 days, never shuffled. Then the shape argument: accuracy is best near
check-in (MAE 0.045 at 0 days) and degrades with horizon (0.081 at 60), which is
exactly what a *correct* model looks like here, because the booking curve has
already resolved most of the answer by check-in day. A model that was equally
accurate at every horizon would be suspicious, not impressive.

### "How do you prevent data leakage?"

Structurally, not by inspection. Booking data is stored on the **pickup grid** —
one row per (booking date × stay date) — so the booking curve is preserved and
every feature can be computed as of a specific point in time. Then a
counterfactual test: change only *future* bookings for a stay date, rebuild the
features, and assert nothing that claims to be as-of-today moved.

If they push: the train/serve parity seam is a single shared function,
`FeatureBuilder._add_derived_features`, called by both the training matrix builder
and the serving-row builder. Two implementations of "what are the features" is how
train/serve skew happens.

### "Why two models?"

They answer different questions. Prophet knows calendar structure — weekly shape,
trend, holidays — but nothing about *today*. The gradient booster knows the
current situation — 30 features including on-the-books occupancy, lead time, and
their interaction — but has no notion of a seasonal cycle. They are blended, both
scored on the **same** holdout so the blend weight has an actual basis, and the
blend is scaled by confidence.

### "Why isn't the pricing itself an ML model / an LLM?"

The strongest answer in the whole project. Pricing needs reproducibility,
auditability and a 28 ms budget; a language model gives up all three. Then the
part that shows you have thought past the obvious:

> The tempting middle position — let a model pick the multipliers and keep the
> guardrails — is worse than either extreme. Guardrails clamp the *output*; they
> cannot detect that the reasoning was wrong. A model that hallucinates "Diwali is
> next week" produces a price inside every band, confidently justified, and
> completely wrong. Guardrails bound damage. They do not create correctness.

### "Where does the AI agent fit, then?"

Where the data is already structured but the *language* is missing: explaining a
decision from the stored breakdown, triaging correlated monitoring alerts, and
(not built yet) reading events out of unstructured text — which is the only one
that would improve the model rather than the operator experience.

Then volunteer the enforcement, because it is the interesting bit: read-only is a
`(method, path)` allowlist checked before any request leaves, and `persist=False`
is hardcoded rather than exposed as a parameter the model could set. Same
reasoning as the `_CONSTRUCTION_TOKEN` in the guardrails module — make the
violating state unrepresentable rather than merely discouraged.

And then the boundary, unprompted: *"that allowlist is in-process. The complete
control would be an API credential with no write scope, and this project doesn't
have one because it ships no auth layer. That's a documented gap, not a solved
problem."* Naming your own limitation before they find it is worth more than the
stronger-sounding claim.

---

## Lead with a bug

If there is one move that separates a good interview from a forgettable one, it is
having a debugging story with a real *"and here is why it was hard to see"*.

Pick one. Do not tell all four.

### The best one: the model that passed every test and was 66% worse than nothing

> Prophet scored MAE 0.246 against a baseline of 0.148 — worse than predicting the
> mean, with R² of −6.5. And it passed every structural test I had: correct
> shapes, bounded intervals, clean serialisation.
>
> The cause was yearly seasonality fitted on less than one full cycle. With ~365
> days the Fourier term is unidentifiable, so it latched onto noise and
> extrapolated it confidently. I now disable yearly seasonality and holiday
> regressors below 730 days of history, with the measurements in the docstring so
> nobody "fixes" it back.
>
> The lesson I took: structural tests prove a model *runs*. Only a baseline proves
> it *works*. If I hadn't had one, I would have shipped a forecaster that was
> worse than an average.

That last paragraph is the whole point. Have a lesson, not just a war story.

### The subtle one: a column that looked perfectly plausible

> The gradient booster ranked occupancy below `weather_score`, which is nonsense —
> on check-in day the rooms on the books essentially *are* the answer. The booking
> curve turned out to be flat at ~0.48 occupancy at every horizon.
>
> `pd.merge_asof` returns a fresh `RangeIndex`, so restoring row order with
> `sort_index()` silently re-sorted by *position* and handed every row some other
> row's on-the-books total. The column still had the right range and the right
> mean — which is why 10 of the 11 tests in that class passed with the bug in
> place. The regression test needs *varying* horizons to catch it; the
> fixed-horizon version passes either way, because a stable sort is a no-op.
>
> After the fix: MAE 0.087 → 0.064, and occupancy became the top feature.

Use this one if the interviewer is a data person. "Right range, right mean, wrong
values" is a failure mode they will recognise and respect.

### The security-flavoured one: a robots.txt check that did nothing

> I had a test asserting the scraper fetches an allowed path. It passed. What I
> did not have was a test asserting it *refuses* a disallowed one — and when I
> wrote it, it failed.
>
> `urllib.robotparser` implements the original 1994 standard where the **first**
> matching rule wins, not Google's longest-match. My robots.txt listed `Allow: /`
> above `Disallow: /admin`, so the blanket allow matched first and the disallow
> beneath it was dead text. The scraper cheerfully fetched a path it had been told
> not to, and every "we honour robots.txt" claim in the codebase was false for
> that path.
>
> A test for the happy path proves nothing about a rule whose entire job is to
> say no.

### The one about a guard that could never fire

> I capped a scraping pass at 400 requests. The schema caps horizons at 12, and
> there are 8 hotels across 4 room types — a maximum of 384. The limit was
> unreachable; the only real protection was that nobody tried.
>
> The actual risk was never request count, it was wall-clock: the same 96 requests
> are instant with the rate limiter off and eight minutes at five seconds apart,
> and it's the second one that hangs a browser. So the guard is now on estimated
> duration. And there's a companion test asserting the same pass is *allowed* when
> it would be fast — otherwise the rejection test passes for the wrong reason and
> I'd never know.

---

## Things to say, and things not to

**Say:**

- "Against a baseline of…" — every time you quote a metric
- "Chronological split, never shuffled" — before they ask
- "It degrades rather than dies" — Kafka down, model missing, competitor data
  stale: each has a defined fallback and a lowered confidence, not a 500
- "I decided *not* to…" — the demo OTA instead of deleting a robots check, no
  MLflow, no vector database. Restraint reads as judgement.

**Don't say:**

- "It's production-ready." It isn't — no auth, single Kafka broker, RF 1. Say
  "production-shaped, and here's the gap list" instead. There is one in the
  README, and pointing at it is stronger than claiming it doesn't exist.
- "The model figures it out." Every adjustment here is named and clamped
  precisely so nobody has to say that.
- "99% accuracy." Nothing in this project is 99% anything, and a number that
  round in a pricing context signals leakage to anyone who has done this.

---

## If they ask what you'd do next

Have an ordered answer — it shows you know what is missing, which is different
from thinking nothing is.

1. **Two years of history.** One year cannot identify yearly seasonality, which
   is why Prophet's yearly term is off. This is the single change that most
   improves the forecast.
2. **Price elasticity.** The engine prices *to* forecast demand; it does not model
   how demand responds to the price it sets. That needs experimentation or an
   instrumental-variables approach — and saying so is more honest than pretending
   the current design closes the loop.
3. **Auth**, which also unlocks the agent's read-only guarantee at the network
   layer rather than in-process.
4. **The event-extraction agent** — the calendar is 19 hand-curated records today,
   and this is the one agent use that improves the model itself.

---

## Quick reference

| They ask | Point them at |
|---|---|
| The whole system in one read | [`docs/technical_overview.md`](technical_overview.md) |
| Overall design, ADRs | [`docs/architecture.md`](architecture.md) |
| End-to-end flows, diagrams | [`docs/technical_deep_dive.md`](technical_deep_dive.md) |
| Features, models, evaluation | [`docs/ml_pipeline.md`](ml_pipeline.md) |
| The agent argument in full | [`docs/ai_agent_design.md`](ai_agent_design.md) |
| Endpoints | [`docs/api.md`](api.md) or `/docs` live |
| Running it | [README → Setup](../README.md#setup) |
