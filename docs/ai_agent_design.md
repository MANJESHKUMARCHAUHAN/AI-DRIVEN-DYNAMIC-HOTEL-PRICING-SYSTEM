# Adding an AI Agent to This System

**Where an LLM agent genuinely belongs, where it must not go, and how to build the one that matters.**

---

## The short answer

This system already makes pricing decisions. It does not need an LLM to make them, and
putting one there would make the system worse — slower, non-deterministic, and impossible
to defend to a revenue manager who asks "why did you charge ₹9,400 last Tuesday?"

What the system *cannot* do today is **talk about** its decisions, **triage** its own
alerts, or **read anything that isn't already a number in a table**. Those are exactly the
jobs an LLM agent is good at.

So the rule that organises this entire document:

> **The agent reads, explains, and proposes. The deterministic engine decides.**

Everything below is a consequence of that one line.

---

## 1. Why the "where" question matters more than the "how"

Adding an agent is easy. Twenty lines of SDK code and a list of tools. The hard part —
and the part an interviewer will actually probe — is knowing which parts of a working
system should stay untouched.

Before deciding, look at what the system already guarantees:

| Guarantee | Where it lives | Would an LLM preserve it? |
|---|---|---|
| Same inputs → same price, every time | `pricing/pricing_engine.py` | **No.** Sampling is non-deterministic by construction. |
| Price is provably inside `[MIN_PRICE, MAX_PRICE]` | `pricing/guardrails.py` | **No.** A generated number has no proof attached. |
| Every decision has a step-by-step arithmetic breakdown | `PricingDecision.breakdown` | Partially — but the breakdown would be a *narrative*, not the calculation. |
| p50 latency ≈ 28 ms | measured, `POST /api/v1/pricing/predict` | **No.** One LLM call is 1–3 seconds. Two orders of magnitude. |
| The price object cannot be constructed outside the guardrail module | `_CONSTRUCTION_TOKEN` in `guardrails.py` | Irrelevant — an LLM emits text, not a `FinalPrice`. |

Four hard guarantees, four "no"s. That table *is* the argument. It is also, almost
verbatim, the answer to the interview question "would you use an LLM for the pricing
itself?"

---

## 2. Where an agent must NOT go

### 2.1 Not the price calculation

The pricing path is nine deterministic steps: demand forecast → adjustment multipliers →
confidence scaling → guardrails → `FinalPrice`. It is unit-tested, it is auditable, and
the arithmetic is the explanation.

Replacing it with an LLM breaks four things at once:

1. **Reproducibility.** Rerun the same room-night tomorrow and you may get a different
   number. There is nothing to regression-test against.
2. **Auditability.** "The model decided ₹9,400" is not an answer a revenue director,
   a franchise owner, or a competition regulator accepts. `base 7000 × 1.18 occupancy
   × 1.14 event × 0.97 competitor, capped at +15% daily change` is.
3. **Latency budget.** 28 ms → ~2 s. The dashboard's 30-night grid makes 30 of these.
4. **The type gate becomes theatre.** `FinalPrice` exists precisely so that no code path
   can serialise a price that did not pass through `guardrails.apply()`. An LLM in that
   path either bypasses the gate (guarantee gone) or is post-processed by it (so why is
   it there?).

There is a subtler trap worth naming, because it sounds reasonable in a design review:
*"let the LLM pick the multipliers, keep the guardrails."* This is worse than it looks.
The guardrails clamp the **output**; they cannot detect that the reasoning was wrong. An
LLM that hallucinates "Diwali is next week" produces a price that is perfectly inside
every band and completely wrong — and the audit trail will faithfully record a confident,
fluent, false justification. Guardrails bound damage; they do not create correctness.

### 2.2 Not the demand forecast

Prophet and the GradientBoostingRegressor beat the baseline by 39% and 55% MAE
respectively, run in milliseconds, cost nothing per call, and expose permutation
importances. An LLM asked to predict a number between 0 and 1 from thirty features would
be worse at all four.

Language models are not regressors. This is not a limitation to work around; it is a
statement about what the tool is for.

### 2.3 Not inside the request path

Anything an end user waits on stays synchronous and deterministic. Every agent use case
below is either **batch** (runs nightly, no one is waiting) or **conversational** (the
user is explicitly asking a question and expects to wait a moment). Nothing sits between
`POST /pricing/predict` and its response.

### 2.4 Not writing prices directly

The agent may call `POST /pricing/predict` with `persist=false` as often as it likes —
that is a simulation, and the schema already supports it precisely so what-ifs never
pollute the audit trail. It may **never** hold a credential that lets it write a rate to
the PMS or channel manager. Not because it would go rogue, but because a system where a
non-deterministic component can move money has no defensible failure story.

---

## 3. Where an agent genuinely earns its keep

Five candidates, ranked by value-to-effort. Each one shares a shape: **the data already
exists and is structured; what's missing is language.**

---

### 3.1 Pricing analyst over the audit trail — *build this first*

**The problem today.** Every decision is stored in `pricing_decisions` with a full
`breakdown` JSON: the base rate, each adjustment and its size, which guardrails fired,
the demand score, the confidence. It is complete. It is also unreadable unless you are
willing to write SQL and squint at JSON.

A revenue manager's real question is:

> *"Why did we drop the price on H004 suites last Tuesday when the city was busy?"*

Answering that today means: query the decision row, decode the breakdown, cross-reference
the competitor table, check whether an event was active, check whether the daily-change
cap clipped the result. Five minutes for someone who knows the schema. Impossible for
someone who doesn't.

**What the agent does.** It has read-only tools mapping onto endpoints that already
exist, so it can chain: fetch the decision → notice the competitor band fired → fetch the
competitor rates for that night → notice two competitors dumped inventory → fetch the
forecast → confirm demand was actually soft despite the event → answer in three sentences,
citing numbers it retrieved rather than numbers it invented.

**Why an LLM is right here.** The task is genuinely open-ended — you cannot enumerate the
questions in advance, which is the standard test for "is this an agent or a workflow?"
The reasoning is *retrieval and narration over structured facts*, which is the thing
language models are unambiguously good at. And it is read-only, so the worst failure is
a wrong sentence, not a wrong price.

**Failure mode.** Confident misattribution — "the price dropped because of low occupancy"
when the cap actually clipped it. Mitigated by making the guardrail list an explicit tool
result and instructing the model to name the specific rule identifier from
`guardrails_applied`, never a generic cause.

---

### 3.2 Monitoring triage copilot

**The problem today.** `scripts/monitor.py` writes `data/monitoring_report.json` — nine
data-quality checks, PSI drift per feature, prediction statistics, each with a severity.
On a bad night, six things go amber at once. Five of them are the same root cause.

Alert fatigue is not a data problem. It is a *summarisation* problem.

**What the agent does.** Reads the report plus the last N pricing decisions plus recent
log lines, and writes an ops note: what is actually wrong, which alerts are downstream of
which, what to check first, and — importantly — which alerts to ignore tonight and why.

Concretely, the kind of link it can make that a threshold cannot: *competitor coverage
dropped to 61% AND prediction variance collapsed AND three features show PSI drift* →
one cause, the producer died four hours ago, everything else is a shadow of it.

**Why an LLM is right here.** Correlating heterogeneous signals into a causal story is
not expressible as a threshold. It runs nightly, so latency is free. And its output is a
recommendation to a human, not an action.

**Note the honest caveat.** The monitoring module already contains a
`seasonality_caveat` check, because drift and seasonality are genuinely confounded in
this domain. The agent should be *told about* that caveat in its system prompt, not
expected to rediscover it. Domain knowledge you already have belongs in the prompt, not
in the model's guesswork.

---

### 3.3 Event intelligence — the one that improves the model

**The problem today.** `features/calendars.py` holds `_CITY_EVENTS`: nineteen
hand-curated `CityEvent` records with fixed `(month, day)` windows and a hand-assigned
`intensity`. Sunburn, JLF, Aero India, wedding seasons.

It is a good calendar. It is also **static and manual**, and that is a real modelling
limitation, not a cosmetic one. Real demand in these cities is moved by things that are
announced three weeks out and appear nowhere in a hardcoded tuple: a conference
relocating, a concert, a cricket fixture change, a political rally, an airport closure.
`local_event_score` can only ever be as good as the calendar behind it.

**What the agent does.** A nightly batch job reads unstructured sources — venue
calendars, city tourism feeds, news — and extracts structured candidates matching the
schema that already exists:

```python
CityEvent(name=..., city=..., start=(month, day), end=(month, day), intensity=...)
```

Output goes into a **review queue**, not into the calendar. A human approves. Approved
events flow into `event_score()` exactly like the hardcoded ones.

**Why an LLM is right here.** This is the *canonical* LLM task: unstructured text →
validated structured record. No regex, no scraper, and no amount of feature engineering
gets you "the Bengaluru Tech Summit moved to a different venue and expanded to five days."
It is also the only item on this list that makes the **pricing model itself** better
rather than making the system easier to operate.

**Constraint that makes it safe.** Extraction is forced through the existing frozen
dataclass via structured outputs, so a malformed event cannot reach the calendar. And
because it is queued for approval, a hallucinated festival costs a human ten seconds of
review, not a mispriced weekend.

---

### 3.4 Scraper self-healing

**The problem today.** `BookingScraper` and `ExpediaScraper` exist as interfaces with
candidate CSS selector lists, disabled by default (ToS and robots — the
`SyntheticCompetitorGenerator` is the working default). When enabled, the failure mode of
any real scraper is: the site redesigns, every selector misses, `ScraperParseError` fires,
and a human spends an afternoon in devtools.

**What the agent does.** On `ScraperParseError`, it receives the fetched HTML and the
expected output schema, and proposes updated selectors. A human reviews the diff, and the
existing validator confirms the extracted prices are plausible before anything is
committed.

**Why it's ranked fourth.** High demo appeal, genuinely useful in production, but it only
matters if you turn the real scrapers on — which this project deliberately does not.
Worth designing, worth mentioning in an interview; not worth building first.

---

### 3.5 What-if assistant for revenue managers

**The problem today.** "What happens to RevPAR if I cap the daily change at 10% instead of
15%?" is answerable — set the env var, rerun, compare — but only by someone comfortable
with the codebase.

**What the agent does.** Drives the existing endpoints with `persist=false` across a
parameter sweep, aggregates, and reports the tradeoff. All simulation; the audit trail is
untouched by construction, because that flag already exists for exactly this purpose.

**Why it's ranked fifth.** It is the most impressive in a demo and the least necessary.
Build it after the analyst, because it reuses the same tools.

---

## 4. Architecture

The agent sits **beside** the API and consumes it exactly as the Streamlit dashboard
does. It gets no database credentials, no direct model access, and no privileged path.

```
                    ┌──────────────────────────────────────────┐
                    │  agent/                                  │
                    │    tools.py    typed wrappers over REST   │
                    │    analyst.py  the conversational loop    │
                    │    events.py   nightly extraction batch   │
                    │    triage.py   nightly monitoring digest  │
                    └───────────────┬──────────────────────────┘
                                    │ HTTP, read-only + persist=false
                                    │ (same surface as the dashboard)
                    ┌───────────────▼──────────────────────────┐
                    │  FastAPI  —  10 endpoints, unchanged      │
                    └───────────────┬──────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
   PricingEngine              ModelRegistry               PostgreSQL
   (deterministic)          (Prophet + GBR)          (audit trail, features)
```

Three properties fall out of this shape, and each is worth stating explicitly in a design
review:

**The API is already the tool surface.** Twelve endpoints, all typed with Pydantic, all
described by the OpenAPI schema FastAPI generates for free. Writing tool definitions is
transcription, not design. That is not luck — it is what having a clean API boundary buys
you.

**Read-only is structural, not policy.** The agent's HTTP client checks a fixed allowlist
before every request, so it is not "instructed not to write"; it *cannot*. A
prompt-injected agent — say, one that read a malicious "event announcement" in section
3.3 — reaches the same branch as a well-behaved one. Section 6 is precise about where
that guarantee's boundary sits.

**Removing the agent breaks nothing.** Delete `ai_agent/` and the pricing system is exactly
what it was — `anthropic` is not even a runtime dependency, and the dashboard page
renders install instructions rather than failing. That is the correct dependency
direction for anything non-deterministic, and it is verified by a test that pins the
page to the SDK-absent path.

---

## 5. Implementation: the pricing analyst

**This section is now describing shipped code**, not a proposal. Read
[`ai_agent/tools.py`](../ai_agent/tools.py) and [`ai_agent/agent.py`](../ai_agent/agent.py)
for the current source; what follows is the reasoning behind the two decisions in
it that are easy to get wrong.

Python SDK, Claude Opus 4.8, adaptive thinking, `effort: "medium"`. The SDK's
tool runner drives the agentic loop.

### 5.1 The two details that carry the design

**Docstrings are the tool-selection prompt.** The model picks tools from them, so
each says *when* to call it, not just what it returns — recent Opus models reach
for tools conservatively, and a prescriptive "call this whenever…" measurably
improves should-call rate. `get_demand_forecast` even encodes a domain fact:

```python
def get_demand_forecast(hotel_id, room_type="deluxe", horizon_days=14) -> str:
    """Demand forecast for the next N nights, with prediction intervals.

    Call this to check whether a price reflected genuinely soft demand or
    something else. Read the interval width, not just the point forecast: low
    confidence scales the whole pricing multiplier back towards the base rate
    (floor 0.5), so a wide interval is frequently the entire explanation for a
    price that looks unresponsive to obvious demand.
    """
```

Without that last sentence the model has to *infer* why a price ignored obvious
demand, and inferring is exactly where it invents.

**`persist` is not a parameter.** It is hardcoded, so there is no argument the
model can set:

```python
body = {
    "hotel_id": hotel_id,
    "room_type": room_type,
    "check_in_date": check_in_date,
    # Not a parameter. If the model could set this, the allowlist would be the
    # only thing between an explanation and a written price.
    "persist": False,
}
```

### 5.2 The allowlist is the real guarantee

Underneath the tools sits a fixed set of `(method, path-prefix)` pairs, checked
before any request leaves:

```python
_ALLOWED_CALLS = (
    ("GET",  "/hotels"),
    ("GET",  "/pricing/"),
    ("GET",  "/competitors/"),
    ("GET",  "/forecast/"),
    ("GET",  "/ingestion/status"),
    ("POST", "/pricing/predict"),      # simulation only; persist is hardcoded
)
```

Anything else raises `ToolCallDenied`. Adding a write tool does not quietly
start working — it fails until somebody deliberately edits that tuple, which is a
diff a reviewer sees. `tests/test_ai_agent.py` asserts both directions, including
that `GET /pricing/` being allowed does not imply `POST /pricing/`.

Note this raises rather than returning an error string, unlike every other
failure in the module. A denied call is a bug in our code, not a condition the
model should be asked to reason about.

### 5.3 What a real exchange looks like

```
User:  Why did we drop the price on H004 suites last Tuesday? The city was busy.

  → get_pricing_history(H004, suite, 2026-08-18, 2026-08-18)
  → get_competitor_rates(H004, 2026-08-18)
  → get_demand_forecast(H004, suite, 7)

Agent: The daily-change cap did it, not soft demand. The engine wanted ₹11,240
       (occupancy 0.81 and the event score both pushed up), but Monday's price
       was ₹9,780 and MAX_DAILY_CHANGE_PERCENT is 0.15, so the cap clipped it
       to ₹11,247 — then the competitor upper band pulled it to ₹9,410 because
       the three tracked competitors averaged ₹8,900.

       guardrails_applied: ["daily_change_cap", "competitor_upper_band"]

       The city being busy is in the number; it was just outranked. If the
       competitor rates look stale to you, two of the three were last observed
       31 hours before the decision.
```

Note what the agent did *not* do: pick a price. It explained one that a deterministic
engine had already produced, and it pointed at a data-quality issue a threshold check
would have missed.

---

## 6. Guardrails for the agent itself

The pricing engine has guardrails. So must the agent — and for the same reason: a
constraint you can only state in a prompt is not a constraint.

| Risk | Control | Where enforced |
|---|---|---|
| Fabricated numbers | Every figure must come from a tool result | Prompt + human-visible tool trace |
| Writing a real price | `_ALLOWED_CALLS` allowlist; `persist=False` hardcoded | **`ai_agent/tools.py`** — in-process, before any request |
| Prompt injection via scraped event text | Extraction is structured-output-only, output goes to a review queue | Schema validation + human approval |
| Runaway loop | `MAX_ITERATIONS = 12` per question, `max_tokens` cap, usage returned per turn | `analyst.py` |
| Stale answers | Tools always hit the live API; nothing is cached across turns | `tools.py` |
| Unverifiable claims | The tool trace is rendered next to every answer | AI Agent page |

The row that matters is the second one. "The system prompt says the agent must not write
prices" is a *suggestion* to a probabilistic system; the allowlist is a branch that runs
before the socket opens. This mirrors the `_CONSTRUCTION_TOKEN` decision in
`guardrails.py`: both times the answer to "how do we guarantee X?" was to make the
violating state unrepresentable rather than merely discouraged.

### Being precise about how strong that guarantee is

It is worth stating exactly what the allowlist does and does not buy, because
overclaiming here is the kind of thing that gets found out in review.

**What it does:** binds this code. No sequence of model outputs reaches a write
endpoint, because the model's only route to the network is `_request`, and
`_request` checks the tuple first. Prompt injection through a scraped event
description hits the same branch as a well-behaved tool call.

**What it does not:** bind a *different* process holding the same API URL. It is
an application-layer control, not a network one. The complete version is an API
credential with no write scope — and **this project does not have one, because it
ships no authentication layer at all** (see Known limitations in the README).

So the honest phrasing is: *the agent cannot write, and the reason is a code path
rather than a credential.* That is a real guarantee with a real boundary, and
naming the boundary is worth more than a stronger-sounding sentence that a
reviewer could puncture in one question.

---

## 7. What it costs

Claude Opus 4.8 is $5 / MTok input, $25 / MTok output.

| Use case | Cadence | Tokens/run (approx.) | Monthly |
|---|---|---|---|
| Analyst | ~40 questions/day, ~12k in / 1.5k out | 30 runs/day × 30 | **~$95** |
| Monitoring triage | nightly, ~25k in / 2k out | 30 runs | **~$5** |
| Event extraction | nightly, ~60k in / 4k out | 30 runs | **~$12** |

Roughly **$110/month** for all three, against a system whose pricing decisions move
five-figure daily revenue. Two levers if that matters: the system prompt is stable and
large, so a `cache_control` breakpoint on it drops repeat input cost by ~90%; and the
overnight batch jobs are latency-insensitive, so the Batches API halves their cost
outright.

Worth stating the counterfactual too: putting an LLM in the *pricing* path would be
roughly 40 requests/day × 30 hotels × 30 nights of forecast — hundreds of thousands of
calls a month, for a job two trained models do for free in milliseconds. The cost
argument and the correctness argument point the same way, which is usually a sign the
design is right.

---

## 8. Interview questions this design answers

**"Would you use an LLM for the pricing itself?"**
No, and the reason is structural rather than aesthetic. Pricing needs reproducibility,
auditability, and a 28 ms budget. An LLM gives up all three. There is also a trap in the
compromise position — letting the model pick multipliers and keeping the guardrails.
Guardrails clamp outputs; they cannot detect wrong reasoning. A hallucinated festival
produces an in-band, confidently-justified, wrong price.

**"So where does an agent add value?"**
Where the data is already structured but the *language* is missing: explaining decisions
from the audit trail, triaging correlated alerts, and extracting events from unstructured
text. That last one is the only one that improves the model rather than the operator
experience — the event calendar is nineteen hardcoded records today.

**"How do you stop it hallucinating a price?"**
Layered. The prompt says cite only tool results. The tool trace is visible so a wrong
citation is checkable. And structurally, the agent's credential cannot write and
`persist=False` is hardcoded rather than exposed as a parameter — so the worst outcome is
a wrong sentence, never a wrong rate.

**"How does the agent get its tools?"**
It calls the same ten REST endpoints the dashboard calls. They are already typed with
Pydantic and already described by OpenAPI, so the tool definitions are transcription. A
clean API boundary is what makes the agent cheap to add — the agent is a consumer, not a
new subsystem.

**"What's the failure mode?"**
Confident misattribution: saying the price fell because of low occupancy when the
daily-change cap actually clipped it. Mitigation is to surface `guardrails_applied` as an
explicit tool result and require the model to name the specific rule identifier rather
than describe a cause in its own words.

**"Agent or workflow?"**
Analyst: agent — the questions cannot be enumerated in advance and the tool sequence
depends on what the previous call returned. Event extraction: closer to a workflow — one
structured-output call per source, orchestrated by code, no model-driven control flow.
Reaching for the agent tier when a workflow suffices is the most common way to overpay.

**"What would you build first?"**
The analyst. Highest value per line of code, entirely read-only, and it forces you to
build the tool wrappers the other four use.

---

## 9. Build order

**Shipped:**

1. **`ai_agent/tools.py`** — six read-only wrappers behind `_ALLOWED_CALLS`, `persist=False`
   hardcoded. 26 tests, most of them asserting what it *cannot* do.
2. **`ai_agent/agent.py`** — the conversational loop, exposed as dashboard page 8 with the
   tool trace rendered next to every answer.
3. **`ai_agent/triage.py`** — `make triage` summarises `data/monitoring_report.json` into an
   ops note. Batch; nobody is waiting.

**Not built:**

4. **`ai_agent/events.py`** — structured extraction into `CityEvent`, queued for approval.
   The highest-value remaining item, because it is the only one that improves the
   *model* rather than the operator experience. Do not wire it into `_CITY_EVENTS` until
   a human has approved a full cycle.
5. Scraper self-healing — worth doing once the third-party scrapers are enabled, which
   this project deliberately does not do. Note the demo OTA's `layout=v2` switch already
   produces the exact `ScraperParseError` such an agent would trigger on, so there is a
   reproducible test case waiting for it.
6. The what-if assistant — reuses the tools already built; mostly a prompt and a page.

Steps 1–3 took an afternoon. Step 4 is where the next real gain is.

---

## Appendix: the one-line summary

> The deterministic engine decides prices, because prices must be reproducible,
> auditable, and fast. The agent handles the parts that need language — explaining
> decisions, triaging alerts, and reading the unstructured world the feature table
> cannot see. It reads and it simulates; it never writes, and that restriction is
> a code path rather than a prompt.

See also: [`technical_deep_dive.md`](technical_deep_dive.md) for the full system flow,
and [`ml_pipeline.md`](ml_pipeline.md) for why the models are what they are.
