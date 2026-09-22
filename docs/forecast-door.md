# The forecast door — a definition to take up in glossql

A proposal, not built. It lives in this repo because glossql keeps one
normative document (`SPEC.md`) and no satellite design notes, and because
every fenced example under glossql's `docs/` must parse and plan — which a
door that does not exist cannot. From a session in the glossql folder, read
it as `../glosskernels/docs/forecast-door.md`. It is written to become, over
there, one issue (what it buys, what stands, done-when) and the forks of a
SPEC §9 question. The examples below are fenced as `text` on purpose.

## What it buys

An agent on the MCP door can ask what a metric will be, not only whether
its last months were surprising. It declares a forecast — which metric, how
far out, from how much history, with which known-future inputs and why —
reads the future months as bands, and reads how that same declaration would
have done on the months already recorded. Every variant it tries leaves a
record. Nothing about the model, the features or the calibration is the
agent's to write.

## What stands

In glossql:

- The pattern. An analysis is declared and then read: an aspect with an
  `x-kind`, glossed with a JSON body whose every entry carries its `basis`,
  served by an operation-named door — `whatif.<scenario>()`,
  `misfit.<frame>()` (`SPEC.md` §9; `skills/glossql-metrics/references/doors.md`).
  Doors take no arguments: "settings are context, never arguments"
  (`grammar.ebnf:174-181`; `crates/session/src/reads.rs:899-928`). The
  registry of doors is one list (`reads.rs:405-437`).
- The series. `metric_series()` serves every grounded metric at its judged
  resolution, at the read's pin.
- The record. A measurement lands keyed by pin, and old pins' rows stay as
  the drift record (`crates/glossary/src/store.rs:351-368`). `band_points()`
  flattens the judged `metric_bands` body.
- The walk. `metric_band_walk` owns a one-step protocol over the six latest
  months: features index, month of year, last month, trailing three-month
  mean, the month a year before (`crates/session/src/search/bands.rs:372-394`);
  `ALPHAS`, `MIN_TRAIN = 5`, `MAX_WALK = 6` are constants (`bands.rs:244-246`).
- What does not exist: any future period. The what-if replays recorded
  months only (`whatif.rs:375-384`); the walk calls months that have landed.
  No query names extra feature columns for the walk, and there is no
  calendar or holiday table.

In the kernel (this repo, branch `harness`):

- `POST /bands` answers many reads in one request: `voices` (`tabicl`,
  `chronos2`, `seasonal_naive`) each on its own beside their `blend`;
  `horizon` per row; `window` (1 a month, 12 a trailing annual total asked
  for as its own series); `pit_history` by voice, read through a shipped
  default record. Covariates need nothing new for TabICL: a column is a
  column.
- The graded protocol for a month `h` out is direct, not recursive
  (`harness/voices.py: recipe`): the label is the month `h` ahead, the lag
  and the trailing mean step back to the last month known at the origin,
  and the year-ago month stays while it is still known (`h <= 12`). An
  annual total is asked for as its own series, never summed from monthly
  bands (summed bands covered 94–99% at a nominal 80).
- Measured: the three-voice blend is at or under the seasonal-naive floor
  at every horizon on tourism_monthly and hospital. On the three panels the
  default records were not built from, the blend's 80% bands hold 79–95% of
  months one and twelve out, and 75–96% of annual totals.

## The declaration

One FACT aspect per forecast, as one FACT aspect is one scenario.

```text
DECLARE ASPECT demand_outlook WITH $${
  "title": "Orders, twelve months out",
  "x-kind": "forecast",
  "type": "object", "required": ["metrics", "horizon"],
  "properties": { ... }
}$$ AS FACT ON DATASET;

GLOSS demand_outlook ON ops AS $${
  "metrics": ["order_count", "throughput"],
  "horizon": 12,
  "window": 36,
  "total": true,
  "covariates": [
    {"metric": "planned_price", "basis": "list price moves demand down; the plan is set a year ahead"},
    {"calendar": "working_days", "basis": "orders are taken on working days only"}
  ]
}$$;
```

| field | meaning | absent |
|---|---|---|
| `metrics` | grounded metrics, by name | required |
| `horizon` | months past the last whole month, 1–24 | required |
| `window` | how many months each read conditions on. A fixed window also gives every read of every metric one table shape, which is what lets the kernel answer them in one pass | all history |
| `total` | also the total over the horizon, asked for as its own series (the trailing sum, read `horizon` out, `window` = `horizon` to the kernel) | false |
| `covariates[].metric` | a grounded metric whose periods are already recorded through the last forecast month — a plan, a price list, a promotion calendar | none |
| `covariates[].calendar` | a shipped monthly series (`working_days`, `holidays`), for the region the workspace's `conventions` name | none |
| `covariates[].lag` | for a series recorded only up to now: use its value this many months before the month called; at least `horizon` | — |
| `covariates[].basis` | the mechanism and the expected direction, stated before any number is read | required per covariate |

## The reads

**`forecast.<name>()`** — the future months.

```text
SELECT metric, month, span, voice, p05, p10, p50, p90, p95, basis
FROM forecast.demand_outlook()
WHERE voice = 'blend' ORDER BY metric, month
```

- One row per metric, month and voice; `voice` is `blend` or a voice's
  name; `span` is `month`, or `total` for the row that carries the
  horizon's total. Sweeps are `WHERE` clauses.
- Bands are calibrated: the door sends the PITs this declaration's
  backtest has landed, per voice and per horizon, as `pit_history`, and the
  kernel weighs them with its default record.
- Judgment rides `basis`, as in the what-if: a metric with too little
  history, a covariate that does not reach the last forecast month
  ("planned_price ends 2027-03; the forecast asks through 2027-12"), a
  past-only series with no `lag`, a `total` asked of a stock (a level has
  no sum over months; its year is its last month) — each is a refusal row
  with its reason, and the other metrics are served.
- The newest month, where partial, is not an origin: the origin is the last
  whole month (the walk's `partial` rule).

**The backtest** — a measurement, `forecast_walk`, run bare like
`metric_bands` (`SELECT forecast_walk() FROM ops`), over every declared
forecast: the same declaration, called from past origins whose months have
since landed, at horizons 1, 3, 6 and `horizon`, and the total. Per
forecast, metric, horizon and voice it reports the quantile loss, the same
loss for seasonal-naive (the floor), the share of actuals inside the 80%
band, how many origins stand behind the numbers, and each point's PIT. It
lands keyed by pin, so every variant tried stays on the record.
`forecast_points()` flattens it, as `band_points()` does the walk.

Two forecasts that differ in one covariate are compared by reading both out
of `forecast_points()` — per metric and pooled over metrics, on the same
origins.

## What the door enforces, and what the skill teaches

The door:

- No future value reaches a training row: the direct protocol above, fills
  from training rows only, covariate availability checked at the pin.
- A covariate without a `basis` does not parse as a forecast.
- Numbers come from the kernel and the replayed record only; an agent has
  no way to edit a band.
- Every evaluated variant is a row in the record, with its origins counted.

The skill (defined below):

- Read the floor first: a forecast that does not beat seasonal-naive on the
  backtest is served with that said, and the blend is the default reading.
- State the mechanism before adding a covariate; keep it only if it helps
  across metrics and origins, not on one metric's mean. With 24–60 months
  there are six to twelve independent origins: the more variants tried, the
  less a win means, and the record shows how many were.
- Below about 36 months, calendar covariates only.
- Known-future inputs only. Weather beyond ten days is the season, which the
  month-of-year feature already carries; its defensible uses are
  normalising history and what-ifs. It would arrive as an imported dataset
  like any other, its snapshots its vintages; the paid key stays with what
  fetches it.

## The skill

Where an agent learns this is a product skill, as it learns apps and
functions: `skills/glossql-forecast/` — a `SKILL.md` and its references,
embedded in the server and served on the door as `skill://` resources and
as a prompt. Its own skill, not a page of `glossql-metrics`: a forecast is
asked for, it is not one of the seven goals a workspace is done by, and its
description has to load on other words — outlook, plan, budget, next
year. `glossql-metrics/references/doors.md` gains three lines pointing to
it, beside the what-if and the sample door.

It teaches judgment and no mechanics. The agent writes declarations and
reads; it never writes model code, feature code or a backtest loop — those
are the door's, which is what keeps a forecast reproducible and a backtest
free of leaks whoever asked. Where the procedure can ride the record it
does, not the prose:

- `workspace_next` gains a `forecasts` surface beside `scenarios` and
  `samples` — how many stand, how many are declared without a body, and the
  act that writes one.
- The `next` routes carry the order: a forecast glossed and not walked →
  run `forecast_walk`; walked → read `forecast.<name>()`; walked and over
  the floor on most of its metrics → say so before any future month is read.

What is left to the page is what no record can decide:

1. **The ask.** Which metric, how far, months or the year's total, and what
   decision it serves. A flow has a total; a stock has a level. Where the
   human has not said, ask — the door's question round exists for it.
2. **Whether to forecast at all.** The metric is grounded, applicable and
   monthly; the bands walk ran and no band is red without a ruling — a
   forecast of a metric with an open data problem forecasts the problem.
   Under about eighteen months, say the history is too short and serve the
   walk's bands instead.
3. **The first declaration is the smallest.** Metrics and horizon, nothing
   else. Walk it. Read the floor before the future.
4. **The window.** All history unless the record says the series changed —
   a ruling, a re-grounding, a regime the human names. The basis is that
   event, never a better backtest.
5. **A covariate.** Mechanism and direction first, in the `basis`. Known
   through the last forecast month, or lagged. One at a time, as a second
   forecast beside the first; kept when it helps across metrics and
   origins. How many variants the record holds is part of the answer.
6. **What the human knows and history does not.** A price change, a lost
   customer, a plant closing: never an adjustment to a band. It is a
   covariate if it is a series, a scenario if it is a lever, and otherwise
   a sentence beside the forecast saying what it does not include.
7. **The read-back.** The band before the median; the 80% band by default;
   one sentence from the backtest — how often that band held, and whether
   the forecast beat last year's month. Then what the forecast is blind to.
8. **Outside data.** A source like any other: landed, structured, grounded
   as a metric, then named as a covariate. The calendar first; the
   customer's own plans next; weather to normalise history or to ask a
   what-if, not to call next spring.

References, each short: `covariates.md` (availability, lags, the calendar,
the comparison read), `read-back.md` (the sentences, with a worked example
off the fixture). Every fenced example in a product skill must parse and
plan, so the skill lands with the door, not before it.

## Forks for the project lead

1. **Declared only, or a bare default too.** A forecast with only
   `metrics` and `horizon` is already minimal. A bare `forecast()` over
   every grounded metric would mirror `metric_bands()`, at the price of a
   second way to ask.
2. **Covariates as grounded metrics, or as raw columns.** The what-if names
   columns; a forecast needs a monthly series, and a metric's grounding
   already says how to aggregate it. Proposed: metrics.
3. **`window` on the walk too.** The bands walk conditions on all history;
   a declared window would change its graded answers slightly and let the
   kernel batch it. Separable from this door.
4. **Voices as rows or as columns.** Rows keep the relation narrow and the
   blend a filter.
5. **The calendar's home.** A shipped function over a region named in
   `conventions`, offline and deterministic; its worth at monthly grain is
   unmeasured and is each dataset's backtest to show.

Decided by the project lead: the skill is its own, `skills/glossql-forecast`,
to be extended over time; the `next` routes carry walk-before-read and the
floor said before the future, and no more; and the forecast door comes
after the metric foundation (the seven goals) stands.

## What the kernel still owes

- Chronos-2 sees no covariates: `/bands` passes it the series only. Until a
  read can carry covariate series for it, a covariate moves TabICL's voice
  and, through it, the blend.
- Default records exist for months and for twelve-month totals; a total
  over another span is read through the twelve-month one.
- One record per request means one request per horizon; the owner batches
  them. Histories per read would make a projection one request.
- The windowed recipe and a covariate admission have not been graded in the
  harness; both should be before the door ships.

## Order, and done when

1. glossql moves its existing doors to the kernel's API (`/bands`,
   `/misfit`), answers unchanged.
2. `forecast_walk` and `forecast_points()`: the backtest first, because the
   future read is calibrated from it.
3. `forecast.<name>()`, without covariates — and `skills/glossql-forecast`
   with it, the `forecasts` surface and the `next` routes.
4. Covariates: metrics first, the calendar after; `covariates.md` with them.

Done when a fixture dataset declares a forecast, reads twelve future months
and a total with a `basis` on every row, reads its backtest against the
floor, is refused by name for a covariate that ends too early, and a second
variant of the declaration shows beside the first in the record.
