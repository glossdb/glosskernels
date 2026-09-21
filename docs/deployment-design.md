# glosskernels — the first deployment

One kernel service on one GPU, shared by every tenant of the SaaS, serving
the reads the product leans on — **bands, what-if, projections**, and the
misfit read — at cube scale.

**What belongs in this service, and what does not.** It is shared by every
tenant, so it carries only what needs the model, and the statistics that
read the model's own output — calibration and blending happen here and
nowhere else. Everything the caller can do stays with the caller:
assembling contexts, building feature rows, aggregating PIT histories. A
request is numbers and a key.

The repo is open (Apache-2.0); it wraps open models. What is private is
deployment — keys, the key→tenant map, quotas, infrastructure — and lives
in `glossdeploy`.

## The API

| route | what it answers |
|---|---|
| `POST /bands` | quantiles for query rows, given context rows |
| `POST /misfit` | how well each row fits its frame; per column on request |
| `GET /healthz` | `status`, `device`, `loaded` |

JSON, matrices as nested lists, `null` for NaN; a refusal is `{"error": …}`
with a 4xx. A large context may be sent as Arrow IPC instead.

### `/bands`

Every band read is the same operation, so it is one route: a walk point is
one read of one row with its actual; a walk is many reads; a what-if or a
projection is a read of many rows.

```json
{
  "alphas": [0.05, 0.10, 0.50, 0.90, 0.95],
  "members": 1,
  "pit_history": [0, 3, 1, "… 100 counts"],
  "voices": ["tabicl", "chronos2"],
  "reads": [
    {"train_x": [[…]], "train_y": […], "test_x": [[…]],
     "actual": [13.2, null], "salt": [8812, 8813],
     "history": […], "cache": true}
  ]
}
```

| field | meaning |
|---|---|
| `members` | 1 (default) is the pinned member — the walk; more is the ensemble of that size — what-if over sparse grids |
| `reads[].train_x`, `train_y`, `test_x` | the context rows and the query rows |
| `reads[].actual` | per query row, where one has landed: its `pit` comes back, always against the raw answer |
| `reads[].salt` | an integer naming each point (a hash of metric and month); places a tied actual repeatably |
| `pit_history` | 100 counts of past PITs per hundredth (zeros for a caller with none yet): `quantiles` are then read through that record and the kernel's default one, and `raw` carries the model's own answer |
| `reads[].cache` | keep this context; the answer carries its `context` id, and a later read sends `"context": "<id>"` in place of the rows |
| `voices`, `reads[].history`, `reads[].horizon`, `season` | `tabicl` and any of `chronos2`, `seasonal_naive`: each answers on its own under `voices`, beside their `blend` (the level-by-level mean). The series voices read `history` — the series up to the origin — `horizon` periods out per test row (1 where left out). `pit_history` is then an object of histories by voice, `blend` among them — one record per request, so a projection is asked a horizon per request. The blend is of the voices as read, each through its record |
| `window` | how many periods each value of the series sums: 1 (the default), a month; 12, a trailing annual total asked for as its own series. It picks the default record a history is weighed with |

The answer is `{"reads": [{"quantiles", "raw"?, "pit"?, "context"?, "support"?}]}`, or
with `voices`, `{"reads": [{"voices": {name: {"quantiles", "raw"?, "pit"?}}, "blend": {…}, "context"?}]}`.
`support` says, per query row, whether each feature lies inside what the
context has seen — a what-if is honest inside (its median 0.06 of the truth
off, against replay's 0.15) and must be flagged outside.

An annual total is asked for as its own series, never summed from monthly
bands (summed bands covered 94–99% at a nominal 80).

### `/misfit`

`{"x": [[…]], "columns": true}` → `scores` per row (log density, higher
fits better) and, with `columns`, per row × column: which cell made the row
improbable. A cell's number is its column's conditional log density,
averaged over the orderings as the score is, so a row's cells sum to its
score; nothing more is computed — the read already fits every conditional,
and this returns them unsummed.

`"folds": n` scores each row from a context it is not in (the frame split
in n, each part scored from the rest). A frame that scores itself lets a
row find itself in the context. A value far outside its column is named
either way; a value ordinary for its column and wrong for its row is not —
planted sign flips in 200 × 6 frames had their cells ranked 9–344 of 1,200
self-fit, 4–77 over two folds, 2–43 over five — at about that many times
the fits. A broken relation implicates both of its ends: the chain rule
charges whichever column comes later in an ordering.

## The context cache

**Per tenant, across requests. Never shared between tenants, never per
request.** The weights are shared by everyone; a context is a customer's
data, so it belongs to one tenant — and within that tenant every request
reuses it. Built (`contexts.py`, `Kernels.answer`): on an L4 a 20k-row,
8-member context builds in 5.1 s and then answers in 0.085 s, against
4.5 s for every read that sends the rows.

- **Key**: `(tenant, sha256(context), members, model version)`; the tenant
  comes from the bearer key. Identical bytes from two tenants are two
  entries — a shared hit would tell one that another holds the same data.
- **Soft state, memory only.** An entry may vanish at any time (idle
  expiry, eviction, restart, another instance). An unknown `context`
  answers `404 context_unknown` and the caller sends the rows again. This
  is what keeps the service portable across providers and safe to scale
  to zero.
- **Immutable.** A new month is a new context; the old one ages out.
- In GPU memory, within a budget (`GLOSSKERNELS_CACHE_MB`; two fifths of
  the device by default), half of it at most one caller's; the least
  recently read gives way, a caller's own first. A context that one
  caller's share could never hold is refused with what to do (fewer
  members, or no `cache`). A pinned-host tier under it is later work.
- Builds run one at a time (the package writes the cache onto the shared
  module while building); queries carry their cache and do not contend.
- The cap on a context is bytes — 24 KB × rows × members in fp16 — not
  rows: 100k rows is fine with one or two members.
- A kept read is the rows' read: float32 within 2e-6 of it. Contexts of
  5,000 rows or more are kept in float16 on CUDA — half the memory, a
  third of the build (17.6 s → 5.1 s at 20k × 8) — and stand within 0.0024
  of the float32 read at worst, in units of the target's spread (median
  0.0006). The percent-level float16 tail seen on the tiny fixtures does
  not appear at this scale.

## Calibration

`glosskernels.calibration`: a pure function of the answer and the record.

- The caller keeps each landed PIT (data, like its training rows) and
  sends them back as the 100-count histogram — a `GROUP BY`, half a
  kilobyte whatever the history's length. Nothing is fitted: a band at
  alpha is read at the level the past PITs put alpha at.
- The kernel ships a default record per voice (`calibration_default.json`,
  from public panels, counted as 200 observations); a tenant's counts are
  added to it and take over as they grow. No tenant's PITs reach another.
  On three panels the default was not built from, TabICL's 80/90 coverage
  goes from 69/84, 70/84, 82/91 to 78/89, 80/88, 85/93 with the quantile
  loss unchanged.
- The PIT is tie-aware: an actual tied with part of the grid (a zero month
  of a mostly-zero metric) is placed uniformly within the tie, from the
  row's bytes and its `salt`, so a replay repeats it.
- The default is kept per voice and per `window`, not per horizon. A month
  is read through the one-step record however far out: records built per
  horizon read the held-out panels no better (TabICL at twelve months,
  80% bands: 57/69/88 raw, 60/73/86 through a twelve-month record, 62/76/91
  through the one-step one). An annual total asked for as its own series
  has its own record, because every voice is too sure of one — a trailing
  sum is smooth, and says little about how far the next year's lands. On
  the three panels the default was not built from, 80% bands on annual
  totals held:

  | | raw | through the totals record |
  |---|---|---|
  | TabICL | 50 / 57 / 55 | 78 / 78 / 69 |
  | Chronos-2 | 85 / 62 / 72 | 90 / 65 / 77 |
  | seasonal-naive | 80 / 57 / 50 | 87 / 68 / 54 |
  | blend of the three | 87 / 65 / 65 | 96 / 79 / 75 |

  (fred_md / cif_2016 / car_parts; 428, 192 and 282 totals.) The panels
  differ from each other by more than any default can close — fred_md's
  blend is over-covered, car_parts' still short — which is the tenant's
  own record's part: it takes over from the default as it grows.
- The blend has no default record: it is the mean of the voices as read,
  each through its own. Against giving the blend a record of its own the
  two came out even over months and totals (179 against 187 points of
  coverage error, raw 227), and this one ships nothing per combination of
  voices. A caller's `blend` history is read on top, its PITs taken
  against that mean.
- The caller keeps a record per horizon, and one for a total; a what-if is
  calibrated from the factual reads of the same context (measured: as
  narrow as its what-ifs).

## Keeping the device busy

A small read does not use a GPU, however big: its forward pass is a long
chain of tiny kernels the host launches one by one — 35 ms for one
walk-sized table on an L4 that is 10% busy, and no faster on a bigger
card. What fills the device is tables of one shape riding the same chain:

| tables in one pass (L4, fp32, 24×5) | 1 | 8 | 64 | 256 | 1024 |
|---|---|---|---|---|---|
| tables per second | 28 | 221 | 799 | 877 | 831 |
| GPU busy | 9% | 15% | 67% | 99% | 99% |

So `/bands` prepares every read of a request, groups their tables by shape
(a read has one table per member), and answers a group per pass. With the
passes batched, what a walk waits on is the host preparing reads — the
package's own preprocessing, a few ms a read — so that runs in a pool of
processes beside the model. The pinned walk's 374 points on an L4: **19.1 s
one at a time, 1.31 s in one request** (0.41 s preparing on three workers,
0.76 s on the GPU), the same distance from the fixtures either way (worst
3e-4, no coverage flips). Batching moves an answer by ~1e-5.

Also measured: the package synchronizes the device and empties PyTorch's
allocator cache to read free memory, three times a pass; the kernel reads
the same number without the flush (8 tables: 46 → 36 ms).

- **One owner per GPU, fed by a bounded queue; handlers `await` it**
  (`owner.py`). The owner takes whatever is waiting — reads from every
  request in the queue — and answers them together, so under load the
  passes fill on their own and an idle service answers at once. No
  batching window, no added latency. Reads asking for different levels
  (each caller's record puts them elsewhere) still share a pass: the
  forward does not depend on them. A read the kernel refuses fails its own
  caller only. Riding with another tenant's tables moves a number by float
  noise (~1e-5, against a fixture tolerance of 2e-3); nothing of one
  tenant's data is visible to another.
  Sixteen callers sending small walks at once on an L4: 35.1 s a request
  at a time, **5.5 s through the owner**, the slowest request 0.75 s.
- **The queue pushes back before memory does.** Waiting jobs are arrays in
  host memory, so the bound is on what they hold (`GLOSSKERNELS_QUEUE_MB`,
  512), half of it at most one caller's; a job that does not fit is a
  `429` with `Retry-After` from the measured rate, one that never could is
  a `413`, and so is a body over `GLOSSKERNELS_MAX_BODY_MB` before it is
  parsed. A cycle takes callers in turn up to a budget
  (`GLOSSKERNELS_CYCLE_MB`, 64), which bounds its length and the device
  memory a pass needs; a late caller rides the next cycle, not behind
  another's backlog.
- **What still idles the device**: a walk's six points are six shapes, so
  a cycle is six launch chains of a few tables each (GPU 13% busy in the
  run above). The chains could be launched side by side from several
  threads — which the GIL forbids today. Free-threaded Python is where
  that becomes possible: the whole stack installs on 3.14t and keeps the
  GIL off (torch ships no 3.13t wheels). Measured so far, on the host
  side only: threads preparing reads go from 0.9x (GIL) to 2.0x (no GIL),
  but the free-threaded build is 1.5x slower single-threaded and the
  process pool still beats it (0.05 s against 0.17 s for 374 reads). The
  test that matters — parallel launch chains on CUDA — is open, as are
  CUDA graphs per shape.
- Threads do not share the device (no lock: same numbers, half the
  throughput on an L4). Hosts with more cores prepare more reads at once;
  past one process's reach, more processes per GPU (MPS).
- Precision is the service's business, not the caller's: fp32 for small
  reads (exact, and faster — 16 s against 29 s for the pinned walk), fp16
  only for large cached contexts and only once its deviation there is
  measured. On the small fixtures fp16 is fine at the median (3e-4) and
  percent-level wrong one fit in a hundred (worst 25%); bf16 is worse.

## Measured (fp16, 8 members, 20 columns)

| | T4 | L4 | L40S |
|---|---|---|---|
| walk point 24×5 | 0.15 s | 0.16 s | 0.04 s |
| 20k-row context, refit per query | 10.0 s | 4.5 s | 1.6 s |
| 20k-row context, cached: build → query | 10.6 → 0.33 s | 5.1 → 0.15 s | 1.9 → 0.08 s |
| 100k-row context, cached | OOM | OOM | 16 → 0.11 s (18.8 GB) |

**Production is GCP, on the L4** (G2 machines; Cloud Run with GPU or GKE).
GCP has no L40S: the later large-memory tier there is an A100 or an RTX
PRO 6000-class card.
A cold container answers after 6–11 s of checkpoint load; keep one warm.
The container is plain (the `Dockerfile`), with no provider's API in it;
more than one instance routes by context id, and a miss is only a rebuild.

## Later, once there is load

A CPU tier for small reads · grouping reads across tenants · free-threaded
Python · attribution (each mask is a fresh build, not a cached query) ·
entity scoring · synthetic twins.

## Build order

- [x] Calibration: tie-aware PIT, histogram histories, the default record.
- [x] `/bands` as the one band route (reads, members, actuals, `pit_history`).
- [x] Reads grouped by shape, prepared in a process pool: the pinned walk's
      374 points in 1.31 s on an L4 (19.1 s one at a time), fixtures held.
- [x] One GPU owner behind a bounded queue, answering what is waiting from
      every caller together, with pushback (429/413) before memory: sixteen
      callers' small walks 35.1 s -> 5.5 s on an L4.
- [x] The context cache behind `/bands`: per caller, soft state, LRU within
      a byte budget. On an L4 a 20k-row, 8-member context builds in 5.1 s
      and answers in 0.085 s, within 0.0024 (target spreads) of the rows' read.
- [ ] Shapes side by side: parallel launch chains (free-threaded 3.14t on
      CUDA) or CUDA graphs per shape. Done when sixteen callers' small walks
      keep an L4 over 60% busy.
- [x] Voices and `blend` through `/bands` (`voices.py`: Chronos-2 batched
      across a cycle's series, seasonal-naive), each read through its own
      record. The harness grades the same functions.
- [x] Projections held in the harness: at or under the naive floor at every
      horizon (met by the three-voice blend). Annual totals: the lever was a
      default record for totals (`window`), not one per horizon — the blend's
      80% bands hold 79 and 75 of 100 on two held-out panels (65 raw) and
      over-cover the third (96, from 87); the tenant's own record closes the rest.
- [x] `/misfit` per column (`columns: true`): the conditionals the score sums, returned unsummed;
      `folds` to score rows from a context they are not in.
- [ ] glossql moves to this API.
