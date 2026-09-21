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
| `voices`, `reads[].history` | more than one voice answers: each comes back on its own, with their `blend`; a voice that reads a series takes it from `history` |

The answer is `{"reads": [{"quantiles", "raw"?, "pit"?, "context"?, "support"?, "voices"?}]}`.
`support` says, per query row, whether each feature lies inside what the
context has seen — a what-if is honest inside (its median 0.06 of the truth
off, against replay's 0.15) and must be flagged outside.

An annual total is asked for as its own series, never summed from monthly
bands (summed bands covered 94–99% at a nominal 80).

### `/misfit`

`{"x": [[…]], "columns": true}` → `scores` per row (log density, higher
fits better) and, with `columns`, per row × column: which cell made the row
improbable.

## The context cache

**Per tenant, across requests. Never shared between tenants, never per
request.** The weights are shared by everyone; a context is a customer's
data, so it belongs to one tenant — and within that tenant every request
reuses it. Build once (5 s at 20k rows on an L4), then 0.15 s per query.

- **Key**: `(tenant, sha256(context), members, model version)`; the tenant
  comes from the bearer key. Identical bytes from two tenants are two
  entries — a shared hit would tell one that another holds the same data.
- **Soft state, memory only.** An entry may vanish at any time (idle
  expiry, eviction, restart, another instance). An unknown `context`
  answers `404 context_unknown` and the caller sends the rows again. This
  is what keeps the service portable across providers and safe to scale
  to zero.
- **Immutable.** A new month is a new context; the old one ages out.
- GPU memory first (≈ 10 GB on an L4 beside the model and a build's working
  set), then pinned host RAM, then gone; LRU within a per-tenant byte quota.
- Builds run one at a time (the package writes the cache onto the shared
  module while building); queries carry their cache and do not contend.
- The cap on a context is bytes — 24 KB × rows × members in fp16 — not
  rows: 100k rows is fine with one or two members.

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
- Projections keep a record per horizon; a what-if is calibrated from the
  factual reads of the same context (measured: as narrow as its what-ifs).

## Sharing the device

- Requests are parsed on the event loop's thread pool and queued; **one
  worker owns the GPU**, handlers `await` it. The host prepares the next
  job while the GPU runs this one.
- Short reads are bound by the host launching kernels, not by waiting
  (GPU 6–13% busy): the remedy is many reads per request, grouped by shape
  into one forward pass — 100 metrics × 6 points is ~94 s one at a time on
  an L4.
- Round-robin across tenants, short jobs ahead of long builds; past a queue
  depth, `429` with `Retry-After`.
- Reads are grouped within a request only, so one tenant's load never
  changes another's numbers.
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

**L4 is the target**; a 48 GB card (L40S, RTX 6000-class) is the later tier.
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
- [ ] The context cache behind `/bands`, on one GPU owner with the async
      queue. Done when a 20k-row, 8-member context builds in ≤ 6 s and a
      cached read answers in ≤ 0.25 s on an L4, within a tolerance of the
      uncached read declared from the fp16-vs-fp32 measurement.
- [ ] Reads grouped by shape. Done when 100 metrics × 6 points answer in
      ≤ 10 s on an L4 and match the pinned fixtures within tolerance.
- [ ] Voices and `blend` (Chronos-2, seasonal-naive), per-horizon records.
      Done when the harness holds: at or under the naive floor at every
      horizon, annual-total coverage within 5 points of 80 (64–72 today).
- [ ] `/misfit` per column.
- [ ] glossql moves to this API.
