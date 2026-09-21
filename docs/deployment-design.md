# glosskernels — the first deployment

Status: design agreed 2026-09-21; nothing here is built yet. What exists
today is a proof of concept: three doors (`band_point`, `band_grid`,
`misfit`) over the reference package, never deployed. Its reads stay as
the graded reference — every fast path below is held to them — and glossql
changes only once this is done.

## What it is for

One kernel service on one GPU, shared by every tenant of the SaaS, serving
the three reads the product leans on — **bands, what-if, projections** —
at cube scale. The proof of concept cannot do that for three measured reasons: every call
refits its context, one lock serializes every caller, and a walk is
hundreds of round trips of tiny tables.

**What belongs in this service, and what does not.** It is shared by every
tenant, so it carries only what needs the model, and the statistics that
read the model's own output — calibration and blending happen here and
nowhere else. Everything the caller can do stays with the caller:
assembling contexts, building feature rows (glossql's walk recipe stays
where it is), aggregating PIT histories. A request is numbers and a key,
as it is today.

This repo stays open (Apache-2.0). It wraps open models; there is nothing
secret in it. What is private is deployment — keys, the key→tenant map,
quotas, infrastructure code — and that lives in `glossdeploy`, not here.

## What the design rests on (measured, fp16, 8-member ensemble, 20 columns)

| | T4 | L4 | L40S |
|---|---|---|---|
| walk point 24×5 | 0.15 s | 0.16 s | 0.04 s |
| 20k-row panel, refit per query | 10.0 s | 4.5 s | 1.6 s |
| 20k-row panel, cached: build → query | 10.6 → 0.33 s | 5.1 → 0.15 s | 1.9 → 0.08 s |
| 100k-row panel, cached | OOM | OOM | 16 → 0.11 s (18.8 GB) |

- The KV cache costs **24 KB per row per ensemble member** (fp16). A cached
  query costs about the same for 1 row or 100, and barely grows with the
  panel.
- Short reads leave the GPU 6–13% busy: they are bound by Python on the
  host. Removing the lock changed no number on any GPU and *halved*
  throughput on the L4 and L40S — threads are not how the device is shared.
- A cold container answers after 6–11 s of checkpoint load (plus image
  pull the first time); warm, 0.4 s.

**L4 (24 GB) is the target.** A 48 GB card (L40S, or RTX 6000-class on
GCP) is the later tier for contexts past ~40k rows × 8 members. T4 is out.

## The context cache

**Per tenant, across requests. Never shared between tenants, never per
request.** The model's weights are shared by everyone; a context is a
customer's data (and the cache is derived from it), so it belongs to one
tenant — but within that tenant every request, user and read reuses it.
That reuse is the whole gain: build once (5 s at 20k rows on an L4), then
0.15 s per what-if, projection or band query.

- **Key**: `(tenant, sha256(context bytes), protocol, model version)`.
  The tenant comes from the bearer key (`GLOSSKERNELS_KEYS` becomes a
  key→tenant map). Two tenants uploading identical bytes get two entries:
  a shared hit would tell one tenant that another holds the same data.
- **Soft state.** The cache may vanish at any time (eviction, restart,
  scale-down, a different instance). A query against an unknown context
  answers `404 context_unknown` and the client uploads it again — glossql
  can always rebuild a context from the record. Nothing is written to
  disk. This is what makes the service portable across AWS, GCP, Scaleway
  or Modal, and safe to scale to zero.
- **Immutable.** A context is a snapshot. A new month is a new context
  (the rows attend to each other, so there is no appending); the old one
  ages out.
- **Tiers.** GPU memory first (L4 budget ≈ 10 GB after the model and the
  build's working set — about two 20k-row ensembles or ten 5k-row ones),
  then pinned host RAM in fp16 (a reload is a copy, not a rebuild — to be
  measured), then gone. LRU within a **per-tenant byte quota**, so one
  tenant cannot evict everyone else.
- **Deletion.** `DELETE` drops a context at once, and a tenant-wide purge
  exists for offboarding.
- **Builds are serialized** by the scheduler: the package writes the cache
  onto the shared module during a build (`model._cache`), so two builds at
  once would race. Queries carry their cache explicitly and do not.
- **Size cap per context** in bytes (24 KB × rows × members), not rows: a
  100k-row panel is fine with one or two members (2.4–4.8 GB). How much
  band quality fewer members cost is a harness question, answered before
  the cap is set.

## The doors

JSON stays for everything small. A context upload may be Arrow IPC
(`Content-Type: application/vnd.apache.arrow.stream`) — at 100k × 20 the
JSON parse alone is seconds of host time.

| door | body | answer |
|---|---|---|
| `PUT /v1/contexts` | `train_x`, `train_y`, `protocol` (`pinned` \| `ensemble`), `members`, `tier` (`graded` \| `fast`) | `context` (the hash), `rows`, `cols`, `cache_mb`, `build_s` — idempotent |
| `POST /v1/contexts/{id}/quantiles` | `test_x`, `alphas` or `grid: true`, `pit_history` (optional) | `quantiles` raw, `calibrated`, `support` per row |
| `DELETE /v1/contexts/{id}` | | |
| `POST /v1/band_walk` | many walk points in one request (below), `voices`, `pit_history` | per voice: `quantiles`, `pit`; plus `blend` and `calibrated` |
| `POST /v1/project` | per horizon the caller's feature rows, plus `history` (the series, for the voices that read one), `alphas`, `voices`, `pit_history` per horizon | per voice and horizon: `quantiles`; `blend`, `calibrated`; `total` when asked |
| `POST /v1/misfit` | `x`, `columns: true` | `scores` per row, and per row × column — which cell made the row improbable |

- **What-if is the contexts door.** glossql assembles the panel (with the
  attributes that place a lever against its usual level — the harness
  showed the read fails without them), uploads it once, and asks for the
  member's rows with the lever moved. `support` says, per query row,
  whether each feature lies inside what the context has seen: the read is
  honest inside (median 0.06 of truth off against replay's 0.15) and must
  be flagged outside. The replay-grid `band_grid` stays as it is.
- **`band_walk`** exists to end the round trips: 100 metrics × 6 points is
  600 `band_point` calls, ~94 s on an L4 today. One request lets the kernel group
  the points that share a shape into one forward pass (the model takes a
  batch of same-shaped tables). *To verify:* the batched path must
  reproduce `band_point` within the fast tier's tolerance on the pinned
  fixtures before it serves anything.
- **Voices.** `tabicl` always; `chronos2` as the second voice (it needs
  the series, not feature rows, so walk points carry `history` too). Each
  voice's answer is returned separately — they land as separate voices on
  the witness plane — and `blend` is the level-by-level mean, with
  `seasonal_naive` as a member for projections (the only configuration at
  or under the naive floor at every horizon on both panels).
- **Annual totals** are asked for as their own series (`total: 12`), never
  summed from monthly bands — summed bands covered 94–99% at a nominal 80.
- **Attribution is not in the first deployment.** Masking a column changes the context, so
  each mask is a fresh build, not a cached query; it needs its own costing.

## Calibration

A pure function here (`glosskernels.calibration`); the record keeps the
history.

- The request may carry `pit_history`: per voice and horizon, **a
  histogram of past PITs — 100 counts, ~0.5 KB**, whatever the history's
  length. Building it is a `GROUP BY` over stored PITs (deduplicated by
  metric, month and voice); reading it is the kernel's.
- The kernel ships a **default record** per voice and horizon, built from
  the harness's public panels, counted as a fixed number of observations
  (proposed: 200). A tenant's counts are added to it, so a new tenant gets
  honest bands on day one and their own record takes over as it grows.
  Measured: another panel's record mends coverage nearly as well as a
  panel's own (77 → 84 vs 83 at a nominal 80); only the *shape* of the
  misses needs the tenant's own history.
- No tenant's PITs ever reach another tenant. The default comes from
  public data only.
- PITs are always taken against the **raw** answer.
- **Before the wire is fixed: a tie-aware PIT.** On a mostly-zero metric
  (car_parts) the current PIT counts every tied quantile as "under", and
  calibrating on it made honest bands too wide. The new PIT places a tied
  actual uniformly within its tie, the draw derived from the request so a
  replay repeats it. `band_point` moves to it when glossql changes.

## Sharing the device

Async on the request side, one owner per GPU.

- Requests are parsed and preprocessed on the event loop's thread pool
  (numpy releases the GIL), then queued. **One worker owns the GPU** and
  runs one job at a time; handlers `await` its result. Waiting on the GPU
  costs the host nothing — CUDA launches are already asynchronous and the
  worker blocks only where a result is copied back — so the host is free
  to prepare the next job while the GPU runs this one.
- What async does **not** fix: short reads are slow because the host is
  *busy* launching kernels, not because it waits. That is what `band_walk`
  batching is for, and later a CPU tier for small reads (deferred until
  there is load to justify it).
- **Fairness**: round-robin across tenants, short jobs ahead of long
  builds; a queue-depth limit answers `429` with `Retry-After`.
- **Batching is within one request only** at first. Nothing about one
  tenant's load then changes another tenant's numbers, and a request is
  reproducible. Batching across tenants waits for load, and for a decision
  on whether bit-for-bit repeatability is promised.
- More throughput on one GPU comes from worker *processes* (NVIDIA MPS),
  not threads. Free-threaded Python stays a later experiment; the
  `concurrency` measurement is its test.

## Precision tiers

Measured on an L4 against the pinned fixtures (the walk's 374 fits, the 16
ensemble grids, the density read); deviation is per fit, relative:

| | median | 99th pct | worst | coverage flips (of 748) | walk time |
|---|---|---|---|---|---|
| fp32 | — | — | 3e-4 | 0 | 16 s |
| fp16 autocast | 3e-4 | 4.9% | 25% | 2 | 29 s |
| bf16 autocast | 3e-3 | 41% | 181% | 7 | 30 s |

So reduced precision is **not** a free switch. The typical fp16 fit is fine
and one fit in a hundred is percent-level wrong; bf16 is out; and on small
tables fp16 is also *slower* (autocast's casts outweigh the arithmetic —
the package's own `auto` heuristic keeps it off below ~1k rows).

- `graded`: fp32, no cache, the proof of concept's reads — what the fixtures pin. fp32 on
  the L4 holds them (worst 3e-4, no flips).
- `fast`: **fp32 for small reads** (walk points, replay grids — exact and
  faster), and fp16 only where memory forces it: large cached contexts,
  where the cache is 24 KB a row a member against 48 KB in fp32. Whether
  fp16's tail exists at that scale is unmeasured — the fixtures are tiny
  tables with raw, unscaled features. *Before step 2 ships:* the contexts
  measurement compares the fp16 cached read against the fp32 uncached one
  on the same panel, and the tolerance is declared from that. If the tail
  is there too, the L4 runs fp32 caches with fewer members (20k rows × 4
  members = 3.8 GB) rather than serve a band that is sometimes wrong.

## Deployment

- One plain container (the existing `Dockerfile`); no provider API in the
  service. Modal for measurement and beta; the SaaS provider (AWS, GCP or
  Scaleway — undecided) for production. Ask any provider for real 48 GB
  availability: an L40S queued 33 minutes on Modal.
- Scale-to-zero costs a 6–11 s first answer: keep one warm instance in
  business hours, or use a memory snapshot where the platform has one.
- More than one instance needs **context affinity**: route by context id
  so a query lands where its cache is; a miss is only a rebuild.
- Per tenant: queue wait, GPU seconds, cache bytes and hit rate. Payloads
  are never logged.

## Deliberately later

A CPU tier for small reads · cross-tenant batching · free-threaded Python ·
a C++ frontend · per-tenant fine-tuning · attribution · entity scoring
(RelBench is its benchmark) · synthetic twins.

## Build order and what "done" means

1. Tie-aware PIT and the default calibration record (small; blocks the wire).
2. Contexts door with the soft-state cache, on one GPU owner with the
   async queue. *Done when:* a 20k-row ensemble builds in ≤ 6 s and a
   cached query answers in ≤ 0.25 s (p50) on an L4, inside the fast tier's
   tolerance of the uncached read.
3. `band_walk` with in-request batching. *Done when:* 100 metrics × 6
   points answer in ≤ 10 s on an L4 (94 s today) and match `band_point`
   within tolerance on the pinned walk.
4. `project` with voices, blend and per-horizon calibration. *Done when*
   the harness targets hold: at or under the naive floor at every horizon,
   annual-total coverage within 5 points of 80 (64–72 today — open).
5. `misfit` per column.
