# glosskernels

The kernel service behind glossql's model doors: the metric-bands
walk, `whatif.<scenario>()` and `misfit.<frame>()`. The server carries
no model; it calls this service at `GLOSSQL_TABICL_URL` with the bearer
in `GLOSSQL_TABICL_TOKEN`. Behind the wire is the reference `tabicl`
package at a pinned version, the regressor checkpoint loaded once.

## Run

```bash
uv sync
uv run glosskernels                 # 127.0.0.1:8100, the machine's best device
GLOSSKERNELS_KEYS=k1 uv run glosskernels   # a bearer per caller
```

| variable | meaning |
|---|---|
| `GLOSSKERNELS_ADDR` | where to listen; default `127.0.0.1:8100` |
| `GLOSSKERNELS_DEVICE` | `cuda`, `mps` or `cpu`; default the first available in that order |
| `GLOSSKERNELS_AUDIENCE`, `GLOSSKERNELS_CALLERS` | the doors take Google-signed ID tokens for that audience (the service's URL) from the listed service accounts — what a caller on GCP mints from its metadata server, no key anywhere; the `auth` extra |
| `GLOSSKERNELS_KEYS` | comma-separated bearer keys instead (a laptop, a test host); neither set serves open |
| `GLOSSKERNELS_VERSION` | the build's name, on `/healthz` and every log line; the image bakes the commit |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | a collector to send traces, metrics and logs to (OTLP/HTTP); unset, the JSON log lines on stdout are all; the `otel` extra |
| `GLOSSKERNELS_QUEUE_MB` | what may wait for the GPU, in MB of reads; default 512, half of it at most one caller's — past it a `429` with `Retry-After` |
| `GLOSSKERNELS_CACHE_MB` | device memory for kept contexts; default two fifths of the GPU (2 GB without one), half of it at most one caller's |
| `GLOSSKERNELS_MAX_BODY_MB` | the largest body parsed; default 256 — past it a `413` |
| `GLOSSKERNELS_PREPARE_WORKERS` | processes preparing reads beside the model; default the cores but one, at most 8 |
| `HF_HUB_OFFLINE` | `1` in the image: the checkpoint is baked, nothing is fetched at start |

The checkpoints come from the Hugging Face hub cache (`jingang/TabICL`,
the v2 regressor; `amazon/chronos-2`, the second voice, loaded on first use). A container bakes it at build (`Dockerfile`); a
laptop fetches it on first start.

## The wire

JSON over HTTP, matrices as nested lists, a `null` where the caller has
NaN. Every answer is `{"error": …}` with a 4xx when the read is refused,
by name.

| route | body | answer |
|---|---|---|
| `POST /bands` | `alphas`, `reads` (each `train_x`, `train_y`, `test_x` rows × cols — or `context` in place of the train rows — and optionally `actual` and `salt` per test row, `cache`), `members`, `pit_history`, `window` | per read: `quantiles` (rows × alphas), `pit` where an actual was given, `raw` when a `pit_history` was, `context` when kept |
| `POST /misfit` | `x` (rows × cols), `columns`, `folds` | `scores` (per row; log density, higher fits the frame better) and, with `columns: true`, `columns` (rows × cols; each column's share of the row's score — they sum to it, the lowest names the cell) |
| `GET /healthz` | | `status`, `version`, `device`, `loaded`, `owner` (its counters and the time of its last cycle) |

With `voices` (`tabicl` and any of `chronos2`, `seasonal_naive`) each read
also carries `history` — the series up to the origin — and optionally
`horizon` per test row; every voice answers on its own under `voices`,
beside their `blend`, and `pit_history` is then an object of histories by
voice, and the blend is of the voices as read through them. `window`
(1; 12 for a trailing annual total asked for as its own series) picks the
default record a history is weighed with. Every band read is `/bands`: a walk point is one read of one row with its
actual, a walk is many reads, a what-if or a projection is a read of many
rows. `members` 1 (the default) runs the pinned member (one estimator, no
normalization, no feature shuffle); more runs the package's ensemble of
that size. With `pit_history` — 100 counts of past PITs per hundredth —
the quantiles are read through that record and the kernel's default one
(`glosskernels.calibration`). `misfit` fits and scores the same frame over
two feature orderings, numeric columns only; with `folds` (2 to 10) each
row is scored from a context it is not in, at that many times the fits. The tests hold the reads to
the port repo's pinned oracle fixtures. Where this is going:
`docs/deployment-design.md`.

## Hosting

One container (`Dockerfile`): the same process on whatever GPU the
host has, CPU without one; not root, the checkpoints and the commit
baked, SIGTERM reaching the process. Nothing in it names a provider.
Production is GCP (`docs/deployment-design.md`); the deployment — the
project, the region, the collector sidecar, who may call — lives in the
private deployment repo, not here.

What the service says about itself (`telemetry.py`): a JSON line on
stdout per start, per owner cycle (jobs, reads, cells, callers, device
seconds, what still waits), per refusal (status, caller, route, reason)
and per failure (with the traceback) — never a payload. A host that
keeps stdout has that as-is; with `OTEL_EXPORTER_OTLP_ENDPOINT` the same
goes to a collector as spans (per request, per cycle), counters
(reads, cycles, refusals by status, failures), a histogram of cycle
seconds and a gauge of the queue.

The bench, on any host with a GPU that runs the container:

```bash
python -m glosskernels.measure                        # the reads' timings, this device
python -m glosskernels.measure --context-rows 5000,20000,100000   # cached contexts, concurrent callers
python -m glosskernels.measure --parity fixtures/     # the pinned oracle here; --amp, --bf16 grade reduced precision
python -m glosskernels.measure --batching             # small reads riding one pass; --kept the context cache
```

## The harness

What decides which voice is served: every voice walked one step ahead
over public monthly panels and scored as the witness plane scores a
band — quantile loss on the door's five alphas, 80/90 coverage, width,
and the PITs' distance from uniform.

```bash
uv sync --group harness                # pandas, pyarrow, chronos-forecasting, synthefy-nori
uv run python -m glosskernels.harness --panel tourism_monthly --series 60
uv run python -m glosskernels.harness --panel hospital --voices seasonal_naive,walk:tabicl,chronos2
# a voice read through its own record: twelve months walked first, the PITs they leave recalibrate the six scored
uv run python -m glosskernels.harness --panel hospital --burn 12 --voices seasonal_naive,walk:tabicl,cal:walk:tabicl
# projections: every month a year out from two origins, and the year's total — monthly bands summed against the total called directly
uv run python -m glosskernels.harness --panel hospital --project 2 --burn 8 \\
    --voices seasonal_naive,walk:tabicl,chronos2,cal:blend:walk:tabicl+chronos2 --out scores.json
```

| voice | the context | the model |
|---|---|---|
| `seasonal_naive` | the series' own year-over-year moves | none — the floor |
| `walk:tabicl` | one series' months, the walk's graded recipe | the pinned member (`members: 1`) |
| `pooled:tabicl` | the panel's recent rows, each series in its own units | the ensemble (`members: 8`) |
| `walk:nori`, `pooled:nori` | the same two contexts | Synthefy Nori (Apache-2.0) |
| `chronos2` | each series alone, as a series | Chronos-2 (Apache-2.0) |
| `seasonal:tabicl` | one series' months, rows anchored on the same month in the last years known — for calls further out | the pinned member |
| `blend:<voice>+<voice>` | its voices' | the mean of their quantiles, level by level |
| `cal:<voice>` | the voice's own landed PITs, the panel's together | the voice, its bands re-read at the levels its record puts them, per horizon |

A what-if cannot be graded on a real panel — the pulled world never
lands — so `--whatif` simulates one whose structure is known (revenue
is price times volume, the recipe; volume answers price, the behavior)
and runs the pulled world for the truth: the model's read from the
panel against replay's arithmetic alone, inside and outside the prices
the panel has seen, clean and with a demand shock no column holds.

Calibration is the kernel's (`glosskernels.calibration`): a pure
function of the raw answer and a histogram of the PITs the caller kept,
added to a default record the kernel ships (`calibration_default.json`,
rebuilt by `python -m glosskernels.harness.defaults` from tourism_monthly
and hospital: per voice, one record for months and one for trailing annual
totals). `cal:<voice>` grades the method on a voice's own record;
`dcal:<voice>` reads as a deployment would, default record included —
grade it on the other panels; `blend:dcal:<voice>+dcal:<voice>` is the
blend as the service forms it. The design for the first deployment is in
`docs/deployment-design.md`.

Panels: `tourism_monthly`, `hospital`, `car_parts`, `fred_md`, `cif_2016` (the Monash archive
from the hub) and `synthetic` (no network). Nori wants torch
under 2.14, and uv resolves every group into one lock, so the `harness`
group holds the whole lock at 2.13; the parity tests pass on 2.13 and 2.14.

## Tests

```bash
uv run pytest                          # the wire, and parity against fixtures/
GLOSSKERNELS_TEST_FITS=374 uv run pytest tests/test_kernels.py   # the whole pinned walk
```
