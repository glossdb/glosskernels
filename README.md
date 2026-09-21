# glosskernels

The kernel service behind glossql's three model doors: the metric-bands
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
| `GLOSSKERNELS_KEYS` | comma-separated bearer keys; unset serves open (a laptop, or a host that authenticates in front) |
| `HF_HUB_OFFLINE` | `1` in the image: the checkpoint is baked, nothing is fetched at start |

The checkpoint comes from the Hugging Face hub cache (`jingang/TabICL`,
the v2 regressor). A container bakes it at build (`Dockerfile`); a
laptop fetches it on first start.

## The wire

JSON over HTTP, one route per model call, matrices as nested lists, a
`null` where the server has NaN. Every answer is `{"error": …}` with a
4xx when the read is refused, by name.

| route | body | answer |
|---|---|---|
| `POST /v1/band_point` | `train_x` (rows × cols), `train_y`, `test_x` (cols), `alphas`, `actual` | `quantiles` (per alpha), `pit` |
| `POST /v1/band_grid` | `train_x`, `train_y`, `test_x` (rows × cols), `alphas` | `quantiles` (rows × alphas) |
| `POST /v1/misfit` | `x` (rows × cols) | `scores` (per row; log density, higher fits the frame better) |
| `GET /healthz` | | `status`, `device`, `loaded` |

`band_point` runs the pinned member (one estimator, no normalization,
no feature shuffle) and reads the PIT off the monotone raw quantile
grid. `band_grid` runs the package's default ensemble. `misfit` fits
and scores the same frame over two feature orderings, numeric columns
only. These are the protocols the candle port was graded to; the tests
hold the reads to the port repo's pinned oracle fixtures.

## Hosts

- Modal (`modal_app.py`): `uv run modal deploy modal_app.py` serves the
  app behind a T4 in the EU with Modal's proxy auth; `uv run modal run
  modal_app.py` prints the timings. With `--context-rows 5000,20000,100000`
  the run adds what a shared instance turns on: a panel held as a cached
  context (`repr` and `kv`) against refitting it per call, and the short
  reads under concurrent callers with and without the service's lock.
  `GLOSSKERNELS_GPU=L4` names the GPU; `--amp` and `--check-parity` grade
  reduced precision against the fixtures.
- A container (`Dockerfile`): the same process on whatever GPU the
  host has, CPU without one.

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
| `walk:tabicl` | one series' months, the walk's graded recipe | the pinned member (`band_point`) |
| `pooled:tabicl` | the panel's recent rows, each series in its own units | the default ensemble (`band_grid`) |
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
and hospital). `cal:<voice>` grades the method on a voice's own record;
`dcal:<voice>` reads as a deployment would, default record included —
grade it on the other panels. The design for the first deployment is in
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
