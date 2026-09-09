"""The kernel on Modal: the same ASGI app behind a T4 in the EU, and a
measurement entrypoint for the first GPU run (cloud-deployment note,
open item 8).

    uv run modal deploy modal_app.py      # the service, proxy-authed
    uv run modal run modal_app.py         # the measurements, printed

The image bakes the regressor checkpoint; a container never fetches
weights at start.
"""

import os
import time
from pathlib import Path

import modal

REGION = "eu-west"
# The GPU under measurement: `GLOSSKERNELS_GPU=L4 modal run …`.
GPU = os.environ.get("GLOSSKERNELS_GPU", "T4")
# Four cores beside the GPU: the chain-rule read builds an estimator per
# conditional on the CPU side, and Modal's default is a fraction of one.
CPU = 4

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_pip_install(
        "tabicl==2.2.0",
        "torch>=2.13",
        "numpy>=2",
        "starlette>=1.6",
        "uvicorn>=0.52",
        "huggingface-hub>=0.30",
    )
    # Bake the regressor: the same hub file kernels.fetch_checkpoints pulls,
    # fetched here without the package on the path yet.
    .run_commands(
        "python -c \"from huggingface_hub import hf_hub_download; "
        "hf_hub_download(repo_id='jingang/TabICL', filename='tabicl-regressor-v2-20260212.ckpt')\""
    )
    .env({"HF_HUB_OFFLINE": "1", "PYTHONPATH": "/root"})
    .add_local_dir("src/glosskernels", remote_path="/root/glosskernels")
)

# The oracle fixtures, for the parity run on the GPU. Resolved on the
# local side only: inside the container this file lives at /root.
if modal.is_local():
    FIXTURES = Path(__file__).resolve().parent / "fixtures"
    if FIXTURES.is_dir():
        image = image.add_local_dir(FIXTURES, remote_path="/root/fixtures")

app = modal.App("glosskernels", image=image)


@app.function(gpu=GPU, cpu=CPU, region=REGION, scaledown_window=300, timeout=600)
@modal.concurrent(max_inputs=8)
@modal.asgi_app(requires_proxy_auth=True)
def serve():
    from glosskernels import kernels
    from glosskernels.app import app as asgi

    kernels.get()
    return asgi


@app.function(gpu=GPU, cpu=CPU, region=REGION, timeout=300)
def probe() -> float:
    """Load the model and answer: the caller's first call to this is the
    cold start (container, torch import, checkpoint), the second is warm."""
    t = time.perf_counter()
    from glosskernels.kernels import Kernels

    Kernels()
    return round(time.perf_counter() - t, 3)


@app.function(gpu=GPU, cpu=CPU, region=REGION, timeout=900)
def measure(use_amp: bool = False, misfit_workers: int = 0) -> dict:
    """The reads at their real shapes on the GPU (glosskernels.measure)."""
    from glosskernels.measure import run

    return run(use_amp=use_amp, misfit_workers=misfit_workers or None)


@app.function(gpu=GPU, cpu=CPU, region=REGION, timeout=1200)
def parity(use_amp: bool = False) -> dict:
    """The pinned-oracle parity numbers on this GPU (glosskernels.parity)."""
    from glosskernels.parity import run

    return run(Path("/root/fixtures"), use_amp=use_amp)


@app.local_entrypoint()
def main(amp: bool = False, check_parity: bool = False, misfit_workers: int = 0):
    if check_parity:
        print(parity.remote(use_amp=amp))
        return
    t = time.perf_counter()
    load_cold = probe.remote()
    cold_s = round(time.perf_counter() - t, 1)
    t = time.perf_counter()
    load_warm = probe.remote()
    warm_s = round(time.perf_counter() - t, 2)
    print({"gpu": GPU, "cold_call_s": cold_s, "cold_load_s": load_cold, "warm_call_s": warm_s, "warm_load_s": load_warm})
    print(measure.remote(use_amp=amp, misfit_workers=misfit_workers))
