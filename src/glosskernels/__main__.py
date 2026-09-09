"""`glosskernels` — serve the kernel on a laptop or in a container.

`GLOSSKERNELS_ADDR` (default `127.0.0.1:8100`) is where it listens;
`GLOSSKERNELS_DEVICE` picks the device (default: cuda, then mps,
then cpu); `GLOSSKERNELS_KEYS` the bearer keys (see app.py). The
model loads before the doors open, so the first request pays no load.
"""

import os

import uvicorn


def main() -> None:
    addr = os.environ.get("GLOSSKERNELS_ADDR", "127.0.0.1:8100")
    host, _, port = addr.rpartition(":")
    from . import kernels

    kernels.get()
    uvicorn.run("glosskernels.app:app", host=host or "127.0.0.1", port=int(port), log_level="info")


if __name__ == "__main__":
    main()
