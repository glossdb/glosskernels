"""`glosskernels` — serve the kernel on a laptop or in a container.

`GLOSSKERNELS_ADDR` (default `127.0.0.1:8100`) is where it listens;
`GLOSSKERNELS_DEVICE` picks the device (default: cuda, then mps,
then cpu); the bearer keys or the ID-token audience gate the doors
(app.py); `GLOSSKERNELS_VERSION` names the build. The model loads
before the doors open, so the first request pays no load, and the port
opening is the readiness signal. Logs are JSON lines on stdout, and
OpenTelemetry when `OTEL_EXPORTER_OTLP_ENDPOINT` is set (telemetry.py).
"""

import os

import uvicorn


def main() -> None:
    addr = os.environ.get("GLOSSKERNELS_ADDR", "127.0.0.1:8100")
    host, _, port = addr.rpartition(":")
    from . import kernels, telemetry

    version = os.environ.get("GLOSSKERNELS_VERSION", "") or "dev"
    wired = telemetry.configure(version)
    k = kernels.get()
    auth = "id-token" if os.environ.get("GLOSSKERNELS_AUDIENCE", "").strip() else "keys" if os.environ.get("GLOSSKERNELS_KEYS", "").strip() else "open"
    telemetry.started(version=version, device=k.device, addr=addr, auth=auth, **wired)
    uvicorn.run("glosskernels.app:app", host=host or "127.0.0.1", port=int(port), log_config=None)


if __name__ == "__main__":
    main()
