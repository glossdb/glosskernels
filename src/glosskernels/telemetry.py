"""What the service says about itself: one JSON line per event on
stdout, and the same as OpenTelemetry when a collector is named.

The events are the ones that decide capacity and health: a cycle of
the owner (reads, callers, cells, device seconds), a refusal (status,
caller, why), a failure (with its traceback), a start. Never a payload.

Logs go to stdout as JSON always — a host that keeps stdout (Cloud
Run, any container runtime) parses that as-is. With
`OTEL_EXPORTER_OTLP_ENDPOINT` set (the standard variable; a collector
sidecar on `http://localhost:4318` in a Cloud Run service) and the
`otel` extra installed, traces, metrics and logs also go there over
OTLP/HTTP: a span per request and per cycle; counters for reads,
cycles and refusals; a histogram of cycle seconds; a gauge of what
waits in the queue. `OTEL_SERVICE_NAME` and `OTEL_RESOURCE_ATTRIBUTES`
name the service, as the SDK reads them. Without the extra or the
endpoint, everything here is the log line."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

log = logging.getLogger("glosskernels")

# Set by `configure()` when the SDK is present and an endpoint is named.
_METERS: dict[str, Any] = {}
_TRACER: Any = None


class _Line(logging.Formatter):
    """One JSON object per record: the message as `event`, `extra` fields beside it."""

    RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        line: dict[str, Any] = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "severity": record.levelname,
            "event": record.getMessage(),
            "logger": record.name,
        }
        line.update({k: v for k, v in record.__dict__.items() if k not in self.RESERVED and not k.startswith("_")})
        if record.exc_info:
            line["traceback"] = self.formatException(record.exc_info)
        return json.dumps(line, default=str)


def configure(version: str) -> dict[str, Any]:
    """Wire stdout logging, and OTLP when named. Returns what was wired, for the start line."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_Line())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    # uvicorn's own loggers ride the same handler, as JSON lines.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).handlers = []
        logging.getLogger(name).propagate = True

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return {"logs": "stdout", "otlp": None}
    try:
        from opentelemetry import metrics, trace
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        log.warning("otel_missing", extra={"endpoint": endpoint, "hint": "install the `otel` extra"})
        return {"logs": "stdout", "otlp": None}

    resource = Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "glosskernels"), "service.version": version})
    traces = TracerProvider(resource=resource)
    traces.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(traces)
    meters = MeterProvider(resource=resource, metric_readers=[PeriodicExportingMetricReader(OTLPMetricExporter())])
    metrics.set_meter_provider(meters)
    logs = LoggerProvider(resource=resource)
    logs.add_log_record_processor(BatchLogRecordProcessor(OTLPLogExporter()))
    set_logger_provider(logs)
    root.addHandler(LoggingHandler(logger_provider=logs))

    meter = meters.get_meter("glosskernels")
    global _TRACER
    _TRACER = traces.get_tracer("glosskernels")
    _METERS.update(
        cycles=meter.create_counter("glosskernels.cycles", description="owner cycles"),
        reads=meter.create_counter("glosskernels.reads", description="reads answered"),
        cells=meter.create_counter("glosskernels.cells", description="cells answered (float64 values)"),
        refusals=meter.create_counter("glosskernels.refusals", description="requests refused, by status"),
        failures=meter.create_counter("glosskernels.failures", description="cycles that raised"),
        cycle_seconds=meter.create_histogram("glosskernels.cycle.seconds", unit="s", description="device seconds per cycle"),
        queue_cells=meter.create_gauge("glosskernels.queue.cells", description="cells waiting after the cycle was taken"),
    )
    return {"logs": "stdout+otlp", "otlp": endpoint}


def _add(name: str, value: float, **attributes: Any) -> None:
    instrument = _METERS.get(name)
    if instrument is None:
        return
    if name == "queue_cells":
        instrument.set(value, attributes)
    elif name == "cycle_seconds":
        instrument.record(value, attributes)
    else:
        instrument.add(value, attributes)


def started(**fields: Any) -> None:
    log.info("started", extra=fields)


def cycle(jobs: int, reads: int, cells: int, callers: int, kind: str, seconds: float, waiting_cells: int) -> None:
    log.info(
        "cycle",
        extra={"jobs": jobs, "reads": reads, "cells": cells, "callers": callers, "kind": kind,
               "seconds": round(seconds, 4), "waiting_cells": waiting_cells},
    )
    _add("cycles", 1, kind=kind)
    _add("reads", reads, kind=kind)
    _add("cells", cells, kind=kind)
    _add("cycle_seconds", seconds, kind=kind)
    _add("queue_cells", waiting_cells)


def failed(kind: str, jobs: int, error: BaseException) -> None:
    log.error("cycle_failed", extra={"kind": kind, "jobs": jobs, "error": f"{type(error).__name__}: {error}"}, exc_info=error)
    _add("failures", 1, kind=kind)


def refused(status: int, caller: str | None, route: str, reason: str) -> None:
    log.info("refused", extra={"status": status, "caller": caller, "route": route, "reason": reason})
    _add("refusals", 1, status=str(status), route=route)


def span(name: str, **attributes: Any):
    """A span when tracing is wired, else nothing — `with span(...):`."""
    if _TRACER is None:
        import contextlib

        return contextlib.nullcontext()
    return _TRACER.start_as_current_span(name, attributes=attributes)


def asgi(app: Any) -> Any:
    """The app behind a request span when tracing is wired, else itself."""
    if _TRACER is None:
        return app
    from opentelemetry.instrumentation.asgi import OpenTelemetryMiddleware

    return OpenTelemetryMiddleware(app)
