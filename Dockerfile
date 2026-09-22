# The kernel service image: the reference package, the checkpoints
# baked in, one process. torch's wheels carry the CUDA runtime
# libraries, so no CUDA base image; the host supplies the driver. On a
# machine without a GPU the same image serves on CPU.
#
#   docker build --build-arg VERSION=$(git rev-parse --short HEAD) -t glosskernels .
#   docker run --gpus all -p 8100:8100 glosskernels
#
# Released as ghcr.io/glossdb/glosskernels:<version> at a v* tag
# (.github/workflows/release.yml), VERSION set to pyproject's version.
#
# What a host sets: GLOSSKERNELS_AUDIENCE and GLOSSKERNELS_CALLERS (or
# GLOSSKERNELS_KEYS) for the doors, OTEL_EXPORTER_OTLP_ENDPOINT for a
# collector, the queue and cache sizes (README). Nothing here names a
# project or a region; that is the deployment's, not the image's.
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
# Not root: everything under /app is made by this user (a chown of the
# built tree would copy its gigabytes into another layer).
RUN useradd --system --uid 10001 --create-home --home-dir /app kernel
USER kernel
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 HF_HOME=/app/hf HF_HUB_OFFLINE=0
COPY --chown=kernel pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra auth --extra otel
COPY --chown=kernel src ./src
RUN uv sync --frozen --no-dev --extra auth --extra otel
# Bake the checkpoints: a container never fetches weights at start.
RUN /app/.venv/bin/python -c "from glosskernels.kernels import fetch_checkpoints; fetch_checkpoints()"
# The build's name, reported by /healthz and every log line.
ARG VERSION=dev
ENV GLOSSKERNELS_VERSION=$VERSION
# The process itself as PID 1, so the host's SIGTERM reaches uvicorn.
ENV GLOSSKERNELS_ADDR=0.0.0.0:8100 HF_HUB_OFFLINE=1
EXPOSE 8100
CMD ["/app/.venv/bin/glosskernels"]
