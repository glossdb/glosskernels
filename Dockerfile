# The kernel service image: the reference package, the checkpoints
# baked in, one process. torch's wheels carry the CUDA runtime
# libraries, so no CUDA base image; the host supplies the driver. On a
# machine without a GPU the same image serves on CPU.
#
#   docker build --build-arg VERSION=$(git rev-parse --short HEAD) -t glosskernels .
#   docker run --gpus all -p 8100:8100 glosskernels
#
# Built by .github/workflows/release.yml as ghcr.io/glossdb/glosskernels,
# VERSION set to pyproject's version plus the commit; a v* tag names it.
#
# What a host sets: GLOSSKERNELS_AUDIENCE and GLOSSKERNELS_CALLERS (or
# GLOSSKERNELS_KEYS) for the doors, OTEL_EXPORTER_OTLP_ENDPOINT for a
# collector, the queue and cache sizes (README). Nothing here names a
# project or a region; that is the deployment's, not the image's.

# The build: uv resolves the lock into a venv and the checkpoints are
# pulled into a hub cache. Nothing of this stage but those two trees
# reaches the image — not uv, not its cache, not the source.
FROM python:3.13-slim AS build
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
# No cache: it would be a second copy of every unpacked wheel. The
# project installed as a wheel, not editable — the source stays here.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 UV_NO_EDITABLE=1 HF_HOME=/app/hf
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --extra auth --extra otel
COPY src ./src
RUN uv sync --frozen --no-dev --extra auth --extra otel
RUN /app/.venv/bin/python -c "from glosskernels.kernels import fetch_checkpoints; fetch_checkpoints()"

# The image: the same interpreter the venv was made against, the venv,
# the checkpoints, an unprivileged user — two layers of payload and no
# tooling. (Not distroless: the venv is bound to python.org's 3.13 at
# /usr/local, and torch wants a libstdc++ and a libgomp beside it.)
FROM python:3.13-slim
RUN useradd --system --uid 10001 --create-home --home-dir /app kernel
COPY --from=build --chown=kernel:kernel /app/.venv /app/.venv
COPY --from=build --chown=kernel:kernel /app/hf /app/hf
USER kernel
WORKDIR /app
# The build's name, reported by /healthz and every log line.
ARG VERSION=dev
# The process itself as PID 1, so the host's SIGTERM reaches uvicorn;
# the checkpoints are baked, nothing is fetched at start.
ENV GLOSSKERNELS_VERSION=$VERSION GLOSSKERNELS_ADDR=0.0.0.0:8100 \
    HF_HOME=/app/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
EXPOSE 8100
CMD ["/app/.venv/bin/glosskernels"]
