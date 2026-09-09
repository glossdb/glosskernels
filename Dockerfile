# The kernel service image: the reference package, the regressor
# checkpoint baked in, one process. torch's wheels carry the CUDA
# runtime libraries, so no CUDA base image; the host supplies the
# driver. On a machine without a GPU the same image serves on CPU.
FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy HF_HUB_OFFLINE=0
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev
# Bake the checkpoint: a container never fetches weights at start.
RUN uv run python -c "from glosskernels.kernels import fetch_checkpoints; fetch_checkpoints()"
ENV GLOSSKERNELS_ADDR=0.0.0.0:8100 HF_HUB_OFFLINE=1
EXPOSE 8100
CMD ["uv", "run", "--no-sync", "glosskernels"]
