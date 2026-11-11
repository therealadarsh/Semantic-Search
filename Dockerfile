# syntax=docker/dockerfile:1
#
# Three stages, so the final image only contains what is needed at runtime:
#   deps     -> installs Python dependencies into a virtualenv
#   index    -> builds the FAISS index with the same model the API uses
#   runtime  -> slim image with the venv, code, model and index; runs as non-root

ARG PYTHON_VERSION=3.12

# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS deps

# CPU-only PyTorch is ~10x smaller than the default CUDA build. Override this
# build arg if you ever deploy to a GPU host.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN python -m venv /opt/venv
COPY requirements.txt .
# The cache mount keeps pip's download cache between builds without putting
# it in the image.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url "$TORCH_INDEX_URL" "$(grep -E '^torch==' requirements.txt)" \
 && pip install -r requirements.txt

# ---------------------------------------------------------------------------
FROM deps AS index

WORKDIR /app
# The model ships in the repo, so the build must never reach the internet.
ENV HF_HUB_OFFLINE=1
COPY scripts/ scripts/
COPY models/ models/
COPY data/ag_news_train_12k/ data/ag_news_train_12k/
RUN python scripts/embed_data.py

# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HUB_OFFLINE=1 \
    PORT=8000

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=deps /opt/venv /opt/venv
COPY models/ models/
COPY --from=index /app/data/ data/
COPY scripts/ scripts/

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ['PORT'] + '/health', timeout=4)"

# Shell form so ${PORT} is expanded (many hosts inject PORT); exec so uvicorn
# is PID 1 and receives SIGTERM for graceful shutdown.
CMD ["sh", "-c", "exec uvicorn app:app --app-dir scripts --host 0.0.0.0 --port ${PORT}"]
