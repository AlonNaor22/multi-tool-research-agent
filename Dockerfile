# syntax=docker/dockerfile:1.7
#
# Multi-tool research agent — FastAPI REST + SSE service.
#
# Two-stage build:
#   1. builder — installs Python deps into an isolated venv. Build tools
#      (gcc, etc.) live only in this stage and never reach the final image.
#   2. runtime — slim image with only the venv + source code, running as a
#      non-root user. ~250 MB on disk vs ~750 MB for a single-stage build.

# ─── Builder stage ─────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

# Build-only dependencies. Some Python packages (pdfplumber's pikepdf
# backend, etc.) compile native extensions on install.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

# Install all Python deps into an isolated venv that the runtime stage
# can copy wholesale without dragging the build toolchain along.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt


# ─── Runtime stage ─────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Non-root user — anything reading mounted volumes from the host should
# match UID 1000 (the default on most Linux distros) for write access.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home --shell /bin/bash app

# Pull the pre-installed venv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Application source. .dockerignore filters out caches, secrets, tests,
# and host-side state dirs (sessions/, output/, observability/) — those
# are docker volumes, not part of the image.
COPY --chown=app:app . /app

# Persistent state directories. docker-compose mounts host paths over
# these; without compose, they live inside the container and vanish
# on rm. Pre-create with the right owner so the app user can write.
RUN mkdir -p /app/sessions /app/output /app/observability \
    && chown -R app:app /app/sessions /app/output /app/observability

USER app

EXPOSE 8000

# Probe /health via stdlib urllib (avoids a curl dependency on the image).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request, sys; \
        sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else sys.exit(1)" \
    || exit 1

CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8000"]
