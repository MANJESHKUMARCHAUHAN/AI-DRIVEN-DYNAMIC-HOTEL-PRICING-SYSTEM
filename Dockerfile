# =============================================================================
# AI-Driven Dynamic Hotel Pricing System
#
# One image serves the API, the dashboard and the streaming worker -- they share
# every dependency, so three images would mean three copies of Prophet, NumPy
# and scikit-learn. Compose selects the role with `command:`.
#
# Multi-stage (ADR-006): Prophet and psycopg2 need a C toolchain to install.
# Compiling in a builder stage and copying only the finished virtualenv keeps
# gcc, g++ and the header packages out of the runtime image.
# =============================================================================

# --------------------------------------------------------------------------- #
# Stage 1 -- builder
# --------------------------------------------------------------------------- #
FROM python:3.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        build-essential \
        gcc \
        g++ \
        python3-dev \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# A virtualenv rather than the system site-packages: it copies to the runtime
# stage as a single self-contained directory.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the requirements first. Layer caching then makes source edits skip
# the expensive Prophet build entirely.
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

# --------------------------------------------------------------------------- #
# Stage 2 -- runtime
# --------------------------------------------------------------------------- #
FROM python:3.11-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="dynamic-hotel-pricing" \
      org.opencontainers.image.description="AI-driven dynamic hotel pricing platform" \
      org.opencontainers.image.source="https://github.com/example/dynamic-hotel-pricing"

# libpq5 is the psycopg2 runtime library. libgomp1 is required by scikit-learn's
# OpenMP-parallelised estimators -- omitting it produces an import-time failure
# that only appears once you actually try to fit a model.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libpq5 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    ENVIRONMENT=docker \
    LOG_FORMAT=json

# Never run as root.
RUN groupadd --system --gid 1001 appuser \
    && useradd --system --uid 1001 --gid appuser --create-home appuser

WORKDIR /app

COPY --chown=appuser:appuser . .

# Volume mount points must exist and be writable by the non-root user before
# the volume is attached.
RUN mkdir -p /app/data/raw /app/data/processed /app/data/synthetic \
             /app/models/artifacts \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

# Uses urllib from the venv rather than curl, so the runtime image needs no
# extra package installed purely for health checking.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
