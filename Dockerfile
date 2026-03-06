# ============================================================
# Stage 1: Build — install dependencies in a clean venv
# ============================================================
FROM python:3.11-slim AS builder

WORKDIR /build

# Install only essential build deps for compiled packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

# Install into a virtual env so we can copy it cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt

# ============================================================
# Stage 2: Runtime — minimal image with only what's needed
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy the pre-built venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install minimal runtime system deps (libgomp for XGBoost)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    # Create data directory for runtime downloads
    mkdir -p /app/data/processed

# Copy only runtime application files (.dockerignore handles exclusions)
COPY app.py .
COPY src/__init__.py src/__init__.py
COPY src/explainability.py src/explainability.py
COPY models/ models/

# Configure Streamlit for container deployment
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    PYTHONUNBUFFERED=1 \
    # Reduce matplotlib memory
    MPLBACKEND=Agg

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py"]
