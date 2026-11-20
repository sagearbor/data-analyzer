# ============================================================================
# Multi-stage Dockerfile for the Data Analyzer Streamlit UI
# ============================================================================
# Stage 1 (builder) has the compilers; stage 2 (runtime) has only what the app
# needs to run: no build-essential, no git, no openssh-client.
#
# Build: docker build --platform linux/amd64 -t data-analyzer-ui:latest .
#
# PIP_TRUSTED_HOSTS: DCRI machines sit behind a TLS-intercepting proxy, so the
# default keeps today's behaviour (pip is told to trust the PyPI hosts even if
# the certificate chain is not verifiable). This weakens build-time TLS
# verification (audit finding B-09). To turn it OFF, build with:
#     --build-arg PIP_TRUSTED_HOSTS=""
# (then the proxy CA must be trusted by the build, e.g. via PIP_CERT).
# ============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder - compile/install Python packages
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ARG PIP_TRUSTED_HOSTS="--trusted-host pypi.org --trusted-host files.pythonhosted.org"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

COPY requirements.txt .
COPY mcp_requirements.txt .

# PIP_TRUSTED_HOSTS is intentionally unquoted so it expands to separate flags.
RUN pip install --no-cache-dir --user ${PIP_TRUSTED_HOSTS} -r requirements.txt \
    && pip install --no-cache-dir --user ${PIP_TRUSTED_HOSTS} -r mcp_requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime - application + installed packages only
# ---------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/appuser/.local/bin:$PATH

# curl is used only by the HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Non-root user (never run the app process as root)
RUN useradd -m -u 1000 -s /bin/bash appuser

WORKDIR /app

COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application files
COPY --chown=appuser:appuser mcp_server.py .
COPY --chown=appuser:appuser web_app.py .
COPY --chown=appuser:appuser demo_dictionaries.py .
COPY --chown=appuser:appuser mermaid_renderer.py .
COPY --chown=appuser:appuser entrypoint.sh .

# Copy source modules
COPY --chown=appuser:appuser src/ ./src/

# Copy assets (required for mermaid diagrams)
COPY --chown=appuser:appuser assets/ ./assets/

# Copy Streamlit configuration
COPY --chown=appuser:appuser .streamlit/ ./.streamlit/

# Writable dirs and executable entrypoint, owned by the app user
RUN mkdir -p /app/data /app/logs \
    && chmod +x /app/entrypoint.sh \
    && chown appuser:appuser /app \
    && chown -R appuser:appuser /app/data /app/logs
USER appuser

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl --fail http://localhost:8002/_stcore/health

# Run the Streamlit app via entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
