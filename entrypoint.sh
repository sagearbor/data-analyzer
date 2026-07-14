#!/bin/bash
# Entrypoint script for Streamlit app with configurable base URL path

# Default to empty string (root path) if not set
BASE_URL_PATH=${BASE_URL_PATH:-""}

# ---------------------------------------------------------------------------
# Security posture (XSRF + CORS)
#
# The original VM/NGINX deployment DISABLED Streamlit's XSRF and CORS guards
# because the NGINX sidecar terminated TLS and rewrote paths, which tripped
# Streamlit's same-origin checks. That posture is UNSAFE when Streamlit is
# exposed directly behind a modern reverse proxy.
#
# Azure Container Apps ingress is a single-origin HTTPS reverse proxy that
# preserves Host/Origin, so the guards work correctly and MUST stay ON.
# Set STREAMLIT_SECURE=1 (done in the ACA manifest) for the hardened posture.
# Unset / 0 keeps the legacy VM/NGINX behavior so nothing else breaks.
# ---------------------------------------------------------------------------
if [ "${STREAMLIT_SECURE:-0}" = "1" ]; then
    SEC_FLAGS="--server.enableXsrfProtection=true --server.enableCORS=true"
else
    SEC_FLAGS="--server.enableXsrfProtection=false --server.enableCORS=false"
fi

# Build streamlit command with additional reverse-proxy-friendly settings
if [ -z "$BASE_URL_PATH" ]; then
    # No base path - serve at root
    exec streamlit run web_app.py \
        --server.port=8002 \
        --server.address=0.0.0.0 \
        $SEC_FLAGS \
        --server.enableWebsocketCompression=false
else
    # With base path - for reverse proxy
    # Note: baseUrlPath tells Streamlit what external path clients use,
    # even if NGINX strips it before proxying
    exec streamlit run web_app.py \
        --server.port=8002 \
        --server.address=0.0.0.0 \
        --server.baseUrlPath="$BASE_URL_PATH" \
        $SEC_FLAGS \
        --server.enableWebsocketCompression=false
fi
