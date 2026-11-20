#!/bin/bash
# Entrypoint script for Streamlit app with configurable base URL path

# Default to empty string (root path) if not set
BASE_URL_PATH=${BASE_URL_PATH:-""}

# ---------------------------------------------------------------------------
# Security posture (XSRF + CORS) - ON BY DEFAULT
#
# Streamlit's XSRF protection and CORS/origin check stay ENABLED. Azure
# Container Apps ingress and a modern single-origin reverse proxy preserve
# Host/Origin, so they work and must stay on.
#
# The legacy VM + NGINX path rewrite (nginx-helper.conf) tripped those checks.
# For that LOCAL-ONLY situation, and nothing else, set
#     STREAMLIT_INSECURE_LOCAL_ONLY=1
# to turn both guards off. The override is IGNORED on Azure Container Apps
# (any CONTAINER_APP_* variable present) or when APP_ENV is not dev/local/test.
# (STREAMLIT_SECURE=1, used by older manifests, is accepted and is now the
# default behaviour.)
# ---------------------------------------------------------------------------
SEC_FLAGS="--server.enableXsrfProtection=true --server.enableCORS=true"
if [ "${STREAMLIT_INSECURE_LOCAL_ONLY:-0}" = "1" ]; then
    ON_ACA=""
    for v in CONTAINER_APP_NAME CONTAINER_APP_REVISION CONTAINER_APP_HOSTNAME CONTAINER_APP_ENV_DNS_SUFFIX; do
        [ -n "${!v}" ] && ON_ACA="1"
    done
    case "${APP_ENV:-dev}" in
        dev|development|local|test) LOCAL_ENV="1" ;;
        *) LOCAL_ENV="" ;;
    esac
    if [ -n "$ON_ACA" ] || [ -z "$LOCAL_ENV" ]; then
        echo "WARNING: STREAMLIT_INSECURE_LOCAL_ONLY is ignored in a deployed environment; XSRF/CORS stay ON." >&2
    else
        echo "WARNING: XSRF and CORS protection are OFF (STREAMLIT_INSECURE_LOCAL_ONLY=1, local use only)." >&2
        SEC_FLAGS="--server.enableXsrfProtection=false --server.enableCORS=false"
    fi
fi

# Upload limit in MB (Streamlit's own default is 200, too large for a 1 GiB
# replica that holds the parsed data in memory). Default 25, below the API's
# 100 MB cap; override with UI_MAX_UPLOAD_MB (positive integer, else default).
case "${UI_MAX_UPLOAD_MB:-}" in
    ""|*[!0-9]*|0) UPLOAD_MB=25 ;;
    *) UPLOAD_MB="$UI_MAX_UPLOAD_MB" ;;
esac
SEC_FLAGS="$SEC_FLAGS --server.maxUploadSize=$UPLOAD_MB"

# ---------------------------------------------------------------------------
# Fail fast on a broken deployment config. In a non-local environment (ACA, or
# APP_ENV other than dev/local/test) the UI needs DATA_ANALYZER_API_URL and
# DATA_ANALYZER_API_KEY to reach the API. Without them the container exits
# non-zero, so the new revision never becomes ready (ACA keeps serving the
# previous healthy revision) and the reason is in the container log - instead
# of a "healthy" UI whose Analyze button cannot work. No-op locally.
# ---------------------------------------------------------------------------
python -m src.runtime_env --check-ui || exit 1

# Build streamlit command with additional reverse-proxy-friendly settings
if [ -z "$BASE_URL_PATH" ]; then
    # No base path - serve at root
    exec streamlit run web_app.py \
        --server.port=8002 \
        --server.address=0.0.0.0 \
        $SEC_FLAGS \
        --server.enableWebsocketCompression=false \
        --browser.gatherUsageStats=false
else
    # With base path - for reverse proxy
    # Note: baseUrlPath tells Streamlit what external path clients use,
    # even if NGINX strips it before proxying
    exec streamlit run web_app.py \
        --server.port=8002 \
        --server.address=0.0.0.0 \
        --server.baseUrlPath="$BASE_URL_PATH" \
        $SEC_FLAGS \
        --server.enableWebsocketCompression=false \
        --browser.gatherUsageStats=false
fi
