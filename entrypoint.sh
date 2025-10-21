#!/bin/bash
# Entrypoint script for Streamlit app with configurable base URL path

# Default to empty string (root path) if not set
BASE_URL_PATH=${BASE_URL_PATH:-""}

# Build streamlit command with additional NGINX-friendly settings
if [ -z "$BASE_URL_PATH" ]; then
    # No base path - serve at root
    exec streamlit run web_app.py \
        --server.port=8002 \
        --server.address=0.0.0.0 \
        --server.enableXsrfProtection=false \
        --server.enableCORS=false \
        --server.enableWebsocketCompression=false
else
    # With base path - for reverse proxy
    # Note: baseUrlPath tells Streamlit what external path clients use,
    # even if NGINX strips it before proxying
    exec streamlit run web_app.py \
        --server.port=8002 \
        --server.address=0.0.0.0 \
        --server.baseUrlPath="$BASE_URL_PATH" \
        --server.enableXsrfProtection=false \
        --server.enableCORS=false \
        --server.enableWebsocketCompression=false
fi
