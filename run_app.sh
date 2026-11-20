#!/bin/bash
# Launch script for Data Analyzer web application with LLM support
# This ensures the app runs with the correct virtual environment

# Get the directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Navigate to the project directory
cd "$DIR"

# Activate virtual environment
source venv/bin/activate

# Ensure all dependencies are installed
echo "Checking dependencies..."
pip install -q -r requirements.txt

# Run the Streamlit app
echo "Starting Data Analyzer with LLM support..."
echo "LLM features will be available if Azure OpenAI credentials are configured in .env"
echo "Running on port 3002 for NGINX reverse proxy compatibility"
# Bind to loopback only (the NGINX proxy on this host reaches it there). XSRF and
# CORS protection come from .streamlit/config.toml and are ON; there is no
# override here - use entrypoint.sh's STREAMLIT_INSECURE_LOCAL_ONLY=1 if a local
# proxy truly needs them off.
streamlit run web_app.py --server.port 3002 --server.address "${STREAMLIT_BIND_ADDRESS:-127.0.0.1}"
