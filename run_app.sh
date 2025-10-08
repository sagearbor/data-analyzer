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
streamlit run web_app.py