#!/usr/bin/env python
"""
Python wrapper to ensure Streamlit runs with the correct virtual environment
This is an alternative to run_app.sh for users who prefer Python scripts
"""

import sys
import os
import subprocess

# Get the directory of this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths
venv_streamlit = os.path.join(script_dir, "venv", "bin", "streamlit")
web_app = os.path.join(script_dir, "web_app.py")

# Check if venv streamlit exists
if not os.path.exists(venv_streamlit):
    print("Error: Virtual environment not found or streamlit not installed in venv")
    print("Please run: source venv/bin/activate && pip install -r requirements.txt")
    sys.exit(1)

# Run streamlit from the virtual environment
print("Starting Data Analyzer with LLM support...")
print("Using virtual environment streamlit:", venv_streamlit)
print("LLM features will be available if Azure OpenAI credentials are configured in .env")
print("-" * 60)

try:
    subprocess.run([venv_streamlit, "run", web_app])
except KeyboardInterrupt:
    print("\nShutting down...")
except Exception as e:
    print(f"Error running streamlit: {e}")
    sys.exit(1)