# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

# Copy requirements files
COPY requirements.txt .
COPY mcp_requirements.txt .

# Install Python dependencies (with SSL verification workaround for WSL/corporate networks)
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r mcp_requirements.txt

# Copy application files
COPY mcp_server.py .
COPY web_app.py .
COPY demo_dictionaries.py .
COPY mermaid_renderer.py .
COPY entrypoint.sh .

# Copy source modules
COPY src/ ./src/

# Copy assets (required for mermaid diagrams)
COPY assets/ ./assets/

# Copy Streamlit configuration
COPY .streamlit/ ./.streamlit/

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8002/_stcore/health

# Run the Streamlit app via entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
