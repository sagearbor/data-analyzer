# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a data quality analysis tool with Azure deployment capabilities that consists of:
- **MCP Server** (`mcp_server.py`): Model Context Protocol server for data analysis
- **Web Application** (`web_app.py`): Streamlit-based web interface
- **Azure Deployment**: Container-based deployment to Azure Container Apps

The project analyzes structured data (currently CSV, extensible for JSON/Excel/Parquet) performing quality checks like data type validation, value range checks, missing value analysis, and duplicate detection.

## Development Commands

### Local Development
```bash
# Set up virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt       # Web app dependencies
pip install -r mcp_requirements.txt   # MCP server dependencies

# Run MCP server locally
python mcp_server.py

# Run web application locally
streamlit run web_app.py
```

### Docker Development
```bash
# Build and run container
docker build -t data-analyzer .
docker run -p 8501:8501 data-analyzer
```

### Azure Deployment
```bash
# Deploy to Azure with default settings
./deploy.sh deploy

# Deploy with custom app name
APP_NAME=myapp ./deploy.sh deploy

# View deployment info
./deploy.sh info

# Clean up resources
./deploy.sh cleanup
```

### Testing
Currently no formal test framework is configured. The README mentions test files that don't exist yet:
- `python -m pytest tests/test_mcp_server.py` (planned)
- `python -m pytest tests/test_web_app.py` (planned)
- `python tests/test_integration.py` (planned)

## Architecture

### Core Components

1. **MCP Server** (`mcp_server.py:28-506`): 
   - `DataLoader` class for loading data formats (currently CSV, extensible)
   - `QualityChecker` class for performing data quality validations
   - `QualityPipeline` class for orchestrating checks
   - MCP server implementation with `analyze_data` and `get_data_info` tools

2. **Web Application** (`web_app.py:1-681`):
   - Streamlit interface with file upload, schema editor, rules editor
   - `MCPClient` class for communicating with MCP server (currently simulated)
   - Interactive dashboard for displaying analysis results

3. **Data Quality Checks**:
   - Row count validation (`mcp_server.py:67-83`)
   - Data type validation (`mcp_server.py:85-150`)
   - Value range validation (`mcp_server.py:152-234`)
   - Summary statistics generation (`mcp_server.py:236-262`)

### Key Patterns

- **Extensible Data Loading**: `DataLoader.load_data()` method designed to support multiple formats
- **MCP Protocol**: Server follows Model Context Protocol standards for tool definitions
- **Async Processing**: MCP server uses async/await patterns
- **Streamlit State Management**: Web app uses `st.session_state` for schema/rules configuration

## Configuration

### Environment Variables (Azure Deployment)
- `APP_NAME`: Application name (default: "data-analyzer")
- `RESOURCE_GROUP`: Azure resource group (default: "rg-{APP_NAME}")
- `LOCATION`: Azure region (default: "eastus")
- `CONTAINER_REGISTRY`: ACR name (default: "{APP_NAME}registry")

### File Formats Supported
- **CSV**: Full support with encoding detection
- **JSON, Excel, Parquet**: Planned (architecture supports extension)

## Important Implementation Notes

- Web app currently simulates MCP calls via `_simulate_mcp_call()` method rather than using actual MCP client
- MCP server supports base64 encoded data and data URLs
- Azure deployment creates: Resource Group, Container Registry, Log Analytics, Container Apps Environment, Container App
- Schema validation supports: int, float, str, bool, datetime types
- Rules validation supports: min/max ranges for numeric data, allowed values for categorical data

## File Structure
- `mcp_server.py`: MCP server implementation
- `web_app.py`: Streamlit web application  
- `requirements.txt`: Web app dependencies
- `mcp_requirements.txt`: MCP server dependencies
- `Dockerfile`: Container configuration
- `deploy.sh`: Azure deployment script
- `azure_config.yaml`: Azure resource templates
- `docs/API.md`: Detailed API documentation