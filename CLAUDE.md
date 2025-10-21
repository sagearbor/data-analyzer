# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## IMPORTANT: Developer Checklist

**All AI agents MUST check and update `developer_checklist.yaml` when working on this project.**

The checklist contains:
- Development phases with dependencies
- Task statuses (TODO, INPROGRESS, DONE)
- Priority levels and effort estimates
- Known bugs and technical debt

When working on tasks:
1. Check `developer_checklist.yaml` for related tasks
2. Update status to INPROGRESS when starting work
3. Add detailed notes about progress
4. Mark as DONE only when fully complete
5. Update the `last_updated` field

Example of updating a task:
```yaml
status: INPROGRESS
notes: |
  INPROGRESS:
  - Completed: Basic implementation
  - Remaining: Tests and documentation
  - Blocked: Waiting for dependency X
```

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
pip install -r requirements.txt       # Web app dependencies (includes LLM support)
pip install -r mcp_requirements.txt   # MCP server dependencies

# Run MCP server locally
python mcp_server.py

# Run web application with LLM support
# Option 1: Use the launch script (recommended)
./run_app.sh

# Option 2: Use Python wrapper
python run_streamlit.py

# Option 3: Manual activation and run
source venv/bin/activate && streamlit run web_app.py
```

**IMPORTANT:** Always use one of the above methods to ensure LLM functionality works correctly. Do NOT run `streamlit run web_app.py` directly without activating the virtual environment first.

**NOTE:** The application runs on port 3002 by default (configured in `run_app.sh`) for NGINX reverse proxy compatibility. When running manually, use: `streamlit run web_app.py --server.port 3002`

### Environment Configuration

The application uses the `APP_ENV` environment variable to control the environment warning banner:

- **`APP_ENV=dev`** (default): Red banner - "Development Environment"
- **`APP_ENV=staging`**: Yellow banner - "Staging Environment"
- **`APP_ENV=prod`**: No banner (production)
- **Not set**: Red banner (fail-safe default)

**Local Development:**
1. Copy `.env.example` to `.env`
2. Set `APP_ENV=dev` in your `.env` file
3. The banner will automatically appear at the bottom of the web interface

**Production Deployment:**
- Contact your IT/DevOps team for deployment procedures
- If using containerized deployment, set `APP_ENV` via environment variables:
  - Production: `APP_ENV=prod` (hides banner)
  - Staging: `APP_ENV=staging` (yellow banner)
- See `./scripts/experimental/azure/` for archived Azure deployment scripts (not used in production)

### Docker Development
```bash
# Build and run container (uses port 8501 internally)
docker build -t data-analyzer .
docker run -p 3002:8501 data-analyzer

# With environment variable
docker run -p 3002:8501 -e APP_ENV=dev data-analyzer
```

### Deployment
**Note:** Experimental deployment scripts are in `./scripts/experimental/azure/` but are NOT used in production. Contact your IT/DevOps team for actual deployment procedures.

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

### Environment Variables
- `APP_ENV`: Environment indicator (dev/staging/prod) - controls warning banner
  - `dev`: Red banner with "Development Environment"
  - `staging`: Yellow banner with "Staging Environment"
  - `prod`: No banner
  - Not set: Red banner (fail-safe default)

**Note:** Azure deployment variables archived in `./scripts/experimental/azure/`

### File Formats Supported
- **CSV**: Full support with encoding detection
- **JSON, Excel, Parquet**: Planned (architecture supports extension)

## Important Implementation Notes

- Web app currently simulates MCP calls via `_simulate_mcp_call()` method rather than using actual MCP client
- MCP server supports base64 encoded data and data URLs
- Schema validation supports: int, float, str, bool, datetime types
- Rules validation supports: min/max ranges for numeric data, allowed values for categorical data
- Environment banner powered by `env-banner-python` package (bottom-positioned)

## File Structure
- `mcp_server.py`: MCP server implementation
- `web_app.py`: Streamlit web application
- `requirements.txt`: Web app dependencies
- `mcp_requirements.txt`: MCP server dependencies
- `Dockerfile`: Container configuration
- `run_app.sh`: App launcher script (port 3002)
- `.env.example`: Environment variable template
- `scripts/experimental/`: Experimental features (not production)
- `docs/API.md`: Detailed API documentation