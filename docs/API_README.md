# Data Analyzer REST API

FastAPI-based REST API for programmatic access to data quality analysis, dictionary parsing, and validation program management.

## Quick Start

### Installation

```bash
# Install API dependencies (includes FastAPI, uvicorn, etc.)
pip install -r requirements.txt -r api_requirements.txt
```

### Running the API Server

**Option 1: Using Python directly**
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Run server (default: http://0.0.0.0:8000)
python api_server.py
```

**Option 2: Using uvicorn directly**
```bash
# Development mode with auto-reload
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

**Option 3: Custom configuration with environment variables**
```bash
export API_HOST=0.0.0.0
export API_PORT=8080
export API_RELOAD=false
python api_server.py
```

### Verifying Installation

```bash
# Run basic tests
./test_api_basic.sh

# Or test manually
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-12-02T21:37:00.317682",
  "services": {
    "llm_client": true,
    "program_manager": true,
    "logic_validator": true,
    "mcp_server": true
  }
}
```

## API Documentation

### Interactive Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

### Available Endpoints (api_1 - Basic Structure)

| Method | Endpoint | Description | Auth Required | Rate Limit |
|--------|----------|-------------|---------------|------------|
| GET | `/` | Root endpoint, API info | No | - |
| GET | `/api/v1/health` | Health check | No | 60/min |

### Upcoming Endpoints (Future Tasks)

The following endpoints will be implemented in subsequent tasks:

**api_2: Authentication**
- API key validation via `X-API-Key` header
- Admin password validation via `X-Admin-Password` header

**api_3: Data Analysis**
- `POST /api/v1/analyze` - Analyze data with optional dictionary
- `POST /api/v1/analyze/with-program` - Analyze using cached program

**api_4: Dictionary Management**
- `POST /api/v1/dictionary/parse` - Parse dictionary and optionally save
- `GET /api/v1/dictionary/{dict_id}` - Get dictionary details

**api_5: Program Management**
- `GET /api/v1/programs` - List programs with search/filters
- `GET /api/v1/programs/{id_or_alias}` - Get program details
- `POST /api/v1/programs/{id}/alias` - Create alias
- `DELETE /api/v1/programs/{id}` - Delete program (admin)
- `POST /api/v1/programs/{id}/restore` - Restore deleted program (admin)

## Architecture

### Service Integration

The API server integrates with the following existing modules:

```
api_server.py
├── src.llm_client.LLMDictionaryParser
│   └── Parses data dictionaries using Azure OpenAI
├── src.program_manager.ProgramManager
│   └── Manages validation programs and caching
├── src.logic_engine.LogicValidator
│   └── Executes conditional validation rules
└── mcp_server
    ├── QualityPipeline - Data quality checks
    └── DataLoader - Multi-format data loading
```

### Error Handling

All errors return standardized JSON responses:

```json
{
  "error": "Error message",
  "detail": "Detailed error information",
  "timestamp": "2024-12-02T21:37:00.317682",
  "request_id": "optional-request-id"
}
```

HTTP status codes:
- `200` - Success
- `400` - Bad Request (invalid input)
- `403` - Forbidden (authentication failed)
- `404` - Not Found
- `429` - Too Many Requests (rate limit exceeded)
- `500` - Internal Server Error

### Rate Limiting

Rate limits are enforced per IP address:
- Health endpoint: 60 requests/minute
- Other endpoints: 10 requests/minute (default)

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1701552000
```

## Configuration

### Environment Variables

Create a `.env` file or set environment variables:

```bash
# API Server Configuration
API_HOST=0.0.0.0          # Host to bind to
API_PORT=8000             # Port to listen on
API_RELOAD=true           # Enable auto-reload (development only)
APP_ENV=dev               # Environment: dev, staging, prod
DEBUG=false               # Enable debug mode (shows detailed errors)

# Azure OpenAI (inherited from main app)
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# Future: API Authentication (api_2)
# API_KEY=your-secret-api-key
# DATA_ANALYZER_ADMIN_PASSWORD=your-admin-password
```

### CORS Configuration

Currently configured to allow all origins for development:

```python
allow_origins=["*"]  # SECURITY: Restrict in production
```

For production, update `api_server.py` to restrict origins:

```python
allow_origins=[
    "https://yourdomain.com",
    "https://app.yourdomain.com",
]
```

## Development

### Project Structure

```
data-analyzer/
├── api_server.py           # FastAPI application (THIS FILE)
├── api_requirements.txt    # API-specific dependencies
├── test_api_basic.sh       # Basic API tests
├── src/
│   ├── llm_client.py      # LLM dictionary parser
│   ├── program_manager.py # Program management
│   └── logic_engine.py    # Logic validation
└── mcp_server.py          # Data quality pipeline
```

### Testing

**Basic functionality test:**
```bash
./test_api_basic.sh
```

**Manual testing:**
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Root endpoint
curl http://localhost:8000/

# OpenAPI schema
curl http://localhost:8000/api/v1/openapi.json | python -m json.tool
```

**Future: Comprehensive tests (api_8):**
```bash
pytest tests/test_api.py
pytest tests/test_api_auth.py
```

### Adding New Endpoints

1. Define Pydantic request/response models in `src/api_models.py` (if needed)
2. Add endpoint to `api_server.py`:

```python
@app.post("/api/v1/your-endpoint")
@limiter.limit("10/minute")
async def your_endpoint(
    request: Request,
    data: YourRequestModel,
    api_key: str = Depends(verify_api_key)  # If auth required
):
    """
    Endpoint description for OpenAPI docs
    """
    try:
        # Implementation
        result = await process_data(data)
        return YourResponseModel(**result)
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

3. Add tests to `tests/test_api.py`
4. Update this README

## Known Issues and Future Improvements

### Current Status (api_1)

✅ **Completed:**
- Basic FastAPI structure with CORS and rate limiting
- Health check endpoint
- Error handling framework
- Service initialization
- OpenAPI documentation generation

⚠️ **Known Issues:**
1. **Deprecation Warning**: Using `on_event` for startup/shutdown
   - **Fix**: Migrate to lifespan event handlers (FastAPI best practice)
   - **Impact**: None currently, but will be required in future FastAPI versions

2. **CORS Configuration**: Currently allows all origins
   - **Fix**: Restrict to specific domains in production
   - **Impact**: Security risk in production

3. **API Key Validation**: Placeholder implementation
   - **Fix**: Implement in api_2 task
   - **Impact**: No authentication currently

### Upcoming Tasks

- **api_2**: Implement API key authentication
- **api_3**: Data analysis endpoints
- **api_4**: Dictionary management endpoints
- **api_5**: Program management endpoints
- **api_6**: Additional system endpoints
- **api_7**: Comprehensive Pydantic models (DONE - see `src/api_models.py`)
- **api_8**: API testing suite
- **api_9**: Comprehensive API documentation
- **api_10**: Docker configuration

## Support

For issues or questions:
1. Check the Swagger docs: http://localhost:8000/api/v1/docs
2. Review `developer_checklist.yaml` for implementation status
3. See `IMPLEMENTATION_PLAN.md` for detailed specifications

## License

[Your License Here]
