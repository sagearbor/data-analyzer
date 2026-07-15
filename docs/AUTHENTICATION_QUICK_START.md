# Authentication Quick Start Guide

A 5-minute guide to using the Data Analyzer API authentication system.

## Setup (One-Time)

1. **Copy the example environment file**:
   ```bash
   cp .env.example .env
   ```

2. **Generate secure credentials**:
   ```bash
   python -c "import secrets; print('API_KEY=' + secrets.token_urlsafe(32))"
   python -c "import secrets; print('ADMIN_PASSWORD=' + secrets.token_urlsafe(32))"
   ```

3. **Update `.env` file** with the generated values:
   ```bash
   DATA_ANALYZER_API_KEY=your-generated-api-key-here
   DATA_ANALYZER_ADMIN_PASSWORD=your-generated-admin-password-here
   ```

4. **Start the API server**:
   ```bash
   uvicorn api_server:app --reload
   ```

## Using Authentication in Code

### Protect a Regular Endpoint

```python
from fastapi import Depends
from api_server import verify_api_key

@app.post("/api/v1/analyze")
async def analyze_data(
    data: AnalyzeRequest,
    api_key: str = Depends(verify_api_key)  # Add this line
):
    # Your endpoint logic here
    return {"status": "success"}
```

### Protect an Admin Endpoint

```python
from fastapi import Depends
from api_server import verify_admin_password

@app.delete("/api/v1/programs/{program_id}")
async def delete_program(
    program_id: str,
    admin_password: str = Depends(verify_admin_password)  # Add this line
):
    # Your endpoint logic here
    return {"status": "deleted"}
```

### No Authentication Required

```python
@app.get("/api/v1/health")
async def health_check():
    # No Depends() - anyone can access
    return {"status": "healthy"}
```

## Making Authenticated Requests

### cURL

```bash
# Regular endpoint
curl -H "X-API-Key: your-api-key-here" \
  http://localhost:8000/api/v1/analyze

# Admin endpoint
curl -H "X-Admin-Password: your-admin-password-here" \
  -X DELETE http://localhost:8000/api/v1/programs/abc123
```

### Python (requests)

```python
import requests

# Regular endpoint
headers = {"X-API-Key": "your-api-key-here"}
response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    json={"data": "..."},
    headers=headers
)

# Admin endpoint
headers = {"X-Admin-Password": "your-admin-password-here"}
response = requests.delete(
    "http://localhost:8000/api/v1/programs/abc123",
    headers=headers
)
```

### JavaScript (fetch)

```javascript
// Regular endpoint
fetch('http://localhost:8000/api/v1/analyze', {
  method: 'POST',
  headers: {
    'X-API-Key': 'your-api-key-here',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({data: '...'})
})

// Admin endpoint
fetch('http://localhost:8000/api/v1/programs/abc123', {
  method: 'DELETE',
  headers: {
    'X-Admin-Password': 'your-admin-password-here'
  }
})
```

## Testing

### Run Tests

```bash
# All authentication tests
pytest tests/test_api_authentication.py tests/test_api_integration_auth.py -v

# Just unit tests
pytest tests/test_api_authentication.py -v

# Just integration tests
pytest tests/test_api_integration_auth.py -v
```

### Manual Testing

```bash
# Start server first
uvicorn api_server:app --reload

# In another terminal
python tests/manual_test_auth.py
```

## Common Issues

### "401 Unauthorized"
- **Cause**: Missing authentication header
- **Fix**: Add the appropriate header to your request

### "403 Forbidden"
- **Cause**: Invalid credentials
- **Fix**: Check that credentials match your `.env` file

### Authentication Not Required
- **Cause**: Environment variables not set
- **Fix**: Check `.env` file exists and contains correct values, then restart server

## Development Mode

For local development, you can disable authentication by removing/commenting these lines from `.env`:

```bash
# DATA_ANALYZER_API_KEY=...
# DATA_ANALYZER_ADMIN_PASSWORD=...
```

**WARNING**: Never disable authentication in production!

## Next Steps

For more details, see:
- Full documentation: `docs/AUTHENTICATION.md`
- Test examples: `tests/test_api_authentication.py`
- API documentation: http://localhost:8000/api/v1/docs (when server running)
