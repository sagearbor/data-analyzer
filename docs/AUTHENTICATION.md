# API Authentication Guide

This document describes the authentication system for the Data Analyzer REST API.

## Overview

The Data Analyzer REST API uses header-based authentication with two levels:
- **API Key Authentication**: For regular API access
- **Admin Password Authentication**: For administrative operations

## Authentication Methods

### API Key Authentication

Used for protecting most API endpoints.

**Header Name**: `X-API-Key`

**Configuration**: Set the `DATA_ANALYZER_API_KEY` environment variable

**Example Request**:
```bash
curl -H "X-API-Key: your-api-key-here" \
  http://localhost:8000/api/v1/analyze
```

**Python Example**:
```python
import requests

headers = {"X-API-Key": "your-api-key-here"}
response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    json={"data": "..."},
    headers=headers
)
```

### Admin Password Authentication

Used for administrative endpoints (program management, system configuration).

**Header Name**: `X-Admin-Password`

**Configuration**: Set the `DATA_ANALYZER_ADMIN_PASSWORD` environment variable

**Example Request**:
```bash
curl -H "X-Admin-Password: your-admin-password-here" \
  http://localhost:8000/api/v1/programs/{program_id}
```

**Python Example**:
```python
import requests

headers = {"X-Admin-Password": "your-admin-password-here"}
response = requests.delete(
    "http://localhost:8000/api/v1/programs/abc123",
    headers=headers
)
```

## Setup

### 1. Generate Secure Credentials

For production deployments, generate cryptographically secure random credentials:

```bash
# Generate API Key
python -c "import secrets; print('API Key:', secrets.token_urlsafe(32))"

# Generate Admin Password
python -c "import secrets; print('Admin Password:', secrets.token_urlsafe(32))"
```

### 2. Configure Environment Variables

Add to your `.env` file:

```bash
# API Authentication
DATA_ANALYZER_API_KEY=your-generated-api-key-here
DATA_ANALYZER_ADMIN_PASSWORD=your-generated-admin-password-here
```

### 3. Restart the API Server

```bash
uvicorn api_server:app --reload
```

## Endpoint Protection

### Protecting an Endpoint with API Key

```python
from fastapi import Depends
from api_server import verify_api_key

@app.post("/api/v1/analyze")
async def analyze_data(
    request: AnalyzeRequest,
    api_key: str = Depends(verify_api_key)
):
    # Endpoint code here
    pass
```

### Protecting an Endpoint with Admin Password

```python
from fastapi import Depends
from api_server import verify_admin_password

@app.delete("/api/v1/programs/{program_id}")
async def delete_program(
    program_id: str,
    admin_password: str = Depends(verify_admin_password)
):
    # Endpoint code here
    pass
```

### Endpoints Without Authentication

Some endpoints don't require authentication (e.g., health checks):

```python
@app.get("/api/v1/health")
async def health_check():
    # No authentication required
    return {"status": "healthy"}
```

## Response Codes

### 401 Unauthorized

Returned when authentication credentials are missing.

**Example Response**:
```json
{
  "error": "Authentication required. Provide valid credentials.",
  "detail": "HTTPException: 401",
  "timestamp": "2025-12-02T10:30:00.000Z"
}
```

**Headers**: `WWW-Authenticate: ApiKey` or `WWW-Authenticate: AdminPassword`

### 403 Forbidden

Returned when authentication credentials are invalid.

**Example Response**:
```json
{
  "error": "Invalid credentials provided.",
  "detail": "HTTPException: 403",
  "timestamp": "2025-12-02T10:30:00.000Z"
}
```

## Security Best Practices

### 1. Keep Credentials Secret

- **Never** commit `.env` files to version control
- **Never** log API keys or passwords in plain text
- Use environment variables or secure secret management systems

### 2. Use HTTPS in Production

Always use HTTPS to encrypt credentials in transit:

```bash
# Production deployment should use HTTPS
https://api.example.com/api/v1/analyze
```

### 3. Rotate Credentials Regularly

Change API keys and admin passwords periodically:

```bash
# Generate new credentials
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Update .env file
# Restart API server
```

### 4. Monitor Failed Authentication Attempts

The API server logs failed authentication attempts:

```
WARNING  api_server:api_server.py:179 Invalid API key attempt from request
```

Monitor these logs to detect potential security issues.

### 5. Use Different Credentials Per Environment

Use different API keys for development, staging, and production:

```bash
# Development
DATA_ANALYZER_API_KEY=dev-key-12345

# Staging
DATA_ANALYZER_API_KEY=staging-key-67890

# Production
DATA_ANALYZER_API_KEY=prod-key-abcdef
```

## Disabling Authentication (Development Only)

For local development, you can disable authentication by not setting the environment variables:

```bash
# .env file - comment out or remove these lines
# DATA_ANALYZER_API_KEY=...
# DATA_ANALYZER_ADMIN_PASSWORD=...
```

When credentials are not configured, the authentication functions return `"unauthenticated"` and allow access. This is useful for local testing but **should never be used in production**.

## Testing Authentication

### Unit Tests

Run the authentication unit tests:

```bash
pytest tests/test_api_authentication.py -v
```

### Integration Tests

Run the integration tests:

```bash
pytest tests/test_api_integration_auth.py -v
```

### Manual Testing

Use the manual testing script:

```bash
# Start the API server first
uvicorn api_server:app --reload

# In another terminal, run the manual tests
python tests/manual_test_auth.py
```

## Troubleshooting

### Authentication Always Fails

**Problem**: Getting 403 even with correct credentials

**Solution**:
1. Check that environment variables are loaded: `printenv | grep DATA_ANALYZER`
2. Verify `.env` file exists and contains correct values
3. Restart the API server after changing `.env`

### Authentication Not Required

**Problem**: Endpoints accessible without credentials

**Solution**:
1. Verify `DATA_ANALYZER_API_KEY` is set in environment
2. Check that endpoint uses `Depends(verify_api_key)`
3. Look for logs: `WARNING: API_KEY not configured - authentication disabled`

### 401 vs 403 Confusion

- **401 Unauthorized**: Credentials are missing (no header provided)
- **403 Forbidden**: Credentials are invalid (wrong key/password provided)

## Rate Limiting

Authentication attempts are subject to rate limiting (configured via slowapi):

- Health endpoint: 60 requests/minute
- Other endpoints: 10 requests/minute (default)

Excessive authentication failures will trigger rate limits to prevent brute-force attacks.

## Future Enhancements

Planned improvements for the authentication system:

1. **Database-backed API keys**: Support multiple API keys per user
2. **Key expiration**: Automatically expire keys after a time period
3. **Scoped permissions**: Different keys with different access levels
4. **API key rotation**: Automated credential rotation
5. **JWT tokens**: Alternative to static API keys
6. **OAuth2 integration**: Support for OAuth2 authentication flows

## Support

For questions or issues with authentication:

1. Check this documentation first
2. Review the test files for examples
3. Check the API server logs for error messages
4. Contact your system administrator for production issues
