# Authentication System - README

This directory contains documentation for the Data Analyzer REST API authentication system.

## Quick Navigation

### For Developers Adding Endpoints
- **Quick Start**: See `AUTHENTICATION_QUICK_START.md` (5-minute guide)
- **Code Examples**: See `../examples/endpoint_with_authentication.py`
- **Copy-Paste Template**:
  ```python
  from fastapi import Depends
  from api_server import verify_api_key

  @app.post("/api/v1/your-endpoint")
  async def your_endpoint(api_key: str = Depends(verify_api_key)):
      # Your code here
      pass
  ```

### For System Administrators
- **Full Guide**: See `AUTHENTICATION.md`
- **Setup**: Generate credentials with `python -c "import secrets; print(secrets.token_urlsafe(32))"`
- **Configure**: Add to `.env` file:
  ```bash
  DATA_ANALYZER_API_KEY=your-generated-key
  DATA_ANALYZER_ADMIN_PASSWORD=your-generated-password
  ```

### For API Users
- **Using the API**: Add header to requests:
  ```bash
  curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/endpoint
  ```
- **Python Client**:
  ```python
  import requests
  headers = {"X-API-Key": "your-key"}
  requests.post(url, headers=headers)
  ```

## Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| `AUTHENTICATION_QUICK_START.md` | 5-minute setup guide | Developers |
| `AUTHENTICATION.md` | Comprehensive reference | All users |
| `../examples/endpoint_with_authentication.py` | Code examples | Developers |
| `../API_AUTHENTICATION_IMPLEMENTATION_REPORT.md` | Implementation details | Technical leads |

## Test Files

| File | Purpose |
|------|---------|
| `../tests/test_api_authentication.py` | Unit tests (12 tests) |
| `../tests/test_api_integration_auth.py` | Integration tests (9 tests) |
| `../tests/manual_test_auth.py` | Manual testing script |

**Run Tests**: `pytest tests/test_api_authentication.py tests/test_api_integration_auth.py -v`

## Implementation Files

| File | Lines | Purpose |
|------|-------|---------|
| `../api_server.py` | 159-231 | Authentication implementation |
| `../.env.example` | 21-26 | Configuration template |

## Authentication Levels

1. **No Authentication**: Public endpoints (health checks)
2. **API Key**: Regular endpoints (data analysis, queries)
3. **Admin Password**: Administrative endpoints (program deletion, system config)

## Security Features

- HTTP 401 for missing credentials
- HTTP 403 for invalid credentials
- No credential leakage in errors or logs
- WWW-Authenticate headers
- Rate limiting compatible
- Environment-based configuration

## Status

- Implementation: COMPLETE
- Tests: 21/21 passing
- Documentation: Complete
- Production Ready: YES

## Support

For issues or questions:
1. Check `AUTHENTICATION_QUICK_START.md` for common scenarios
2. Review `AUTHENTICATION.md` for detailed information
3. Look at code examples in `../examples/endpoint_with_authentication.py`
4. Run tests to verify your setup
5. Check API server logs for authentication failures

## Last Updated

2025-12-02
