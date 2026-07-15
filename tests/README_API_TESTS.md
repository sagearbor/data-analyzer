# API Test Suite - Quick Start Guide

This directory contains comprehensive tests for the Data Analyzer REST API.

## Test Files

### Main Test Files

1. **test_api.py** (27 tests)
   - Root endpoint
   - Health check endpoint
   - Dictionary parse endpoint (POST /api/v1/dictionary/parse)
   - Dictionary retrieval endpoint (GET /api/v1/dictionary/{dict_id})
   - OpenAPI documentation endpoints
   - Error handling

2. **test_api_integration.py** (24 tests)
   - End-to-end workflow tests
   - Multi-format dictionary parsing
   - Error recovery and graceful degradation
   - Concurrent request handling
   - File size limits
   - Authentication flows

3. **test_api_authentication.py** (12 tests - pre-existing)
   - API key authentication
   - Admin password authentication
   - Security headers
   - Error message validation

4. **conftest.py** (shared fixtures)
   - API test environment setup
   - Test client with authentication
   - Sample dictionary data (REDCap, FHIR)
   - Sample data with violations
   - Cleanup utilities

## Running Tests

### Run All API Tests
```bash
# From project root
pytest tests/test_api*.py -v

# With coverage report
pytest tests/test_api*.py --cov=api_server --cov-report=html
```

### Run Specific Test Files
```bash
# Authentication tests only (all passing)
pytest tests/test_api_authentication.py -v

# Endpoint tests
pytest tests/test_api.py -v

# Integration tests
pytest tests/test_api_integration.py -v
```

### Run Specific Test Classes
```bash
# Health check tests
pytest tests/test_api.py::TestHealthEndpoint -v

# Authentication tests
pytest tests/test_api_authentication.py::TestAPIKeyAuthentication -v

# Integration workflow tests
pytest tests/test_api_integration.py::TestDictionaryWorkflow -v
```

### Run Specific Tests
```bash
# Single test
pytest tests/test_api.py::TestHealthEndpoint::test_health_check_no_auth_required -v

# Multiple specific tests
pytest tests/test_api.py::TestHealthEndpoint tests/test_api.py::TestRootEndpoint -v
```

### Run with Different Output Formats
```bash
# Short output
pytest tests/test_api*.py -v --tb=short

# Minimal output (just pass/fail)
pytest tests/test_api*.py --tb=no

# Capture output (don't show print statements)
pytest tests/test_api*.py -v --capture=no
```

## Current Test Status

**Total Tests**: 63
**Passing**: 39 (62%)
**Failing**: 23 (file upload mocking issues)
**Error**: 1

### Passing Test Categories
- ✓ Health checks (4 tests)
- ✓ Authentication (16 tests)
- ✓ Basic endpoint functionality (6 tests)
- ✓ Error handling (3 tests)
- ✓ Integration workflows (10 tests)

### Known Issues
- File upload tests failing due to Pydantic v2 mocking complexity (not API bugs)
- OpenAPI documentation endpoints return 404 (may be intentional)

See `tests/API_TEST_REPORT.md` for detailed analysis.

## Test Fixtures

### Available Fixtures from conftest.py

```python
# API client and authentication
api_client          # FastAPI TestClient with environment configured
api_headers         # Headers with valid API key
admin_headers       # Headers with API key + admin password

# Sample dictionary data
sample_redcap_dictionary     # REDCap CSV format
sample_fhir_dictionary       # FHIR JSON format
real_redcap_dictionary_file  # Path to actual test file

# Sample data
sample_test_data_with_violations  # CSV with validation violations

# Database cleanup
cleanup_test_programs  # List to track programs created during tests
```

### Example Usage

```python
def test_my_endpoint(api_client, api_headers):
    """Test example using fixtures"""
    response = api_client.get("/api/v1/endpoint", headers=api_headers)
    assert response.status_code == 200
```

## Writing New Tests

### Template for Endpoint Tests

```python
class TestMyEndpoint:
    """Test description"""

    def test_success_case(self, api_client, api_headers):
        """Test successful request"""
        response = api_client.get("/api/v1/my-endpoint", headers=api_headers)
        assert response.status_code == 200
        data = response.json()
        assert "expected_field" in data

    def test_missing_auth(self, api_client):
        """Test request without authentication"""
        response = api_client.get("/api/v1/my-endpoint")
        assert response.status_code == 401

    def test_invalid_auth(self, api_client):
        """Test request with invalid authentication"""
        headers = {"X-API-Key": "wrong-key"}
        response = api_client.get("/api/v1/my-endpoint", headers=headers)
        assert response.status_code == 403
```

### Template for Integration Tests

```python
@patch('api_server.program_manager')
def test_workflow(mock_pm, api_client, api_headers):
    """Test complete workflow"""
    # Setup mocks
    mock_pm.method.return_value = mock_value

    # Step 1: First endpoint
    response1 = api_client.post("/api/v1/endpoint1", headers=api_headers)
    assert response1.status_code == 200
    data1 = response1.json()

    # Step 2: Use result from step 1
    response2 = api_client.get(f"/api/v1/endpoint2/{data1['id']}", headers=api_headers)
    assert response2.status_code == 200
```

## Test Organization

Tests are organized by:

1. **Functionality**: Each major endpoint gets its own test class
2. **Test Type**: Unit tests vs integration tests
3. **Scenario**: Success cases, auth failures, error handling, edge cases

### Test Naming Convention

```python
class Test<EndpointName><EndpointType>:  # e.g., TestDictionaryParseEndpoint
    def test_<scenario>_<expected_outcome>  # e.g., test_missing_auth_returns_401
```

## Continuous Integration

To run tests in CI/CD pipeline:

```yaml
# Example GitHub Actions
- name: Run API Tests
  run: |
    pytest tests/test_api*.py -v --cov=api_server --cov-report=xml

- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    file: ./coverage.xml
```

## Troubleshooting

### Tests Failing with Import Errors
```bash
# Ensure you're in the project root directory
cd /home/scb2/PROJECTS/gitRepos-wsl/data-analyzer

# Activate virtual environment
source venv/bin/activate

# Install test dependencies
pip install -r requirements.txt
```

### Tests Failing with Environment Variable Errors
Tests automatically mock environment variables. If you see env var errors, check:
- Tests are using the `mock_env` or `api_test_env` fixtures
- Environment variables are correctly defined in the fixture

### Slow Test Execution
```bash
# Run tests in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest tests/test_api*.py -v -n auto
```

## Additional Resources

- **API Test Report**: `tests/API_TEST_REPORT.md` - Detailed test analysis
- **API Server Code**: `api_server.py` - Implementation being tested
- **Developer Checklist**: `developer_checklist.yaml` - Task tracking (api_8)

## Contributing

When adding new API endpoints:

1. Add endpoint tests to `test_api.py`
2. Add integration workflow tests to `test_api_integration.py`
3. Add any new fixtures to `conftest.py`
4. Run full test suite to ensure no regressions
5. Update `API_TEST_REPORT.md` with new coverage
6. Update this README if test structure changes

## Questions?

See the comprehensive test report for detailed information:
- **tests/API_TEST_REPORT.md**

Or check the developer checklist:
- **developer_checklist.yaml** (task api_8)
