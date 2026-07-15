"""
Comprehensive API Endpoint Tests

Tests all REST API endpoints in api_server.py for:
- Valid requests and responses
- Invalid inputs and error handling
- Authentication and authorization
- Rate limiting
- Edge cases and boundary conditions

Run with:
    pytest tests/test_api.py -v
    pytest tests/test_api.py -v --cov=api_server
"""

import os
import pytest
import json
import tempfile
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from io import BytesIO


# Mock environment before importing api_server
@pytest.fixture(scope="module")
def mock_env():
    """Set up test environment variables"""
    with patch.dict(os.environ, {
        "DATA_ANALYZER_API_KEY": "test-api-key-12345",
        "DATA_ANALYZER_ADMIN_PASSWORD": "test-admin-pass-67890",
        "APP_ENV": "test",
    }):
        yield


@pytest.fixture(scope="module")
def client(mock_env):
    """Create test client with authentication configured"""
    from api_server import app
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """
    Reset slowapi's in-memory rate limit counters before each test.

    The `client` fixture is module-scoped so its underlying app (and the
    app.state.limiter it holds) persists across every test in this module.
    Without a reset, the production rate limits (e.g. 5/minute on
    /api/v1/dictionary/parse) accumulate across unrelated tests and cause
    order-dependent 429 failures. This does not touch api_server.py or its
    rate-limiting behavior - it only clears counters between tests so each
    test starts with a fresh limiter window, as it would against a freshly
    started server.
    """
    try:
        from api_server import limiter
        limiter.reset()
    except ImportError:
        pass
    yield


@pytest.fixture
def api_headers():
    """Standard API headers with valid API key"""
    return {"X-API-Key": "test-api-key-12345"}


@pytest.fixture
def admin_headers():
    """Admin headers with valid admin password"""
    return {
        "X-API-Key": "test-api-key-12345",
        "X-Admin-Password": "test-admin-pass-67890"
    }


@pytest.fixture
def invalid_api_headers():
    """Invalid API key headers"""
    return {"X-API-Key": "wrong-key"}


@pytest.fixture
def sample_dictionary_csv():
    """Sample REDCap data dictionary CSV content"""
    return """Variable / Field Name,Form Name,Section Header,Field Type,Field Label,Choices OR Calculations,Field Note,Text Validation Type,Min,Max,Branching Logic,Required,Identifier,Custom Alignment
record_id,demographics,,text,Record ID,,,,,,,,y,
age,demographics,,text,Age (years),,Enter age in years,integer,0,120,,y,,
gender,demographics,,radio,Gender,1 Male | 2 Female | 3 Other,,,,,,y,,
weight_kg,demographics,,text,Weight (kg),,Weight in kilograms,number,0,300,"[age] > 0",y,,
bmi,demographics,,calc,BMI,"[weight_kg] / ([height_cm]/100)^2",,,,,,,,
"""


@pytest.fixture
def sample_dictionary_json():
    """Sample FHIR-like data dictionary JSON content"""
    return json.dumps({
        "resourceType": "Questionnaire",
        "id": "test-questionnaire",
        "title": "Test Data Dictionary",
        "status": "active",
        "item": [
            {
                "linkId": "1",
                "text": "Record ID",
                "type": "string",
                "required": True
            },
            {
                "linkId": "2",
                "text": "Age",
                "type": "integer",
                "required": True
            }
        ]
    })


@pytest.fixture
def sample_data_csv():
    """Sample CSV data with intentional violations"""
    return """record_id,age,gender,weight_kg
1,25,1,70.5
2,150,2,65.0
3,30,4,80.0
4,35,,75.5
5,-5,1,350.0
"""


# ============================================================================
# Root Endpoint Tests
# ============================================================================


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root_endpoint(self, client):
        """Root endpoint should return API information"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data
        assert data["health"] == "/api/v1/health"


# ============================================================================
# Health Check Endpoint Tests
# ============================================================================


class TestHealthEndpoint:
    """Test health check endpoint"""

    def test_health_check_no_auth_required(self, client):
        """Health check should work without authentication"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "services" in data

        # Check service status structure
        services = data["services"]
        assert "llm_client" in services
        assert "program_manager" in services
        assert "logic_validator" in services
        assert "mcp_server" in services

    def test_health_check_status_values(self, client):
        """Health check status should be 'healthy' or 'degraded'"""
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]

    def test_health_check_version(self, client):
        """Health check should return version number"""
        response = client.get("/api/v1/health")
        data = response.json()
        assert data["version"] == "1.0.0"


# ============================================================================
# Dictionary Parse Endpoint Tests
# ============================================================================


class TestDictionaryParseEndpoint:
    """Test POST /api/v1/dictionary/parse endpoint"""

    def test_parse_dictionary_missing_auth(self, client, sample_dictionary_csv):
        """Request without API key should return 401"""
        files = {"dictionary_file": ("test.csv", sample_dictionary_csv, "text/csv")}
        response = client.post("/api/v1/dictionary/parse", files=files)
        assert response.status_code == 401

    def test_parse_dictionary_invalid_auth(self, client, sample_dictionary_csv, invalid_api_headers):
        """Request with invalid API key should return 403"""
        files = {"dictionary_file": ("test.csv", sample_dictionary_csv, "text/csv")}
        response = client.post("/api/v1/dictionary/parse", files=files, headers=invalid_api_headers)
        assert response.status_code == 403

    @patch('api_server.program_manager')
    def test_parse_dictionary_csv_valid(self, mock_pm, client, sample_dictionary_csv, api_headers):
        """Valid CSV dictionary should be parsed successfully"""
        # Mock the program manager response using the real ValidationProgram
        # attribute names read by api_server.parse_dictionary()
        mock_program = MagicMock()
        mock_program.program_id = "test-id-123"
        mock_program.name = "test-program"
        mock_program.created_at = "2025-12-02T00:00:00"
        mock_program.dictionary_format = "redcap_csv"
        mock_program.num_fields = 5
        mock_program.num_basic_rules = 3
        mock_program.num_logic_rules = 1
        mock_program.schema = {"record_id": "str", "age": "int"}
        mock_program.generated_code = "# validation code"
        mock_program.model_used = "gpt-5-nano"

        mock_pm.create_program_from_dictionary.return_value = mock_program

        files = {"dictionary_file": ("test.csv", sample_dictionary_csv, "text/csv")}
        data = {"save_program": "true"}
        response = client.post("/api/v1/dictionary/parse", files=files, data=data, headers=api_headers)

        # Should succeed or return 503 if services not available
        assert response.status_code in [200, 503]

    def test_parse_dictionary_unsupported_format(self, client, api_headers):
        """Unsupported file format should return 400"""
        files = {"dictionary_file": ("test.docx", b"fake docx content", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}
        response = client.post("/api/v1/dictionary/parse", files=files, headers=api_headers)
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    def test_parse_dictionary_pdf_not_implemented(self, client, api_headers):
        """PDF format should return 501 (not implemented)"""
        files = {"dictionary_file": ("test.pdf", b"%PDF-1.4 fake content", "application/pdf")}
        response = client.post("/api/v1/dictionary/parse", files=files, headers=api_headers)
        # Should return 501 or 503 depending on service availability
        assert response.status_code in [501, 503]

    @patch('api_server.program_manager')
    def test_parse_dictionary_json_valid(self, mock_pm, client, sample_dictionary_json, api_headers):
        """Valid JSON dictionary should be parsed successfully"""
        mock_program = MagicMock()
        mock_program.program_id = "test-id-json"
        mock_program.name = "test-program-json"
        mock_program.created_at = "2025-12-02T00:00:00"
        mock_program.dictionary_format = "fhir_json"
        mock_program.num_fields = 2
        mock_program.num_basic_rules = 1
        mock_program.num_logic_rules = 0
        mock_program.schema = {"record_id": "str", "age": "int"}
        mock_program.generated_code = "# validation code"
        mock_program.model_used = "gpt-5-nano"

        mock_pm.create_program_from_dictionary.return_value = mock_program

        files = {"dictionary_file": ("test.json", sample_dictionary_json, "application/json")}
        data = {"save_program": "true"}
        response = client.post("/api/v1/dictionary/parse", files=files, data=data, headers=api_headers)

        assert response.status_code in [200, 503]

    @patch('api_server.program_manager')
    def test_parse_dictionary_save_program_false(self, mock_pm, client, sample_dictionary_csv, api_headers):
        """Parse without saving should work"""
        mock_program = MagicMock()
        mock_program.program_id = "test-id-nosave"
        mock_program.name = "test-program-nosave"
        mock_program.created_at = "2025-12-02T00:00:00"
        mock_program.dictionary_format = "redcap_csv"
        mock_program.num_fields = 5
        mock_program.num_basic_rules = 3
        mock_program.num_logic_rules = 1
        mock_program.schema = {"record_id": "str", "age": "int"}
        mock_program.generated_code = "# validation code"
        mock_program.model_used = "gpt-5-nano"

        mock_pm.create_program_from_dictionary.return_value = mock_program

        files = {"dictionary_file": ("test.csv", sample_dictionary_csv, "text/csv")}
        data = {"save_program": "false"}
        response = client.post("/api/v1/dictionary/parse", files=files, data=data, headers=api_headers)
        # Should succeed or return 503 if services not available
        assert response.status_code in [200, 503]

    @patch('api_server.program_manager')
    def test_parse_dictionary_custom_name(self, mock_pm, client, sample_dictionary_csv, api_headers):
        """Custom program name should be accepted"""
        mock_program = MagicMock()
        mock_program.program_id = "test-id-customname"
        mock_program.name = "my-custom-program"
        mock_program.created_at = "2025-12-02T00:00:00"
        mock_program.dictionary_format = "redcap_csv"
        mock_program.num_fields = 5
        mock_program.num_basic_rules = 3
        mock_program.num_logic_rules = 1
        mock_program.schema = {"record_id": "str", "age": "int"}
        mock_program.generated_code = "# validation code"
        mock_program.model_used = "gpt-5-nano"

        mock_pm.create_program_from_dictionary.return_value = mock_program
        mock_pm.db.save_program.return_value = None

        files = {"dictionary_file": ("test.csv", sample_dictionary_csv, "text/csv")}
        data = {"save_program": "true", "program_name": "my-custom-program"}
        response = client.post("/api/v1/dictionary/parse", files=files, data=data, headers=api_headers)
        assert response.status_code in [200, 503]

    @patch('api_server.program_manager')
    def test_parse_dictionary_empty_file(self, mock_pm, client, api_headers):
        """Empty file content should fail parsing (mocked - no live LLM call)"""
        mock_pm.create_program_from_dictionary.side_effect = RuntimeError(
            "No fields could be extracted from empty dictionary content"
        )

        files = {"dictionary_file": ("empty.csv", b"", "text/csv")}
        response = client.post("/api/v1/dictionary/parse", files=files, headers=api_headers)
        # Empty file might be handled differently - check for error
        assert response.status_code in [400, 500, 503]

    @patch('api_server.program_manager')
    def test_parse_dictionary_malformed_csv(self, mock_pm, client, api_headers):
        """Malformed CSV should return error (mocked - no live LLM call)"""
        mock_pm.create_program_from_dictionary.side_effect = RuntimeError(
            "Failed to parse malformed CSV dictionary"
        )

        malformed_csv = "incomplete,header\nrow1,value1,extra_value\nrow2"
        files = {"dictionary_file": ("bad.csv", malformed_csv, "text/csv")}
        response = client.post("/api/v1/dictionary/parse", files=files, headers=api_headers)
        # Should fail parsing
        assert response.status_code in [400, 500, 503]


# ============================================================================
# Dictionary Retrieval Endpoint Tests
# ============================================================================


class TestDictionaryGetEndpoint:
    """Test GET /api/v1/dictionary/{dict_id} endpoint"""

    def test_get_dictionary_missing_auth(self, client):
        """Request without API key should return 401"""
        response = client.get("/api/v1/dictionary/test-id")
        assert response.status_code == 401

    def test_get_dictionary_invalid_auth(self, client, invalid_api_headers):
        """Request with invalid API key should return 403"""
        response = client.get("/api/v1/dictionary/test-id", headers=invalid_api_headers)
        assert response.status_code == 403

    @patch('api_server.program_manager')
    def test_get_dictionary_not_found(self, mock_pm, client, api_headers):
        """Non-existent program should return 404"""
        mock_pm.db.load_program.return_value = None

        response = client.get("/api/v1/dictionary/nonexistent-id", headers=api_headers)
        # Should return 404 or 503 if services not available
        assert response.status_code in [404, 503]

    @patch('api_server.program_manager')
    def test_get_dictionary_deleted(self, mock_pm, client, api_headers):
        """Deleted program should return 404"""
        mock_program = MagicMock()
        mock_program.status = "deleted"
        mock_program.name = "deleted-program"

        mock_pm.db.load_program.return_value = mock_program

        response = client.get("/api/v1/dictionary/deleted-id", headers=api_headers)
        assert response.status_code in [404, 503]

    @patch('api_server.program_manager')
    def test_get_dictionary_valid_id(self, mock_pm, client, api_headers):
        """Valid program ID should return program details"""
        # Attribute names must match ValidationProgram (src/program_cache.py),
        # since convert_validation_program_to_detail() reads these directly.
        mock_program = MagicMock()
        mock_program.program_id = "test-id-123"
        mock_program.name = "test-program"
        mock_program.status = "active"
        mock_program.created_at = "2025-12-02T00:00:00"
        mock_program.dictionary_source = "test.csv"
        mock_program.dictionary_format = "redcap_csv"
        mock_program.created_by = "test-user"
        mock_program.num_fields = 5
        mock_program.num_basic_rules = 3
        mock_program.num_logic_rules = 1
        mock_program.schema = {"record_id": "str", "age": "int"}
        mock_program.conditional_rules = []
        mock_program.generated_code = "# validation code"
        mock_program.aliases = []
        mock_program.use_count = 0
        mock_program.last_used = None
        mock_program.model_used = "gpt-5-nano"
        mock_program.generation_time_seconds = 1.5
        mock_program.version = 1

        mock_pm.db.load_program.return_value = mock_program

        response = client.get("/api/v1/dictionary/test-id-123", headers=api_headers)
        assert response.status_code in [200, 503]

    @patch('api_server.program_manager')
    def test_get_dictionary_by_name(self, mock_pm, client, api_headers):
        """Program can be retrieved by name"""
        mock_program = MagicMock()
        mock_program.program_id = "test-id-456"
        mock_program.name = "my-program-name"
        mock_program.status = "active"
        mock_program.created_at = "2025-12-02T00:00:00"
        mock_program.dictionary_source = "test.json"
        mock_program.dictionary_format = "fhir_json"
        mock_program.created_by = "test-user"
        mock_program.num_fields = 3
        mock_program.num_basic_rules = 2
        mock_program.num_logic_rules = 0
        mock_program.schema = {"patient_id": "str"}
        mock_program.conditional_rules = []
        mock_program.generated_code = "# code"
        mock_program.aliases = []
        mock_program.use_count = 5
        mock_program.last_used = "2025-12-02T12:00:00"
        mock_program.model_used = "gpt-5-nano"
        mock_program.generation_time_seconds = 1.2
        mock_program.version = 1

        mock_pm.db.load_program.return_value = mock_program

        response = client.get("/api/v1/dictionary/my-program-name", headers=api_headers)
        assert response.status_code in [200, 503]

    @patch('api_server.program_manager')
    def test_get_dictionary_by_alias(self, mock_pm, client, api_headers):
        """Program can be retrieved by alias"""
        mock_program = MagicMock()
        mock_program.program_id = "test-id-789"
        mock_program.name = "original-name"
        mock_program.status = "active"
        mock_program.created_at = "2025-12-02T00:00:00"
        mock_program.dictionary_source = "test.txt"
        mock_program.dictionary_format = "generic"
        mock_program.created_by = "test-user"
        mock_program.num_fields = 10
        mock_program.num_basic_rules = 8
        mock_program.num_logic_rules = 3
        mock_program.schema = {"field1": "int", "field2": "str"}
        mock_program.conditional_rules = []
        mock_program.generated_code = "# code"
        mock_program.aliases = ["my-alias", "another-alias"]
        mock_program.use_count = 15
        mock_program.last_used = "2025-12-02T15:00:00"
        mock_program.model_used = "gpt-5-nano"
        mock_program.generation_time_seconds = 2.1
        mock_program.version = 1

        mock_pm.db.load_program.return_value = mock_program

        response = client.get("/api/v1/dictionary/my-alias", headers=api_headers)
        assert response.status_code in [200, 503]


# ============================================================================
# OpenAPI Documentation Tests
# ============================================================================


class TestOpenAPIEndpoints:
    """Test OpenAPI documentation endpoints"""

    def test_openapi_json(self, client):
        """OpenAPI JSON schema should be accessible at the versioned path"""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema

    def test_docs_endpoint(self, client):
        """Swagger UI docs should be accessible at the versioned path"""
        response = client.get("/api/v1/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_endpoint(self, client):
        """ReDoc documentation should be accessible at the versioned path"""
        response = client.get("/api/v1/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling and edge cases"""

    def test_404_nonexistent_endpoint(self, client):
        """Non-existent endpoint should return 404"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Wrong HTTP method should return 405"""
        response = client.post("/api/v1/health")
        assert response.status_code == 405

    def test_invalid_json_body(self, client, api_headers):
        """Invalid JSON should return 422"""
        response = client.post(
            "/api/v1/dictionary/parse",
            data="not json",
            headers={**api_headers, "Content-Type": "application/json"}
        )
        # FastAPI handles this as 422 Unprocessable Entity
        assert response.status_code in [400, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
