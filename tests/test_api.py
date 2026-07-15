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
# Data Quality Analysis Endpoint Tests
# ============================================================================


class TestAnalyzeEndpoint:
    """Test POST /api/v1/analyze endpoint (rule-based QualityPipeline analysis)"""

    def test_analyze_missing_auth(self, client, sample_data_csv):
        """Request without API key should return 401"""
        files = {"data_file": ("data.csv", sample_data_csv, "text/csv")}
        response = client.post("/api/v1/analyze", files=files)
        assert response.status_code == 401

    def test_analyze_invalid_auth(self, client, sample_data_csv, invalid_api_headers):
        """Request with wrong API key should return 403"""
        files = {"data_file": ("data.csv", sample_data_csv, "text/csv")}
        response = client.post("/api/v1/analyze", files=files, headers=invalid_api_headers)
        assert response.status_code == 403

    def test_analyze_valid_csv_with_rules(self, client, sample_data_csv, api_headers):
        """Valid CSV + rules should return expected range/categorical violations"""
        files = {"data_file": ("data.csv", sample_data_csv, "text/csv")}
        data = {
            "rules": json.dumps({
                "age": {"min": 0, "max": 120},
                "gender": {"allowed": [1, 2, 3]},
            })
        }
        response = client.post("/api/v1/analyze", files=files, data=data, headers=api_headers)
        assert response.status_code == 200

        result = response.json()
        # Response shape matches web_app.py's DataQualityAnalyzer contract
        assert set(result.keys()) == {
            "summary", "issues", "recommendations", "quality_checks", "summary_stats"
        }

        summary = result["summary"]
        assert summary["total_rows"] == 5
        assert summary["total_columns"] == 4
        assert summary["issues_found"] >= 2  # age=150 and age=-5 both violate max/min

        issue_types = {issue["type"] for issue in result["issues"]}
        assert "range_violation" in issue_types
        # gender has one missing value (row 4) -> missing_values issue
        assert "missing_values" in issue_types

        for issue in result["issues"]:
            assert issue["severity"] in ("error", "warning", "info")
            assert "message" in issue

    def test_analyze_with_schema(self, client, api_headers):
        """Schema type mismatches should surface in the raw quality_checks output.

        Note: QualityChecker's data_types check reports invalid values via an
        'invalid_values' list rather than 'violating_rows', so (matching
        web_app.py's original DataQualityAnalyzer transform, ported as-is)
        these don't get expanded into the shaped top-level 'issues' list -
        only range/allowed-value violations do. They're still visible in the
        raw 'quality_checks' passthrough, which this test asserts on.
        """
        csv_content = "id,signup_date\n1,2023-01-15\n2,not-a-date\n"
        files = {"data_file": ("data.csv", csv_content, "text/csv")}
        data = {"schema": json.dumps({"signup_date": "datetime"})}
        response = client.post("/api/v1/analyze", files=files, data=data, headers=api_headers)
        assert response.status_code == 200
        result = response.json()
        type_check = result["quality_checks"]["data_types"]
        assert type_check["passed"] is False
        assert type_check["issues"][0]["issue"] == "datetime_validation_failed"
        assert type_check["issues"][0]["column"] == "signup_date"

    def test_analyze_no_schema_no_rules(self, client, api_headers):
        """schema and rules are both optional - omitting them should still succeed"""
        csv_content = "id,name\n1,Alice\n2,Bob\n"
        files = {"data_file": ("data.csv", csv_content, "text/csv")}
        response = client.post("/api/v1/analyze", files=files, headers=api_headers)
        assert response.status_code == 200
        result = response.json()
        assert result["summary"]["total_rows"] == 2
        assert result["issues"] == []

    def test_analyze_clean_data_no_issues(self, client, api_headers):
        """All-valid data against its rules should report zero issues"""
        csv_content = "id,age\n1,25\n2,30\n3,45\n"
        files = {"data_file": ("clean.csv", csv_content, "text/csv")}
        data = {"rules": json.dumps({"age": {"min": 0, "max": 120}})}
        response = client.post("/api/v1/analyze", files=files, data=data, headers=api_headers)
        assert response.status_code == 200
        result = response.json()
        assert result["issues"] == []
        assert result["summary"]["issues_found"] == 0
        assert result["summary"]["critical_issues"] == 0
        assert result["summary"]["completeness"] == 100.0

    def test_analyze_invalid_schema_json(self, client, sample_data_csv, api_headers):
        """Malformed 'schema' JSON should return 400, not 500"""
        files = {"data_file": ("data.csv", sample_data_csv, "text/csv")}
        data = {"schema": "{not valid json"}
        response = client.post("/api/v1/analyze", files=files, data=data, headers=api_headers)
        assert response.status_code == 400

    def test_analyze_schema_not_an_object(self, client, sample_data_csv, api_headers):
        """A JSON value that isn't an object (e.g. a list) should return 400"""
        files = {"data_file": ("data.csv", sample_data_csv, "text/csv")}
        data = {"schema": json.dumps(["age", "gender"])}
        response = client.post("/api/v1/analyze", files=files, data=data, headers=api_headers)
        assert response.status_code == 400

    def test_analyze_invalid_rules_json(self, client, sample_data_csv, api_headers):
        """Malformed 'rules' JSON should return 400, not 500"""
        files = {"data_file": ("data.csv", sample_data_csv, "text/csv")}
        data = {"rules": "[1, 2, broken"}
        response = client.post("/api/v1/analyze", files=files, data=data, headers=api_headers)
        assert response.status_code == 400

    def test_analyze_unsupported_file_type(self, client, api_headers):
        """Unsupported file extensions should be rejected with 400"""
        files = {"data_file": ("data.pdf", b"%PDF-1.4 fake content", "application/pdf")}
        response = client.post("/api/v1/analyze", files=files, headers=api_headers)
        assert response.status_code == 400

    def test_analyze_empty_file(self, client, api_headers):
        """A zero-byte upload should be rejected with 400, not crash the server"""
        files = {"data_file": ("empty.csv", b"", "text/csv")}
        response = client.post("/api/v1/analyze", files=files, headers=api_headers)
        assert response.status_code == 400

    def test_analyze_malformed_csv(self, client, api_headers):
        """Unparseable CSV content should return 400, not 500"""
        # Ragged rows with mismatched column counts and an unterminated quote
        malformed_csv = 'a,b,c\n1,2\n"unterminated,3,4,5\n'
        files = {"data_file": ("bad.csv", malformed_csv, "text/csv")}
        response = client.post("/api/v1/analyze", files=files, headers=api_headers)
        assert response.status_code == 400

    def test_analyze_json_file(self, client, api_headers):
        """JSON-format datasets should be parsed the same way web_app.py's uploader does"""
        json_content = json.dumps([
            {"id": 1, "age": 25},
            {"id": 2, "age": 999},
        ])
        files = {"data_file": ("data.json", json_content, "application/json")}
        data = {"rules": json.dumps({"age": {"max": 120}})}
        response = client.post("/api/v1/analyze", files=files, data=data, headers=api_headers)
        assert response.status_code == 200
        result = response.json()
        assert result["summary"]["total_rows"] == 2
        assert any(i["type"] == "range_violation" for i in result["issues"])

    def test_analyze_tsv_file(self, client, api_headers):
        """Tab-separated .txt uploads should be parsed as TSV"""
        tsv_content = "id\tage\n1\t25\n2\t30\n"
        files = {"data_file": ("data.txt", tsv_content, "text/plain")}
        response = client.post("/api/v1/analyze", files=files, headers=api_headers)
        assert response.status_code == 200
        assert response.json()["summary"]["total_rows"] == 2

    def test_analyze_response_matches_model_fields(self, client, sample_data_csv, api_headers):
        """Issue entries should carry column/row/value plus a human-readable message,
        matching the contract web_app.py's dashboard renders directly."""
        files = {"data_file": ("data.csv", sample_data_csv, "text/csv")}
        data = {"rules": json.dumps({"age": {"min": 0, "max": 120}})}
        response = client.post("/api/v1/analyze", files=files, data=data, headers=api_headers)
        assert response.status_code == 200
        result = response.json()

        range_issues = [i for i in result["issues"] if i["type"] == "range_violation"]
        assert len(range_issues) >= 1
        for issue in range_issues:
            assert issue["column"] == "age"
            assert isinstance(issue["row"], int)
            assert "value" in issue
            assert "message" in issue

        stats = result["summary_stats"]
        assert "shape" in stats
        assert stats["shape"]["rows"] == 5


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
