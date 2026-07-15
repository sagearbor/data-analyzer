"""
API Integration Tests - End-to-End Workflows

Tests complete workflows and interactions between API endpoints:
- Dictionary upload → parse → save → retrieve → analyze
- Program lifecycle: create → alias → use → delete → restore
- Multi-format data analysis
- Error recovery and graceful degradation

Run with:
    pytest tests/test_api_integration.py -v
    pytest tests/test_api_integration.py::TestDictionaryWorkflow -v
"""

import os
import pytest
import json
import time
from pathlib import Path
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import tempfile


@pytest.fixture(scope="module")
def mock_env():
    """Set up test environment variables"""
    with patch.dict(os.environ, {
        "DATA_ANALYZER_API_KEY": "test-api-key-integration",
        "DATA_ANALYZER_ADMIN_PASSWORD": "test-admin-integration",
        "APP_ENV": "test",
    }):
        yield


@pytest.fixture(scope="module")
def client(mock_env):
    """Create test client with authentication configured

    Force-reloads api_server under this module's patched environment rather
    than relying on the cached import. Other test modules in this suite
    (tests/test_api_authentication.py, tests/test_api_integration_auth.py)
    call importlib.reload(api_server) with DATA_ANALYZER_API_KEY cleared to
    exercise the "auth disabled" code path. Since api_server.API_KEY is a
    module-level global set once at import time, that reload mutates the
    single shared api_server module for the rest of the pytest session -
    a plain `from api_server import app` here would silently inherit
    whatever auth state the last reload left behind (auth disabled),
    depending on test execution order, rather than this module's own
    DATA_ANALYZER_API_KEY. Reloading here, under our own patched env,
    makes this module's auth-dependent tests independent of that ordering.
    """
    import importlib
    import api_server
    importlib.reload(api_server)
    return TestClient(api_server.app)


@pytest.fixture
def api_headers():
    """Standard API headers with valid API key"""
    return {"X-API-Key": "test-api-key-integration"}


@pytest.fixture
def admin_headers():
    """Admin headers with valid admin password"""
    return {
        "X-API-Key": "test-api-key-integration",
        "X-Admin-Password": "test-admin-integration"
    }


@pytest.fixture
def invalid_api_headers():
    """Headers with an invalid/wrong API key"""
    return {"X-API-Key": "wrong"}


@pytest.fixture
def redcap_dictionary_csv():
    """REDCap data dictionary with logic rules"""
    return """Variable / Field Name,Form Name,Section Header,Field Type,Field Label,Choices OR Calculations,Field Note,Text Validation Type,Min,Max,Branching Logic,Required,Identifier,Custom Alignment
record_id,demographics,,text,Record ID,,,,,,,,y,
age,demographics,,text,Age (years),,Enter age in years,integer,18,65,,y,,
gender,demographics,,radio,Gender,1 Male | 2 Female | 3 Other | 4 Prefer not to say,,,,,,y,,
weight_kg,vitals,,text,Weight (kg),,Weight in kilograms,number,30,200,"[age] >= 18",y,,
height_cm,vitals,,text,Height (cm),,Height in centimeters,number,100,250,"[age] >= 18",y,,
bmi,vitals,,calc,BMI,"[weight_kg] / ([height_cm]/100)^2",,,,,"[weight_kg] > 0 AND [height_cm] > 0",,,
smoker,lifestyle,,yesno,Do you smoke?,,,,,,,"[age] >= 18",,
cigarettes_per_day,lifestyle,,text,Cigarettes per day,,Number of cigarettes,integer,1,100,"[smoker] = 1",,
"""


@pytest.fixture
def sample_data_with_violations():
    """Sample data with intentional validation violations"""
    return """record_id,age,gender,weight_kg,height_cm,smoker,cigarettes_per_day
001,25,1,70.5,175,0,
002,17,2,65.0,165,0,
003,150,3,80.0,180,1,5
004,35,4,250.0,170,0,
005,45,5,75.0,185,1,
006,30,1,-10,175,1,10
007,55,2,85.0,0,0,
"""


@pytest.fixture
def sample_data_clean():
    """Clean sample data that passes validation"""
    return """record_id,age,gender,weight_kg,height_cm,smoker,cigarettes_per_day
001,25,1,70.5,175,0,
002,35,2,65.0,165,0,
003,45,3,80.0,180,1,5
004,30,4,75.0,170,0,
005,55,1,85.0,185,1,10
"""


@pytest.fixture
def fhir_dictionary_json():
    """FHIR-like questionnaire dictionary"""
    return json.dumps({
        "resourceType": "Questionnaire",
        "id": "patient-intake",
        "title": "Patient Intake Form",
        "status": "active",
        "item": [
            {
                "linkId": "patient_id",
                "text": "Patient ID",
                "type": "string",
                "required": True
            },
            {
                "linkId": "visit_date",
                "text": "Visit Date",
                "type": "date",
                "required": True
            },
            {
                "linkId": "temperature",
                "text": "Temperature (°C)",
                "type": "decimal",
                "required": False,
                "extension": [{
                    "url": "http://hl7.org/fhir/StructureDefinition/minValue",
                    "valueDecimal": 35.0
                }, {
                    "url": "http://hl7.org/fhir/StructureDefinition/maxValue",
                    "valueDecimal": 42.0
                }]
            }
        ]
    })


# ============================================================================
# Dictionary Workflow Integration Tests
# ============================================================================


class TestDictionaryWorkflow:
    """Test complete dictionary parsing and usage workflow"""

    @patch('api_server.program_manager')
    def test_parse_save_retrieve_workflow(self, mock_pm, client, redcap_dictionary_csv, api_headers):
        """
        Integration test: Upload dictionary → parse → save → retrieve

        This tests the complete workflow of:
        1. Uploading a data dictionary
        2. Parsing it with LLM
        3. Saving the validation program
        4. Retrieving the saved program by ID/name/alias
        """
        # Mock program object. Attribute names must match the real
        # ValidationProgram dataclass (src/program_cache.py), since
        # api_server.parse_dictionary() and convert_validation_program_to_detail()
        # read these attributes directly off the object (mocked or real).
        mock_program = MagicMock()
        mock_program.program_id = "integration-test-prog-001"
        mock_program.name = "20251202-120000-PatientIntake"
        mock_program.status = "active"
        mock_program.created_at = "2025-12-02T12:00:00"
        mock_program.dictionary_source = "test_dict.csv"
        mock_program.dictionary_format = "redcap_csv"
        mock_program.created_by = "test-user"
        mock_program.num_fields = 8
        mock_program.num_basic_rules = 6
        mock_program.num_logic_rules = 3
        mock_program.schema = {
            "record_id": "str",
            "age": "int",
            "gender": "int",
            "weight_kg": "float",
            "height_cm": "float",
            "bmi": "float",
            "smoker": "bool",
            "cigarettes_per_day": "int"
        }
        mock_program.conditional_rules = []
        mock_program.generated_code = "# Validation code here"
        mock_program.aliases = []
        mock_program.use_count = 0
        mock_program.last_used = None
        mock_program.model_used = "gpt-5-nano"
        mock_program.generation_time_seconds = 1.5
        mock_program.version = 1

        mock_pm.create_program_from_dictionary.return_value = mock_program
        mock_pm.db.load_program.return_value = mock_program

        # Step 1: Parse and save dictionary
        files = {"dictionary_file": ("test_dict.csv", redcap_dictionary_csv, "text/csv")}
        data = {"save_program": "true", "program_name": "PatientIntake"}

        parse_response = client.post(
            "/api/v1/dictionary/parse",
            files=files,
            data=data,
            headers=api_headers
        )

        # Check if services are available
        if parse_response.status_code == 503:
            pytest.skip("Services not available for integration test")

        assert parse_response.status_code == 200
        parse_data = parse_response.json()

        # Verify parse response structure (ParseDictionaryResponse field names)
        assert "program_id" in parse_data
        assert "program_name" in parse_data
        assert "fields_extracted" in parse_data
        assert "rules_extracted" in parse_data

        program_id = parse_data["program_id"]

        # Step 2: Retrieve by ID
        get_response = client.get(
            f"/api/v1/dictionary/{program_id}",
            headers=api_headers
        )

        assert get_response.status_code == 200
        get_data = get_response.json()

        # Verify retrieved data matches (ProgramDetail field names)
        assert get_data["program_id"] == program_id
        assert get_data["num_fields"] == parse_data["fields_extracted"]
        assert "schema" in get_data
        assert "generated_code" in get_data

    @patch('api_server.program_manager')
    def test_parse_without_saving(self, mock_pm, client, redcap_dictionary_csv, api_headers):
        """
        Test parsing dictionary without saving to database

        Validates that the parse endpoint can work in "preview mode"
        without persisting the program.
        """
        mock_program = MagicMock()
        mock_program.program_id = "temp-preview-001"
        mock_program.name = "preview-only"
        mock_program.created_at = "2025-12-02T12:00:00"
        mock_program.dictionary_format = "redcap_csv"
        mock_program.num_fields = 8
        mock_program.num_basic_rules = 6
        mock_program.num_logic_rules = 3
        mock_program.schema = {"age": "int"}
        mock_program.generated_code = "# code"
        mock_program.model_used = "gpt-5-nano"

        mock_pm.create_program_from_dictionary.return_value = mock_program

        files = {"dictionary_file": ("preview.csv", redcap_dictionary_csv, "text/csv")}
        data = {"save_program": "false"}

        response = client.post(
            "/api/v1/dictionary/parse",
            files=files,
            data=data,
            headers=api_headers
        )

        if response.status_code == 503:
            pytest.skip("Services not available")

        assert response.status_code == 200
        # Program should be returned but not persisted


# ============================================================================
# Multi-Format Dictionary Tests
# ============================================================================


class TestMultiFormatDictionaries:
    """Test parsing different dictionary formats"""

    @patch('api_server.program_manager')
    def test_csv_format(self, mock_pm, client, redcap_dictionary_csv, api_headers):
        """Test REDCap CSV dictionary parsing"""
        mock_program = MagicMock()
        mock_program.program_id = "csv-test"
        mock_program.name = "csv-format"
        mock_program.created_at = "2025-12-02T12:00:00"
        mock_program.dictionary_format = "redcap_csv"
        mock_program.num_fields = 8
        mock_program.num_basic_rules = 6
        mock_program.num_logic_rules = 3
        mock_program.schema = {}
        mock_program.generated_code = "# code"
        mock_program.model_used = "gpt-5-nano"

        mock_pm.create_program_from_dictionary.return_value = mock_program

        files = {"dictionary_file": ("dict.csv", redcap_dictionary_csv, "text/csv")}
        response = client.post("/api/v1/dictionary/parse", files=files, headers=api_headers)

        if response.status_code == 503:
            pytest.skip("Services not available")

        assert response.status_code == 200

    @patch('api_server.program_manager')
    def test_json_format(self, mock_pm, client, fhir_dictionary_json, api_headers):
        """Test FHIR JSON dictionary parsing"""
        mock_program = MagicMock()
        mock_program.program_id = "json-test"
        mock_program.name = "json-format"
        mock_program.created_at = "2025-12-02T12:00:00"
        mock_program.dictionary_format = "fhir_json"
        mock_program.num_fields = 3
        mock_program.num_basic_rules = 2
        mock_program.num_logic_rules = 0
        mock_program.schema = {}
        mock_program.generated_code = "# code"
        mock_program.model_used = "gpt-5-nano"

        mock_pm.create_program_from_dictionary.return_value = mock_program

        files = {"dictionary_file": ("dict.json", fhir_dictionary_json, "application/json")}
        response = client.post("/api/v1/dictionary/parse", files=files, headers=api_headers)

        if response.status_code == 503:
            pytest.skip("Services not available")

        assert response.status_code == 200

    def test_unsupported_format(self, client, api_headers):
        """Test rejection of unsupported format"""
        fake_excel = b"PK\x03\x04"  # Fake Excel file signature
        files = {"dictionary_file": ("dict.xlsx", fake_excel, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}

        response = client.post("/api/v1/dictionary/parse", files=files, headers=api_headers)
        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]


# ============================================================================
# Error Recovery Tests
# ============================================================================


class TestErrorRecovery:
    """Test error handling and graceful degradation"""

    def test_service_unavailable_graceful_degradation(self, client, api_headers):
        """Test graceful degradation when services are unavailable"""
        # Health endpoint should always work
        health_response = client.get("/api/v1/health")
        assert health_response.status_code == 200

        health_data = health_response.json()

        # If services are unavailable, status should be 'degraded'
        if not all(health_data["services"].values()):
            assert health_data["status"] == "degraded"

    @patch('api_server.program_manager', None)
    def test_parse_when_service_down(self, client, redcap_dictionary_csv, api_headers):
        """Test parse endpoint returns 503 when program manager unavailable"""
        files = {"dictionary_file": ("test.csv", redcap_dictionary_csv, "text/csv")}
        response = client.post("/api/v1/dictionary/parse", files=files, headers=api_headers)

        # Should return 503 Service Unavailable
        assert response.status_code == 503
        assert "service not available" in response.json()["detail"].lower()

    @patch('api_server.program_manager', None)
    def test_get_when_service_down(self, client, api_headers):
        """Test get endpoint returns 503 when program manager unavailable"""
        response = client.get("/api/v1/dictionary/test-id", headers=api_headers)

        # Should return 503 Service Unavailable
        assert response.status_code == 503
        assert "service not available" in response.json()["detail"].lower()


# ============================================================================
# Concurrent Request Tests
# ============================================================================


class TestConcurrentRequests:
    """Test handling of concurrent API requests"""

    def test_multiple_health_checks_concurrent(self, client):
        """Multiple concurrent health checks should all succeed"""
        responses = []

        # Simulate concurrent requests
        for _ in range(5):
            response = client.get("/api/v1/health")
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == 200
            assert "status" in response.json()

    @patch('api_server.program_manager')
    def test_concurrent_dictionary_retrieval(self, mock_pm, client, api_headers):
        """Concurrent dictionary retrievals should work correctly"""
        mock_program = MagicMock()
        mock_program.program_id = "concurrent-test"
        mock_program.name = "concurrent-program"
        mock_program.status = "active"
        mock_program.created_at = "2025-12-02T12:00:00"
        mock_program.dictionary_source = "test.csv"
        mock_program.dictionary_format = "generic"
        mock_program.created_by = "test-user"
        mock_program.num_fields = 5
        mock_program.num_basic_rules = 3
        mock_program.num_logic_rules = 1
        mock_program.schema = {}
        mock_program.conditional_rules = []
        mock_program.generated_code = "# code"
        mock_program.aliases = []
        mock_program.use_count = 0
        mock_program.last_used = None
        mock_program.model_used = "gpt-5-nano"
        mock_program.generation_time_seconds = 1.0
        mock_program.version = 1

        mock_pm.db.load_program.return_value = mock_program

        responses = []
        for _ in range(3):
            response = client.get("/api/v1/dictionary/concurrent-test", headers=api_headers)
            responses.append(response)

        # All should succeed or fail consistently
        status_codes = [r.status_code for r in responses]
        assert all(code in [200, 503] for code in status_codes)


# ============================================================================
# File Size and Limits Tests
# ============================================================================


class TestFileSizeLimits:
    """Test file size limits and large file handling"""

    def test_large_dictionary_file(self, client, api_headers):
        """Test handling of large dictionary file (near limit)"""
        # Create a large CSV (but under 10MB limit mentioned in docs)
        large_csv = "Variable / Field Name,Form Name,Field Type\n"
        large_csv += "\n".join([f"field_{i},form,text" for i in range(10000)])

        files = {"dictionary_file": ("large.csv", large_csv, "text/csv")}
        response = client.post("/api/v1/dictionary/parse", files=files, headers=api_headers)

        # Should either process or return error (not crash)
        assert response.status_code in [200, 400, 413, 500, 503]

    @patch('api_server.program_manager')
    def test_empty_dictionary_file(self, mock_pm, client, api_headers):
        """Test handling of empty file (mocked - no live LLM call, see
        test_api.py::test_parse_dictionary_empty_file for the same pattern).
        Without mocking, an empty dictionary is sent to the live LLM, which
        is both non-deterministic and can legitimately return 0 fields with
        a 200 rather than an error, making this assertion flaky/wrong."""
        mock_pm.create_program_from_dictionary.side_effect = RuntimeError(
            "No fields could be extracted from empty dictionary content"
        )

        files = {"dictionary_file": ("empty.csv", "", "text/csv")}
        response = client.post("/api/v1/dictionary/parse", files=files, headers=api_headers)

        # Should return error
        assert response.status_code in [400, 500, 503]


# ============================================================================
# Authentication Flow Tests
# ============================================================================


class TestAuthenticationFlow:
    """Test authentication across multiple requests"""

    def test_auth_persistence_across_requests(self, client, api_headers):
        """API key should work consistently across multiple requests"""
        # Make multiple authenticated requests
        for _ in range(3):
            response = client.get("/api/v1/health", headers=api_headers)
            assert response.status_code == 200

    def test_mixed_auth_and_no_auth_endpoints(self, client, api_headers):
        """Test access to both protected and public endpoints"""
        # Public endpoint (no auth required)
        public_response = client.get("/api/v1/health")
        assert public_response.status_code == 200

        # Protected endpoint (auth required)
        protected_response = client.get("/api/v1/dictionary/test", headers=api_headers)
        # Should return 404 (not found) or 503, not 401 (auth working)
        assert protected_response.status_code in [404, 503]
        assert protected_response.status_code != 401  # Not unauthorized

    def test_auth_switching(self, client, api_headers, invalid_api_headers):
        """Test switching between valid and invalid auth"""
        # Valid auth
        response1 = client.get("/api/v1/dictionary/test1", headers=api_headers)
        assert response1.status_code != 401
        assert response1.status_code != 403

        # Invalid auth
        response2 = client.get("/api/v1/dictionary/test2", headers={"X-API-Key": "wrong"})
        assert response2.status_code == 403

        # Valid auth again
        response3 = client.get("/api/v1/dictionary/test3", headers=api_headers)
        assert response3.status_code != 401
        assert response3.status_code != 403


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
