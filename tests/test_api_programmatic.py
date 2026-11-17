"""
Programmatic-access API tests (ST-365)

Verifies the pieces that make the data analyzer callable by other programs:
- ST-359: POST /api/v1/analyze/json (dataset sent inline as JSON)
- ST-360: GET /api/v1/version, health reports the real version, request size limit
- ST-361: examples/api_client.py

Fully offline: FastAPI TestClient only, no network, no LLM. All data below is
synthetic and made up for these tests.

Run with:
    pytest tests/test_api_programmatic.py --no-cov -q
"""

import base64
import importlib.util
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.__version__ import __version__

REPO_ROOT = Path(__file__).resolve().parent.parent
# The fake key tests/conftest.py pins before any test module is imported.
TEST_API_KEY = os.environ["DATA_ANALYZER_API_KEY"]

# Synthetic widget inventory. Row 3 (index 2) has a planted out-of-range
# quantity (950) and row 4 (index 3) a planted disallowed colour ("plaid").
SYNTHETIC_CSV = (
    "widget_id,colour,quantity,unit_price\n"
    "1,red,10,2.50\n"
    "2,blue,20,3.75\n"
    "3,green,950,1.25\n"
    "4,plaid,15,4.00\n"
    "5,red,30,2.50\n"
)
SYNTHETIC_SCHEMA = {"widget_id": "int", "colour": "str", "quantity": "int", "unit_price": "float"}
SYNTHETIC_RULES = {
    "quantity": {"min": 0, "max": 100},
    "colour": {"allowed": ["red", "blue", "green"]},
}


# Mock environment before importing api_server (same pattern as tests/test_api.py)
@pytest.fixture(scope="module")
def mock_env():
    """Set up test environment variables"""
    with patch.dict(os.environ, {
        "DATA_ANALYZER_API_KEY": TEST_API_KEY,
        "DATA_ANALYZER_ADMIN_PASSWORD": "test-admin-pass-67890",
        "APP_ENV": "test",
    }):
        yield


@pytest.fixture(scope="module")
def client(mock_env):
    """Create test client with authentication configured"""
    from api_server import app
    return TestClient(app)


@pytest.fixture
def api_headers():
    """Standard API headers with valid API key"""
    return {"X-API-Key": TEST_API_KEY}


@pytest.fixture(scope="module")
def api_client_module():
    """Load examples/api_client.py (examples/ is not a package)."""
    path = REPO_ROOT / "examples" / "api_client.py"
    spec = importlib.util.spec_from_file_location("example_api_client", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_error_response(response, status_code):
    """Assert a response is the standard JSON ErrorResponse, never HTML."""
    assert response.status_code == status_code
    assert response.headers["content-type"].startswith("application/json")
    body = response.json()
    assert isinstance(body["error"], str) and body["error"]
    assert "timestamp" in body
    return body


# ============================================================================
# ST-359: POST /api/v1/analyze/json
# ============================================================================


class TestAnalyzeJsonRoute:
    """ST-359: inline JSON analysis route"""

    def test_ST_359_json_route_csv_text_happy_path(self, client, api_headers):
        """ST-359: CSV text in a JSON body is analyzed and returns the report"""
        response = client.post(
            "/api/v1/analyze/json",
            json={"data_content": SYNTHETIC_CSV},
            headers=api_headers,
        )
        assert response.status_code == 200
        data = response.json()
        assert set(data) >= {"summary", "issues", "recommendations", "quality_checks", "summary_stats"}
        assert data["summary"]["total_rows"] == 5
        assert data["summary"]["total_columns"] == 4

    def test_ST_359_json_route_base64_input(self, client, api_headers):
        """ST-359: base64 data_content gives the same report as plain text"""
        encoded = base64.b64encode(SYNTHETIC_CSV.encode("utf-8")).decode("ascii")
        as_b64 = client.post(
            "/api/v1/analyze/json",
            json={"data_content": encoded, "encoding": "base64", "format": "csv"},
            headers=api_headers,
        )
        as_text = client.post(
            "/api/v1/analyze/json",
            json={"data_content": SYNTHETIC_CSV, "format": "csv"},
            headers=api_headers,
        )
        assert as_b64.status_code == 200
        assert as_b64.json() == as_text.json()

    def test_ST_359_json_route_tsv_and_json_formats(self, client, api_headers):
        """ST-359: format=tsv and format=json are parsed as such"""
        tsv = client.post(
            "/api/v1/analyze/json",
            json={"data_content": SYNTHETIC_CSV.replace(",", "\t"), "format": "tsv"},
            headers=api_headers,
        )
        records = '[{"widget_id": 1, "quantity": 10}, {"widget_id": 2, "quantity": 20}]'
        as_json = client.post(
            "/api/v1/analyze/json",
            json={"data_content": records, "format": "json"},
            headers=api_headers,
        )
        assert tsv.status_code == 200
        assert tsv.json()["summary"]["total_columns"] == 4
        assert as_json.status_code == 200
        assert as_json.json()["summary"]["total_rows"] == 2

    def test_ST_359_json_route_schema_rules_find_planted_issue(self, client, api_headers):
        """ST-359: schema + rules flag the planted range and categorical violations"""
        response = client.post(
            "/api/v1/analyze/json",
            json={
                "data_content": SYNTHETIC_CSV,
                "schema": SYNTHETIC_SCHEMA,
                "rules": SYNTHETIC_RULES,
            },
            headers=api_headers,
        )
        assert response.status_code == 200
        issues = response.json()["issues"]
        range_hits = [i for i in issues if i["type"] == "range_violation"]
        colour_hits = [i for i in issues if i["type"] == "invalid_categorical_value"]
        assert [(i["column"], i["row"], i["value"]) for i in range_hits] == [("quantity", 2, 950)]
        assert [(i["column"], i["row"], i["value"]) for i in colour_hits] == [("colour", 3, "plaid")]
        assert response.json()["summary"]["critical_issues"] == 2

    def test_ST_359_json_route_matches_multipart_route(self, client, api_headers):
        """ST-359: JSON route and multipart route return identical reports"""
        import json as _json

        via_json = client.post(
            "/api/v1/analyze/json",
            json={
                "data_content": SYNTHETIC_CSV,
                "schema": SYNTHETIC_SCHEMA,
                "rules": SYNTHETIC_RULES,
                "min_rows": 3,
            },
            headers=api_headers,
        )
        via_multipart = client.post(
            "/api/v1/analyze",
            files={"data_file": ("widgets.csv", SYNTHETIC_CSV.encode("utf-8"), "text/csv")},
            data={
                "schema": _json.dumps(SYNTHETIC_SCHEMA),
                "rules": _json.dumps(SYNTHETIC_RULES),
                "min_rows": "3",
            },
            headers=api_headers,
        )
        assert via_json.status_code == 200
        assert via_multipart.status_code == 200
        assert via_json.json() == via_multipart.json()
        assert via_json.json()["summary"]["issues_found"] > 0  # not trivially equal

    def test_ST_359_json_route_missing_body_returns_json_error(self, client, api_headers):
        """ST-359: no body / missing data_content -> 422 ErrorResponse"""
        no_body = client.post("/api/v1/analyze/json", headers=api_headers)
        body = assert_error_response(no_body, 422)
        assert isinstance(body["detail"], str)

        missing_field = client.post(
            "/api/v1/analyze/json", json={"format": "csv"}, headers=api_headers
        )
        body = assert_error_response(missing_field, 422)
        assert "data_content" in body["detail"]

    def test_ST_359_json_route_invalid_body_returns_json_error(self, client, api_headers):
        """ST-359: malformed JSON / bad format / non-object rules -> 422 ErrorResponse"""
        malformed = client.post(
            "/api/v1/analyze/json",
            content="{not json",
            headers={**api_headers, "Content-Type": "application/json"},
        )
        assert_error_response(malformed, 422)

        bad_format = client.post(
            "/api/v1/analyze/json",
            json={"data_content": SYNTHETIC_CSV, "format": "parquet"},
            headers=api_headers,
        )
        body = assert_error_response(bad_format, 422)
        assert "format" in body["detail"]
        # the caller's dataset must not be echoed back in the error
        assert "plaid" not in bad_format.text

        bad_rules = client.post(
            "/api/v1/analyze/json",
            json={"data_content": SYNTHETIC_CSV, "rules": "not-an-object"},
            headers=api_headers,
        )
        assert_error_response(bad_rules, 422)

    def test_ST_359_json_route_bad_content_returns_400_json_error(self, client, api_headers):
        """ST-359: invalid base64 or unparseable content -> 400 ErrorResponse"""
        bad_b64 = client.post(
            "/api/v1/analyze/json",
            json={"data_content": "@@not base64@@", "encoding": "base64"},
            headers=api_headers,
        )
        body = assert_error_response(bad_b64, 400)
        assert "base64" in body["error"]

        bad_json_data = client.post(
            "/api/v1/analyze/json",
            json={"data_content": "this is not json", "format": "json"},
            headers=api_headers,
        )
        assert_error_response(bad_json_data, 400)

    def test_ST_359_json_route_requires_api_key(self, client):
        """ST-359: no key -> 401, wrong key -> 403 (same as the multipart route)"""
        payload = {"data_content": SYNTHETIC_CSV}
        missing = client.post("/api/v1/analyze/json", json=payload)
        assert_error_response(missing, 401)
        assert missing.headers.get("WWW-Authenticate") == "ApiKey"

        wrong = client.post(
            "/api/v1/analyze/json", json=payload, headers={"X-API-Key": "wrong-key"}
        )
        assert_error_response(wrong, 403)

        multipart_missing = client.post(
            "/api/v1/analyze",
            files={"data_file": ("widgets.csv", SYNTHETIC_CSV.encode("utf-8"), "text/csv")},
        )
        assert multipart_missing.status_code == missing.status_code

    def test_ST_359_json_route_accepts_right_api_key(self, client, api_headers):
        """ST-359: the configured key is accepted"""
        response = client.post(
            "/api/v1/analyze/json", json={"data_content": SYNTHETIC_CSV}, headers=api_headers
        )
        assert response.status_code == 200

    def test_ST_359_other_routes_keep_default_422_body(self, client, api_headers):
        """ST-359: the ErrorResponse 422 handler is scoped to the JSON route only"""
        response = client.post("/api/v1/analyze", headers=api_headers)  # no file
        assert response.status_code == 422
        assert isinstance(response.json()["detail"], list)


# ============================================================================
# ST-360: version endpoint, health version, request size limit
# ============================================================================


class TestVersionAndLimits:
    """ST-360: version reporting and maximum request size"""

    def test_ST_360_version_endpoint_equals_package_version(self, client):
        """ST-360: GET /api/v1/version returns src.__version__, no key needed"""
        response = client.get("/api/v1/version")
        assert response.status_code == 200
        assert response.json()["version"] == __version__

    def test_ST_360_health_reports_same_version(self, client):
        """ST-360: health reports the package version, not a hard-coded one"""
        health = client.get("/api/v1/health").json()
        version = client.get("/api/v1/version").json()
        assert health["version"] == __version__ == version["version"]

    def test_ST_360_openapi_reports_package_version(self, client):
        """ST-360: the OpenAPI document carries the package version too"""
        spec = client.get("/api/v1/openapi.json").json()
        assert spec["info"]["version"] == __version__

    def test_ST_360_oversize_json_request_returns_413(self, client, api_headers, monkeypatch):
        """ST-360: JSON route rejects a body over the limit with 413 ErrorResponse"""
        monkeypatch.setenv("DATA_ANALYZER_MAX_UPLOAD_BYTES", "200")
        big = "widget_id,quantity\n" + "".join(f"{i},{i}\n" for i in range(200))
        response = client.post(
            "/api/v1/analyze/json", json={"data_content": big}, headers=api_headers
        )
        body = assert_error_response(response, 413)
        assert "200" in body["error"]
        assert response.headers.get("X-Content-Type-Options") == "nosniff"

    def test_ST_360_oversize_multipart_request_returns_413(self, client, api_headers, monkeypatch):
        """ST-360: multipart route rejects an upload over the limit with 413 ErrorResponse"""
        monkeypatch.setenv("DATA_ANALYZER_MAX_UPLOAD_BYTES", "200")
        big = "widget_id,quantity\n" + "".join(f"{i},{i}\n" for i in range(200))
        response = client.post(
            "/api/v1/analyze",
            files={"data_file": ("widgets.csv", big.encode("utf-8"), "text/csv")},
            headers=api_headers,
        )
        assert_error_response(response, 413)

    def test_ST_360_request_under_limit_is_accepted(self, client, api_headers, monkeypatch):
        """ST-360: a request under a configured limit still succeeds"""
        monkeypatch.setenv("DATA_ANALYZER_MAX_UPLOAD_BYTES", "5000")
        response = client.post(
            "/api/v1/analyze/json", json={"data_content": SYNTHETIC_CSV}, headers=api_headers
        )
        assert response.status_code == 200

    def test_ST_360_decoded_content_over_limit_returns_413(self, client, api_headers, monkeypatch):
        """ST-360: the limit also applies to the content itself, not only Content-Length"""
        import api_server

        # Bypass the Content-Length pre-check so only the in-handler check can fire.
        monkeypatch.setenv("DATA_ANALYZER_MAX_UPLOAD_BYTES", "100000")
        calls = iter([100000, 20])
        monkeypatch.setattr(api_server, "_max_request_bytes", lambda: next(calls, 20))
        response = client.post(
            "/api/v1/analyze/json", json={"data_content": SYNTHETIC_CSV}, headers=api_headers
        )
        assert_error_response(response, 413)

    def test_ST_360_new_routes_in_openapi(self, client):
        """ST-359/ST-360: both new routes are documented, JSON route with a request model"""
        spec = client.get("/api/v1/openapi.json").json()
        assert "get" in spec["paths"]["/api/v1/version"]
        post = spec["paths"]["/api/v1/analyze/json"]["post"]
        ref = post["requestBody"]["content"]["application/json"]["schema"]["$ref"]
        assert ref.endswith("/AnalyzeJsonRequest")
        props = spec["components"]["schemas"]["AnalyzeJsonRequest"]["properties"]
        assert {"data_content", "format", "encoding", "schema", "rules", "min_rows"} <= set(props)
        ok_ref = post["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        assert ok_ref.endswith("/QualityAnalysisResponse")
        assert "413" in post["responses"]


# ============================================================================
# ST-361: examples/api_client.py
# ============================================================================


class TestExampleClient:
    """ST-361: the example client works against the app (TestClient as session)"""

    BASE = "http://testserver"

    def test_ST_361_client_health_and_version(self, client, api_client_module):
        """ST-361: health() and version() return the parsed JSON"""
        health = api_client_module.health(base_url=self.BASE, session=client)
        version = api_client_module.version(base_url=self.BASE, session=client)
        assert health["version"] == __version__
        assert version["version"] == __version__

    def test_ST_361_client_analyze_json(self, client, api_client_module):
        """ST-361: analyze_json() posts inline content and returns the report"""
        report = api_client_module.analyze_json(
            SYNTHETIC_CSV,
            schema=SYNTHETIC_SCHEMA,
            rules=SYNTHETIC_RULES,
            base_url=self.BASE,
            api_key=TEST_API_KEY,
            session=client,
        )
        assert report["summary"]["total_rows"] == 5
        assert report["summary"]["critical_issues"] == 2

    def test_ST_361_client_analyze_file_matches_analyze_json(self, client, api_client_module, tmp_path):
        """ST-361: analyze_file() uploads a file and agrees with analyze_json()"""
        csv_path = tmp_path / "widgets.csv"
        csv_path.write_text(SYNTHETIC_CSV)
        common = dict(
            schema=SYNTHETIC_SCHEMA, rules=SYNTHETIC_RULES,
            base_url=self.BASE, api_key=TEST_API_KEY, session=client,
        )
        from_file = api_client_module.analyze_file(str(csv_path), **common)
        from_json = api_client_module.analyze_json(SYNTHETIC_CSV, **common)
        assert from_file == from_json

    def test_ST_361_client_reads_url_and_key_from_env(self, client, api_client_module, monkeypatch):
        """ST-361: base URL and API key default to the environment variables"""
        monkeypatch.setenv("DATA_ANALYZER_API_URL", self.BASE + "/")
        monkeypatch.setenv("DATA_ANALYZER_API_KEY", TEST_API_KEY)
        report = api_client_module.analyze_json(SYNTHETIC_CSV, session=client)
        assert report["summary"]["total_rows"] == 5

    def test_ST_361_client_raises_api_error_with_error_response(self, client, api_client_module):
        """ST-361: a non-2xx reply raises ApiError carrying the ErrorResponse"""
        with pytest.raises(api_client_module.ApiError) as excinfo:
            api_client_module.analyze_json(
                SYNTHETIC_CSV, base_url=self.BASE, api_key="wrong-key", session=client
            )
        assert excinfo.value.status_code == 403
        assert excinfo.value.payload["error"]
