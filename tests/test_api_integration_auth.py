"""
Integration Tests for API Authentication

Demonstrates authentication working with actual API endpoints using real HTTP requests.

Run with:
    pytest tests/test_api_integration_auth.py -v
"""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


@pytest.fixture
def app_with_auth():
    """Create app instance with authentication configured"""
    with patch.dict(os.environ, {
        "DATA_ANALYZER_API_KEY": "test-api-key-integration",
        "DATA_ANALYZER_ADMIN_PASSWORD": "test-admin-password-integration",
    }):
        # Import after environment is set up
        import importlib
        import api_server
        importlib.reload(api_server)
        yield api_server.app


@pytest.fixture
def client_with_auth(app_with_auth):
    """Create test client with authentication configured"""
    return TestClient(app_with_auth)


class TestHealthEndpointIntegration:
    """Integration tests for health endpoint (no auth required)"""

    def test_health_endpoint_no_auth(self, client_with_auth):
        """Health endpoint should work without authentication"""
        response = client_with_auth.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "services" in data


class TestAuthenticationHeadersIntegration:
    """Integration tests for authentication headers"""

    def test_root_endpoint_no_auth(self, client_with_auth):
        """Root endpoint should work without authentication"""
        response = client_with_auth.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_missing_api_key_header_format(self, client_with_auth):
        """
        End-to-end: no X-API-Key header at all on a protected endpoint must be
        rejected with 401 before the handler runs, and the response must carry
        the WWW-Authenticate challenge header that verify_api_key sets.

        This exercises FastAPI's real APIKeyHeader/Depends wiring via an actual
        HTTP request (not a direct call to verify_api_key()), which is what
        Swagger UI and curl actually go through.
        """
        files = {"data_file": ("data.csv", b"a,b\n1,2\n", "text/csv")}
        response = client_with_auth.post("/api/v1/analyze", files=files)

        assert response.status_code == 401
        assert response.headers.get("www-authenticate") == "ApiKey"
        body = response.json()
        assert "detail" in body or "error" in body

    def test_invalid_api_key_header_format(self, client_with_auth):
        """
        End-to-end: a wrong X-API-Key value on a protected endpoint must be
        rejected with 403 (distinct from the 401 for a missing header),
        exercised via a real HTTP request through FastAPI's dependency wiring.
        """
        files = {"data_file": ("data.csv", b"a,b\n1,2\n", "text/csv")}
        response = client_with_auth.post(
            "/api/v1/analyze",
            files=files,
            headers={"X-API-Key": "definitely-wrong-key"},
        )

        assert response.status_code == 403
        body = response.json()
        assert "detail" in body or "error" in body

    def test_valid_api_key_reaches_handler(self, client_with_auth):
        """
        End-to-end: the correct X-API-Key value must be accepted (not 401/403)
        and the request must actually reach and complete the /api/v1/analyze
        handler, proving auth success is distinguishable from auth rejection.
        """
        files = {"data_file": ("data.csv", b"a,b\n1,2\n", "text/csv")}
        response = client_with_auth.post(
            "/api/v1/analyze",
            files=files,
            headers={"X-API-Key": "test-api-key-integration"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "summary" in data or "issues" in data or "quality_checks" in data

    def test_dictionary_endpoint_rejects_missing_and_wrong_key(self, client_with_auth):
        """
        End-to-end auth coverage for a SECOND protected route
        (/api/v1/dictionary/{dict_id}), proving the auth wiring gap isn't
        specific to /api/v1/analyze. Uses a nonexistent dict_id: if auth were
        bypassed, this would fall through to "not found" (404) instead of
        being rejected at the dependency layer (401/403).
        """
        no_header_response = client_with_auth.get("/api/v1/dictionary/nonexistent-id-xyz")
        assert no_header_response.status_code == 401
        assert no_header_response.headers.get("www-authenticate") == "ApiKey"

        wrong_key_response = client_with_auth.get(
            "/api/v1/dictionary/nonexistent-id-xyz",
            headers={"X-API-Key": "definitely-wrong-key"},
        )
        assert wrong_key_response.status_code == 403


class TestSecurityBestPractices:
    """Verify security best practices are followed"""

    def test_api_key_not_in_logs(self, client_with_auth, caplog):
        """Verify API keys are not logged in plain text"""
        # Attempt authentication with valid key
        from api_server import verify_api_key
        import asyncio

        asyncio.run(verify_api_key(api_key="test-api-key-integration"))

        # Check logs don't contain the actual key
        for record in caplog.records:
            assert "test-api-key-integration" not in record.message

    def test_admin_password_not_in_logs(self, client_with_auth, caplog):
        """Verify admin passwords are not logged in plain text"""
        from api_server import verify_admin_password
        import asyncio

        asyncio.run(verify_admin_password(admin_password="test-admin-password-integration"))

        # Check logs don't contain the actual password
        for record in caplog.records:
            assert "test-admin-password-integration" not in record.message

    def test_error_messages_dont_leak_credentials(self, client_with_auth):
        """Verify error messages don't leak credential information"""
        from api_server import verify_api_key
        from fastapi import HTTPException
        import asyncio

        # Test with wrong API key
        try:
            asyncio.run(verify_api_key(api_key="wrong-key"))
        except HTTPException as e:
            # Error message should be generic
            assert "wrong-key" not in e.detail
            assert "test-api-key-integration" not in e.detail
            assert "DATA_ANALYZER_API_KEY" not in e.detail


class TestAuthenticationConfiguration:
    """Test authentication configuration behavior"""

    def test_auth_enabled_when_configured(self, client_with_auth):
        """When API key is configured, authentication should be enforced"""
        from api_server import API_KEY, ADMIN_PASSWORD

        assert API_KEY is not None
        assert API_KEY == "test-api-key-integration"
        assert ADMIN_PASSWORD is not None
        assert ADMIN_PASSWORD == "test-admin-password-integration"

    def test_auth_disabled_when_not_configured(self):
        """When API key is not configured, authentication should be disabled"""
        # load_dotenv() in api_server re-reads the local .env file even with a
        # cleared os.environ — no-op it so the test is hermetic on dev machines.
        with patch.dict(os.environ, {}, clear=True), \
             patch("dotenv.load_dotenv", lambda *a, **k: None):
            import importlib
            import api_server
            importlib.reload(api_server)

            from api_server import API_KEY, ADMIN_PASSWORD

            assert API_KEY is None
            assert ADMIN_PASSWORD is None

    def test_unconfigured_auth_permits_request_without_header_end_to_end(self):
        """
        End-to-end: when DATA_ANALYZER_API_KEY is not configured at all, a
        protected endpoint must still succeed (200) for a request with NO
        X-API-Key header. This is the intentional "permissive when
        unconfigured" behavior (see verify_api_key's `if not API_KEY: return
        "unauthenticated"` branch) and must not regress into either an
        unconditional 401 (breaking unconfigured deployments) or the reported
        bypass shape (auth silently skipped even when configured).

        Uses a real HTTP request through the actual FastAPI dependency wiring,
        not a direct call to verify_api_key(), mirroring the other tests in
        this class.
        """
        with patch.dict(os.environ, {}, clear=True), \
             patch("dotenv.load_dotenv", lambda *a, **k: None):
            import importlib
            import api_server
            importlib.reload(api_server)

            assert api_server.API_KEY is None  # sanity check on the fixture setup

            client = TestClient(api_server.app)
            files = {"data_file": ("data.csv", b"a,b\n1,2\n", "text/csv")}
            response = client.post("/api/v1/analyze", files=files)

            assert response.status_code == 200
            data = response.json()
            assert "summary" in data or "issues" in data or "quality_checks" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
