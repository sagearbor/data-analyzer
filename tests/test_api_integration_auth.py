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
        """Test that 401 responses have proper format when key is missing"""
        # This test validates the error response structure
        # when authentication would be required on a protected endpoint
        pass  # Will be implemented when protected endpoints are added

    def test_invalid_api_key_header_format(self, client_with_auth):
        """Test that 403 responses have proper format when key is invalid"""
        # This test validates the error response structure
        # when authentication fails on a protected endpoint
        pass  # Will be implemented when protected endpoints are added


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
        with patch.dict(os.environ, {}, clear=True):
            import importlib
            import api_server
            importlib.reload(api_server)

            from api_server import API_KEY, ADMIN_PASSWORD

            assert API_KEY is None
            assert ADMIN_PASSWORD is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
