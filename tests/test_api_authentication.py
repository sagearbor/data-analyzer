"""
Test API Authentication System

Tests the API key and admin password authentication mechanisms in api_server.py.

Run with:
    pytest tests/test_api_authentication.py -v
"""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch


# Import the FastAPI app
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We need to mock the environment variables before importing api_server
@pytest.fixture
def test_env_vars():
    """Set up test environment variables"""
    with patch.dict(os.environ, {
        "DATA_ANALYZER_API_KEY": "test-api-key-12345",
        "DATA_ANALYZER_ADMIN_PASSWORD": "test-admin-password-67890",
    }):
        yield


@pytest.fixture
def client(test_env_vars):
    """Create a test client with authentication configured"""
    # Import after environment is set up
    from api_server import app
    return TestClient(app)


@pytest.fixture
def client_no_auth():
    """Create a test client with no authentication configured"""
    with patch.dict(os.environ, {}, clear=True):
        # Need to reload the module to pick up new environment
        import importlib
        import api_server
        importlib.reload(api_server)
        return TestClient(api_server.app)


class TestHealthEndpoint:
    """Test that health endpoint works without authentication"""

    def test_health_no_auth_required(self, client):
        """Health endpoint should work without API key"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "services" in data


class TestAPIKeyAuthentication:
    """Test API key authentication mechanism"""

    def test_missing_api_key(self, client):
        """Request without API key should return 401"""
        # Create a protected endpoint for testing (we'll use a placeholder)
        # For now, we'll test the verify_api_key dependency directly
        from api_server import verify_api_key, api_key_header
        from fastapi import Request

        # Test with no header
        with pytest.raises(Exception) as exc_info:
            import asyncio
            asyncio.run(verify_api_key(api_key=None))

        # Should raise HTTPException with 401
        assert "401" in str(exc_info.value) or "Authentication required" in str(exc_info.value)

    def test_invalid_api_key(self, client):
        """Request with invalid API key should return 403"""
        from api_server import verify_api_key

        with pytest.raises(Exception) as exc_info:
            import asyncio
            asyncio.run(verify_api_key(api_key="wrong-key"))

        # Should raise HTTPException with 403
        assert "403" in str(exc_info.value) or "Invalid credentials" in str(exc_info.value)

    def test_valid_api_key(self, client):
        """Request with valid API key should succeed"""
        from api_server import verify_api_key

        import asyncio
        result = asyncio.run(verify_api_key(api_key="test-api-key-12345"))
        assert result == "test-api-key-12345"

    def test_no_auth_configured(self, client_no_auth):
        """When API key not configured, authentication should be disabled"""
        from api_server import verify_api_key

        import asyncio
        result = asyncio.run(verify_api_key(api_key=None))
        assert result == "unauthenticated"


class TestAdminPasswordAuthentication:
    """Test admin password authentication mechanism"""

    def test_missing_admin_password(self, test_env_vars):
        """Request without admin password should return 401"""
        # Reimport to get the correct environment variables
        import importlib
        import api_server
        importlib.reload(api_server)
        from api_server import verify_admin_password

        with pytest.raises(Exception) as exc_info:
            import asyncio
            asyncio.run(verify_admin_password(admin_password=None))

        assert "401" in str(exc_info.value) or "Admin authentication required" in str(exc_info.value)

    def test_invalid_admin_password(self, test_env_vars):
        """Request with invalid admin password should return 403"""
        # Reimport to get the correct environment variables
        import importlib
        import api_server
        importlib.reload(api_server)
        from api_server import verify_admin_password

        with pytest.raises(Exception) as exc_info:
            import asyncio
            asyncio.run(verify_admin_password(admin_password="wrong-password"))

        assert "403" in str(exc_info.value) or "Invalid admin credentials" in str(exc_info.value)

    def test_valid_admin_password(self, test_env_vars):
        """Request with valid admin password should succeed"""
        # Reimport to get the correct environment variables
        import importlib
        import api_server
        importlib.reload(api_server)
        from api_server import verify_admin_password

        import asyncio
        result = asyncio.run(verify_admin_password(admin_password="test-admin-password-67890"))
        assert result == "test-admin-password-67890"

    def test_no_admin_auth_configured(self, client_no_auth):
        """When admin password not configured, authentication should be disabled"""
        from api_server import verify_admin_password

        import asyncio
        result = asyncio.run(verify_admin_password(admin_password=None))
        assert result == "unauthenticated"


class TestSecurityHeaders:
    """Test that proper WWW-Authenticate headers are returned"""

    def test_api_key_401_header(self, client):
        """401 response should include WWW-Authenticate header for API key"""
        from api_server import verify_api_key
        from fastapi import HTTPException

        import asyncio
        try:
            asyncio.run(verify_api_key(api_key=None))
        except HTTPException as e:
            assert e.status_code == 401
            assert "WWW-Authenticate" in e.headers
            assert e.headers["WWW-Authenticate"] == "ApiKey"

    def test_admin_password_401_header(self, client):
        """401 response should include WWW-Authenticate header for admin"""
        from api_server import verify_admin_password
        from fastapi import HTTPException

        import asyncio
        try:
            asyncio.run(verify_admin_password(admin_password=None))
        except HTTPException as e:
            assert e.status_code == 401
            assert "WWW-Authenticate" in e.headers
            assert e.headers["WWW-Authenticate"] == "AdminPassword"


class TestErrorMessages:
    """Test that error messages don't leak sensitive information"""

    def test_generic_error_messages(self, client):
        """Error messages should be generic and not reveal auth details"""
        from api_server import verify_api_key
        from fastapi import HTTPException

        import asyncio

        # Test missing key
        try:
            asyncio.run(verify_api_key(api_key=None))
        except HTTPException as e:
            assert "Authentication required" in e.detail
            assert "API_KEY" not in e.detail  # Don't reveal env var names
            assert "test-api-key" not in e.detail  # Don't reveal actual key

        # Test invalid key
        try:
            asyncio.run(verify_api_key(api_key="wrong"))
        except HTTPException as e:
            assert "Invalid credentials" in e.detail
            assert "test-api-key" not in e.detail  # Don't reveal actual key
            assert "wrong" not in e.detail  # Don't echo back the wrong key


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
