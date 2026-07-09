#!/usr/bin/env python
"""
Manual Authentication Testing Script

This script demonstrates the authentication system in action with real HTTP requests.
It's useful for manual verification and understanding how to use the API.

Usage:
    python tests/manual_test_auth.py

Requirements:
    - API server running on localhost:8000
    - Environment variables set in .env file
"""

import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("DATA_ANALYZER_API_KEY", "your-api-key-here")
ADMIN_PASSWORD = os.getenv("DATA_ANALYZER_ADMIN_PASSWORD", "your-admin-password-here")


def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_health_endpoint():
    """Test health endpoint (no authentication required)"""
    print_section("Test 1: Health Endpoint (No Auth Required)")

    response = requests.get(f"{API_BASE_URL}/api/v1/health")

    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    if response.status_code == 200:
        print("✓ PASS: Health endpoint accessible without authentication")
    else:
        print("✗ FAIL: Health endpoint should be accessible")


def test_protected_endpoint_no_key():
    """Test accessing protected endpoint without API key"""
    print_section("Test 2: Protected Endpoint - No API Key")

    # NOTE: This is a placeholder - replace with actual protected endpoint when available
    print("Skipped: No protected endpoints implemented yet")
    print("Expected behavior: 401 Unauthorized")


def test_protected_endpoint_invalid_key():
    """Test accessing protected endpoint with invalid API key"""
    print_section("Test 3: Protected Endpoint - Invalid API Key")

    # NOTE: This is a placeholder - replace with actual protected endpoint when available
    print("Skipped: No protected endpoints implemented yet")
    print("Expected behavior: 403 Forbidden")


def test_protected_endpoint_valid_key():
    """Test accessing protected endpoint with valid API key"""
    print_section("Test 4: Protected Endpoint - Valid API Key")

    # NOTE: This is a placeholder - replace with actual protected endpoint when available
    print("Skipped: No protected endpoints implemented yet")
    print("Expected behavior: 200 OK with data")


def test_admin_endpoint_no_password():
    """Test accessing admin endpoint without password"""
    print_section("Test 5: Admin Endpoint - No Password")

    # NOTE: This is a placeholder - replace with actual admin endpoint when available
    print("Skipped: No admin endpoints implemented yet")
    print("Expected behavior: 401 Unauthorized")


def test_admin_endpoint_invalid_password():
    """Test accessing admin endpoint with invalid password"""
    print_section("Test 6: Admin Endpoint - Invalid Password")

    # NOTE: This is a placeholder - replace with actual admin endpoint when available
    print("Skipped: No admin endpoints implemented yet")
    print("Expected behavior: 403 Forbidden")


def test_admin_endpoint_valid_password():
    """Test accessing admin endpoint with valid password"""
    print_section("Test 7: Admin Endpoint - Valid Password")

    # NOTE: This is a placeholder - replace with actual admin endpoint when available
    print("Skipped: No admin endpoints implemented yet")
    print("Expected behavior: 200 OK with data")


def main():
    """Run all manual tests"""
    print("\n" + "=" * 70)
    print("  Manual Authentication Testing")
    print("  API Server: " + API_BASE_URL)
    print("=" * 70)

    try:
        # Test 1: Health endpoint (should work without auth)
        test_health_endpoint()

        # Test 2-7: Protected and admin endpoints
        # These will be implemented when actual protected endpoints exist
        test_protected_endpoint_no_key()
        test_protected_endpoint_invalid_key()
        test_protected_endpoint_valid_key()
        test_admin_endpoint_no_password()
        test_admin_endpoint_invalid_password()
        test_admin_endpoint_valid_password()

        print_section("Summary")
        print("Authentication system is ready for use!")
        print("\nNext steps:")
        print("1. Add Depends(verify_api_key) to protected endpoints")
        print("2. Add Depends(verify_admin_password) to admin endpoints")
        print("3. Update this script with actual endpoint URLs")

    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Could not connect to API server")
        print(f"Make sure the server is running on {API_BASE_URL}")
        print("\nStart the server with:")
        print("  uvicorn api_server:app --reload")


if __name__ == "__main__":
    main()
