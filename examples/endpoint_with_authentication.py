"""
Example: How to Add Authentication to Your Endpoints

This file demonstrates how to use the authentication system
in your FastAPI endpoints.

Copy these patterns when creating new endpoints.
"""

from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from typing import Optional

# Import authentication dependencies from api_server
from api_server import verify_api_key, verify_admin_password

# Example models
class AnalyzeRequest(BaseModel):
    data_file: str
    dictionary_file: Optional[str] = None

class ProgramRequest(BaseModel):
    name: str
    rules: dict

# Create app instance (in real code, this is already done in api_server.py)
app = FastAPI()


# ============================================================================
# Example 1: Public Endpoint (No Authentication)
# ============================================================================

@app.get("/api/v1/health")
async def health_check():
    """
    Public endpoint - anyone can access

    Use for: Health checks, status endpoints, public information
    """
    return {"status": "healthy"}


# ============================================================================
# Example 2: Protected Endpoint (API Key Required)
# ============================================================================

@app.post("/api/v1/analyze")
async def analyze_data(
    request: AnalyzeRequest,
    api_key: str = Depends(verify_api_key)  # <-- Add this parameter
):
    """
    Protected endpoint - requires valid API key

    Use for: Regular API operations, data analysis, queries

    The api_key parameter is automatically populated from the X-API-Key header.
    If authentication fails, FastAPI returns 401 or 403 automatically.
    You don't need to check the api_key value - its presence means auth succeeded.
    """
    # Your endpoint logic here
    # The api_key parameter confirms authentication succeeded

    return {
        "status": "success",
        "message": f"Data analyzed successfully",
        # Don't include api_key in response
    }


# ============================================================================
# Example 3: Admin Endpoint (Admin Password Required)
# ============================================================================

@app.delete("/api/v1/programs/{program_id}")
async def delete_program(
    program_id: str,
    admin_password: str = Depends(verify_admin_password)  # <-- Add this parameter
):
    """
    Admin endpoint - requires admin password

    Use for: Administrative operations, program management, system configuration

    The admin_password parameter is automatically populated from the X-Admin-Password header.
    Only use this for destructive or administrative operations.
    """
    # Your endpoint logic here
    # The admin_password parameter confirms admin auth succeeded

    return {
        "status": "success",
        "message": f"Program {program_id} deleted",
        # Don't include admin_password in response
    }


# ============================================================================
# Example 4: Endpoint with Both Authentication and Business Logic Validation
# ============================================================================

@app.post("/api/v1/programs")
async def create_program(
    request: ProgramRequest,
    api_key: str = Depends(verify_api_key)  # Authentication
):
    """
    Shows how to combine authentication with business logic validation

    1. Authentication happens first (via Depends)
    2. Then your business logic runs
    3. You can still raise HTTPException for business logic errors
    """
    # Authentication already passed if we got here

    # Business logic validation
    if not request.name:
        raise HTTPException(
            status_code=400,
            detail="Program name is required"
        )

    if len(request.rules) == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one rule is required"
        )

    # Your endpoint logic here
    return {
        "status": "success",
        "program_id": "abc123",
        "message": "Program created successfully"
    }


# ============================================================================
# Example 5: Client Usage Examples
# ============================================================================

def example_client_usage():
    """
    Examples of how clients should call authenticated endpoints
    """
    import requests

    # Public endpoint - no authentication
    response = requests.get("http://localhost:8000/api/v1/health")

    # Protected endpoint - API key required
    headers = {"X-API-Key": "your-api-key-here"}
    response = requests.post(
        "http://localhost:8000/api/v1/analyze",
        json={"data_file": "data.csv"},
        headers=headers
    )

    # Admin endpoint - admin password required
    admin_headers = {"X-Admin-Password": "your-admin-password-here"}
    response = requests.delete(
        "http://localhost:8000/api/v1/programs/abc123",
        headers=admin_headers
    )


# ============================================================================
# Example 6: Testing Authenticated Endpoints
# ============================================================================

def test_authenticated_endpoint():
    """
    Example of how to test authenticated endpoints
    """
    from fastapi.testclient import TestClient
    import os
    from unittest.mock import patch

    # Set up test environment with authentication
    with patch.dict(os.environ, {
        "DATA_ANALYZER_API_KEY": "test-key",
        "DATA_ANALYZER_ADMIN_PASSWORD": "test-password",
    }):
        client = TestClient(app)

        # Test with valid API key
        response = client.post(
            "/api/v1/analyze",
            json={"data_file": "test.csv"},
            headers={"X-API-Key": "test-key"}
        )
        assert response.status_code == 200

        # Test with invalid API key
        response = client.post(
            "/api/v1/analyze",
            json={"data_file": "test.csv"},
            headers={"X-API-Key": "wrong-key"}
        )
        assert response.status_code == 403

        # Test without API key
        response = client.post(
            "/api/v1/analyze",
            json={"data_file": "test.csv"}
        )
        assert response.status_code == 401


# ============================================================================
# Quick Reference
# ============================================================================

"""
QUICK REFERENCE - Copy these lines into your endpoints:

1. Public endpoint (no auth):
   @app.get("/api/v1/endpoint")
   async def my_endpoint():
       pass

2. Protected endpoint (API key):
   @app.post("/api/v1/endpoint")
   async def my_endpoint(api_key: str = Depends(verify_api_key)):
       pass

3. Admin endpoint (admin password):
   @app.delete("/api/v1/endpoint")
   async def my_endpoint(admin_password: str = Depends(verify_admin_password)):
       pass

CLIENT USAGE:

cURL:
  curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/endpoint
  curl -H "X-Admin-Password: your-password" http://localhost:8000/api/v1/endpoint

Python:
  headers = {"X-API-Key": "your-key"}
  requests.post("http://localhost:8000/api/v1/endpoint", headers=headers)

JavaScript:
  fetch('http://localhost:8000/api/v1/endpoint', {
    headers: {'X-API-Key': 'your-key'}
  })
"""
