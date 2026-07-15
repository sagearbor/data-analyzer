#!/bin/bash
# Basic API Server Testing Script
# Tests that the FastAPI server starts and basic endpoints work

set -e

echo "========================================"
echo "Data Analyzer API - Basic Test"
echo "========================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test 1: Start server in background
echo ""
echo "Starting API server on port 8000..."
source venv/bin/activate
python api_server.py > /tmp/api_test.log 2>&1 &
API_PID=$!
echo "Server started with PID: $API_PID"

# Wait for server to start
echo "Waiting for server to initialize..."
sleep 3

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping API server..."
    kill $API_PID 2>/dev/null || true
    echo "Cleanup complete"
}
trap cleanup EXIT

# Test 2: Health endpoint
echo ""
echo "Test 1: Health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/api/v1/health)
if echo "$HEALTH_RESPONSE" | grep -q '"status":"healthy"'; then
    echo -e "${GREEN}✓ Health endpoint working${NC}"
    echo "$HEALTH_RESPONSE" | python -m json.tool
else
    echo -e "${RED}✗ Health endpoint failed${NC}"
    exit 1
fi

# Test 3: Root endpoint
echo ""
echo "Test 2: Root endpoint..."
ROOT_RESPONSE=$(curl -s http://localhost:8000/)
if echo "$ROOT_RESPONSE" | grep -q '"docs":"/api/v1/docs"'; then
    echo -e "${GREEN}✓ Root endpoint working${NC}"
    echo "$ROOT_RESPONSE" | python -m json.tool
else
    echo -e "${RED}✗ Root endpoint failed${NC}"
    exit 1
fi

# Test 4: OpenAPI schema
echo ""
echo "Test 3: OpenAPI schema generation..."
OPENAPI_RESPONSE=$(curl -s http://localhost:8000/api/v1/openapi.json)
if echo "$OPENAPI_RESPONSE" | grep -q '"openapi":"3.1.0"'; then
    echo -e "${GREEN}✓ OpenAPI schema generated${NC}"
    echo "OpenAPI version: $(echo $OPENAPI_RESPONSE | python -c 'import sys, json; print(json.load(sys.stdin)["openapi"])')"
else
    echo -e "${RED}✗ OpenAPI schema failed${NC}"
    exit 1
fi

# Test 5: Swagger docs available
echo ""
echo "Test 4: Swagger documentation..."
DOCS_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/v1/docs)
if [ "$DOCS_RESPONSE" = "200" ]; then
    echo -e "${GREEN}✓ Swagger docs available at http://localhost:8000/api/v1/docs${NC}"
else
    echo -e "${RED}✗ Swagger docs failed (HTTP $DOCS_RESPONSE)${NC}"
    exit 1
fi

echo ""
echo "========================================"
echo -e "${GREEN}All basic tests passed!${NC}"
echo "========================================"
echo ""
echo "Server is running at: http://localhost:8000"
echo "Swagger docs: http://localhost:8000/api/v1/docs"
echo "Health check: http://localhost:8000/api/v1/health"
echo ""
echo "Server logs:"
tail -20 /tmp/api_test.log
