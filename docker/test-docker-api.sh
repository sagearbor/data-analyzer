#!/bin/bash
# Test script for Docker API deployment
# This script verifies that the Docker image builds and runs correctly

set -e

echo "============================================================"
echo "Docker API Test Script"
echo "============================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="data-analyzer-api:latest"
CONTAINER_NAME="data-analyzer-api-test"
TEST_PORT=9000

echo ""
echo "Step 1: Building Docker image..."
if docker build -f Dockerfile.api -t $IMAGE_NAME . > /tmp/docker-build-test.log 2>&1; then
    echo -e "${GREEN}✓${NC} Image built successfully"

    # Get image size
    IMAGE_SIZE=$(docker images $IMAGE_NAME --format "{{.Size}}")
    echo "  Image size: $IMAGE_SIZE"
else
    echo -e "${RED}✗${NC} Image build failed"
    echo "  See /tmp/docker-build-test.log for details"
    exit 1
fi

echo ""
echo "Step 2: Checking for .env file..."
if [ -f .env ]; then
    echo -e "${GREEN}✓${NC} .env file found"
else
    echo -e "${YELLOW}!${NC} .env file not found (will use defaults)"
fi

echo ""
echo "Step 3: Starting container..."
# Remove any existing test container
docker rm -f $CONTAINER_NAME 2>/dev/null || true

# Start container
if [ -f .env ]; then
    docker run -d --name $CONTAINER_NAME --env-file .env -p $TEST_PORT:8000 $IMAGE_NAME
else
    docker run -d --name $CONTAINER_NAME -p $TEST_PORT:8000 $IMAGE_NAME
fi

echo -e "${GREEN}✓${NC} Container started on port $TEST_PORT"

echo ""
echo "Step 4: Waiting for container to become healthy..."
MAX_WAIT=60
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if docker inspect --format='{{.State.Health.Status}}' $CONTAINER_NAME 2>/dev/null | grep -q "healthy"; then
        echo -e "${GREEN}✓${NC} Container is healthy"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo -n "."
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${RED}✗${NC} Container failed to become healthy"
    echo "  Container logs:"
    docker logs $CONTAINER_NAME
    docker rm -f $CONTAINER_NAME
    exit 1
fi

echo ""
echo "Step 5: Testing health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:$TEST_PORT/api/v1/health)
if echo $HEALTH_RESPONSE | grep -q '"version"'; then
    echo -e "${GREEN}✓${NC} Health endpoint responding"
    echo "  Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}✗${NC} Health endpoint not responding correctly"
    docker logs $CONTAINER_NAME
    docker rm -f $CONTAINER_NAME
    exit 1
fi

echo ""
echo "Step 6: Verifying security (non-root user)..."
CURRENT_USER=$(docker exec $CONTAINER_NAME whoami)
if [ "$CURRENT_USER" = "apiuser" ]; then
    echo -e "${GREEN}✓${NC} Running as non-root user: $CURRENT_USER"
else
    echo -e "${YELLOW}!${NC} Warning: Running as user: $CURRENT_USER"
fi

echo ""
echo "Step 7: Verifying volume mount..."
docker exec $CONTAINER_NAME ls -la /app/data/ > /dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Volume mount accessible"
else
    echo -e "${RED}✗${NC} Volume mount not accessible"
fi

echo ""
echo "Step 8: Checking container logs..."
echo "  Last 10 log lines:"
docker logs --tail 10 $CONTAINER_NAME | sed 's/^/    /'

echo ""
echo "Step 9: Cleanup..."
docker stop $CONTAINER_NAME > /dev/null
docker rm $CONTAINER_NAME > /dev/null
echo -e "${GREEN}✓${NC} Test container removed"

echo ""
echo "============================================================"
echo -e "${GREEN}All tests passed!${NC}"
echo "============================================================"
echo ""
echo "To run the API in production:"
echo "  docker run -d --name data-analyzer-api --env-file .env -p 8000:8000 $IMAGE_NAME"
echo ""
echo "To run with docker-compose:"
echo "  docker-compose -f docker-compose.api.yml up -d"
echo ""
