# Docker Deployment Guide - Data Analyzer REST API

This guide covers Docker-based deployment of the Data Analyzer REST API server.

## Table of Contents

- [Quick Start](#quick-start)
- [Build Instructions](#build-instructions)
- [Running the API](#running-the-api)
- [Environment Configuration](#environment-configuration)
- [Volume Management](#volume-management)
- [Health Checks](#health-checks)
- [Troubleshooting](#troubleshooting)
- [Multi-Container Deployment](#multi-container-deployment)
- [Production Considerations](#production-considerations)

---

## Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose 1.29+
- `.env` file with required credentials (see [Environment Configuration](#environment-configuration))

### Minimal Setup

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env and set required values:
#    - AZURE_OPENAI_API_KEY
#    - AZURE_OPENAI_ENDPOINT
#    - DATA_ANALYZER_API_KEY
#    - DATA_ANALYZER_ADMIN_PASSWORD

# 3. Build and start the API
docker-compose -f docker-compose.api.yml up -d

# 4. Verify API is running
curl http://localhost:8000/api/v1/health
```

The API will be available at `http://localhost:8000`

---

## Build Instructions

### Building the Image

```bash
# Build using docker-compose (recommended)
docker-compose -f docker-compose.api.yml build

# Or build directly with Docker
docker build -f Dockerfile.api -t data-analyzer-api:latest .
```

### Multi-stage Build Benefits

The `Dockerfile.api` uses a multi-stage build:

1. **Builder stage**: Installs dependencies and compiles packages
2. **Runtime stage**: Creates minimal production image

This results in:
- Smaller image size (~300-400MB vs 1GB+)
- Faster deployment and startup
- Reduced attack surface (no build tools in production)

### Build Arguments

```bash
# Specify Python version (default: 3.11)
docker build -f Dockerfile.api \
  --build-arg PYTHON_VERSION=3.10 \
  -t data-analyzer-api:latest .
```

---

## Running the API

### Production Mode

```bash
# Start in detached mode
docker-compose -f docker-compose.api.yml up -d

# View logs
docker-compose -f docker-compose.api.yml logs -f api

# Stop
docker-compose -f docker-compose.api.yml down
```

### Development Mode (Hot Reload)

```bash
# Start with development overrides
docker-compose -f docker-compose.api.yml -f docker-compose.dev.yml up

# This mounts your source code as volumes, enabling hot reload
# Changes to api_server.py or src/ will automatically restart the server
```

### Running Without Docker Compose

```bash
# Build image
docker build -f Dockerfile.api -t data-analyzer-api:latest .

# Run container
docker run -d \
  --name data-analyzer-api \
  -p 8000:8000 \
  --env-file .env \
  -v data-analyzer-program-cache:/app/data \
  data-analyzer-api:latest

# View logs
docker logs -f data-analyzer-api

# Stop and remove
docker stop data-analyzer-api
docker rm data-analyzer-api
```

---

## Environment Configuration

### Required Environment Variables

Create a `.env` file with the following variables:

```bash
# ============================================================================
# Azure OpenAI Configuration (Required)
# ============================================================================
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# ============================================================================
# API Security (Required)
# ============================================================================
# API key for client authentication
DATA_ANALYZER_API_KEY=your-secure-api-key-here

# Admin password for protected endpoints
DATA_ANALYZER_ADMIN_PASSWORD=your-secure-admin-password-here

# ============================================================================
# Application Configuration (Optional)
# ============================================================================
# Environment: dev, staging, prod
APP_ENV=prod

# Logging level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO

# Program cache directory (inside container)
PROGRAM_CACHE_DIR=/app/data/programs
```

### Security Best Practices

1. **Never commit .env to version control**
   ```bash
   # Ensure .env is in .gitignore
   echo ".env" >> .gitignore
   ```

2. **Use strong credentials**
   ```bash
   # Generate secure API key
   openssl rand -base64 32

   # Generate secure admin password
   openssl rand -base64 24
   ```

3. **Use Docker secrets for production**
   ```yaml
   # docker-compose.api.yml (production variant)
   services:
     api:
       secrets:
         - azure_openai_api_key
         - api_key

   secrets:
     azure_openai_api_key:
       external: true
     api_key:
       external: true
   ```

---

## Volume Management

### Program Cache Volume

The API uses a named volume to persist validation programs:

```bash
# List volumes
docker volume ls | grep data-analyzer

# Inspect volume
docker volume inspect data-analyzer-program-cache

# Backup volume
docker run --rm \
  -v data-analyzer-program-cache:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/program-cache-backup.tar.gz -C /data .

# Restore volume
docker run --rm \
  -v data-analyzer-program-cache:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/program-cache-backup.tar.gz -C /data

# Remove volume (WARNING: deletes all cached programs)
docker-compose -f docker-compose.api.yml down -v
```

### Log Volume (Development)

In development mode, logs are mounted to `./logs`:

```bash
# View logs directory
ls -la ./logs/

# Clear logs
rm -rf ./logs/*
```

---

## Health Checks

### API Health Endpoint

```bash
# Check health
curl http://localhost:8000/api/v1/health

# Expected response
{
  "status": "healthy",
  "timestamp": "2025-12-02T12:00:00Z",
  "services": {
    "llm_client": "available",
    "program_manager": "available",
    "logic_validator": "available"
  }
}
```

### Docker Health Check

Docker automatically monitors API health:

```bash
# View health status
docker inspect data-analyzer-api | jq '.[0].State.Health'

# View health check logs
docker inspect data-analyzer-api | jq '.[0].State.Health.Log'
```

Health check configuration (in `Dockerfile.api`):
- **Interval**: 30 seconds
- **Timeout**: 10 seconds
- **Retries**: 3
- **Start period**: 40 seconds

---

## Troubleshooting

### Container Won't Start

```bash
# Check container logs
docker-compose -f docker-compose.api.yml logs api

# Common issues:
# 1. Missing environment variables
#    Solution: Verify .env file exists and contains required values

# 2. Port 8000 already in use
docker ps | grep 8000
#    Solution: Change port mapping in docker-compose.api.yml

# 3. Volume permission issues
#    Solution: Ensure volume is accessible
docker volume inspect data-analyzer-program-cache
```

### Health Check Failing

```bash
# Check if API is listening
docker exec data-analyzer-api curl -f http://localhost:8000/api/v1/health

# Check if Azure OpenAI is configured
docker exec data-analyzer-api env | grep AZURE_OPENAI

# View detailed logs
docker-compose -f docker-compose.api.yml logs -f api
```

### Azure OpenAI Connection Issues

```bash
# Test connection from inside container
docker exec -it data-analyzer-api python -c "
import os
print('Endpoint:', os.getenv('AZURE_OPENAI_ENDPOINT'))
print('Key exists:', bool(os.getenv('AZURE_OPENAI_API_KEY')))
"

# Check network connectivity
docker exec data-analyzer-api curl -I https://api.openai.azure.com
```

### Out of Memory

```bash
# Check container memory usage
docker stats data-analyzer-api

# Increase memory limit in docker-compose.api.yml
# deploy:
#   resources:
#     limits:
#       memory: 4G  # Increase from 2G
```

### Hot Reload Not Working (Development)

```bash
# Ensure using development override
docker-compose -f docker-compose.api.yml -f docker-compose.dev.yml up

# Check if volumes are mounted
docker inspect data-analyzer-api | jq '.[0].Mounts'

# Should see source code mounted as volumes
```

---

## Multi-Container Deployment

### Running API + Web App Together

You can run both the API server and Streamlit web app in the same environment:

```yaml
# docker-compose.full.yml
version: '3.8'

services:
  # API Server
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    image: data-analyzer-api:latest
    container_name: data-analyzer-api
    restart: unless-stopped
    ports:
      - "8000:8000"
    env_file:
      - .env
    volumes:
      - program-cache:/app/data
    networks:
      - app-network

  # Web Application
  web:
    build:
      context: .
      dockerfile: Dockerfile
    image: data-analyzer:latest
    container_name: data-analyzer-web
    restart: unless-stopped
    ports:
      - "3002:8002"
    environment:
      - APP_ENV=prod
      - API_BASE_URL=http://api:8000
    env_file:
      - .env
    depends_on:
      - api
    networks:
      - app-network

volumes:
  program-cache:

networks:
  app-network:
    driver: bridge
```

### Using an API Gateway

For production, use NGINX as a reverse proxy:

```nginx
# nginx.conf
upstream api_backend {
    server localhost:8000;
}

upstream web_backend {
    server localhost:3002;
}

server {
    listen 80;
    server_name example.com;

    # API endpoints
    location /api/ {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Web application
    location / {
        proxy_pass http://web_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## Production Considerations

### Resource Limits

Adjust resource limits based on your workload:

```yaml
# docker-compose.api.yml
deploy:
  resources:
    limits:
      cpus: '4.0'      # High concurrency
      memory: 4G       # Large dictionaries
    reservations:
      cpus: '1.0'
      memory: 1G
```

### Scaling

```bash
# Scale to multiple replicas (requires load balancer)
docker-compose -f docker-compose.api.yml up -d --scale api=3

# Or use Docker Swarm
docker stack deploy -c docker-compose.api.yml data-analyzer
```

### Monitoring

```bash
# Install Prometheus exporter for FastAPI
pip install prometheus-fastapi-instrumentator

# Add to api_server.py:
# from prometheus_fastapi_instrumentator import Instrumentator
# Instrumentator().instrument(app).expose(app)

# Scrape metrics at /metrics
curl http://localhost:8000/metrics
```

### Logging

```bash
# Forward logs to external system
# docker-compose.api.yml
logging:
  driver: "syslog"
  options:
    syslog-address: "tcp://loghost:514"
    tag: "data-analyzer-api"
```

### Security

1. **Use HTTPS in production**
   - Terminate SSL at load balancer or reverse proxy
   - Never expose port 8000 directly to internet

2. **Run as non-root user** (already configured in Dockerfile.api)

3. **Keep secrets out of environment variables**
   ```bash
   # Use Docker secrets or external secret management
   # (AWS Secrets Manager, Azure Key Vault, HashiCorp Vault)
   ```

4. **Regular security updates**
   ```bash
   # Rebuild image regularly with updated base image
   docker build --pull --no-cache -f Dockerfile.api -t data-analyzer-api:latest .
   ```

### Backup Strategy

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups/data-analyzer"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup program cache
docker run --rm \
  -v data-analyzer-program-cache:/data \
  -v $BACKUP_DIR:/backup \
  alpine tar czf /backup/cache-$DATE.tar.gz -C /data .

# Keep last 7 days
find $BACKUP_DIR -name "cache-*.tar.gz" -mtime +7 -delete
```

---

## Support

For issues and questions:
- Check logs: `docker-compose -f docker-compose.api.yml logs -f api`
- Review API documentation: http://localhost:8000/api/v1/docs
- See troubleshooting section above

## Related Documentation

- [API Documentation](../docs/API.md)
- [Main README](../README.md)
- [CLAUDE.md](../CLAUDE.md) - Development guide
