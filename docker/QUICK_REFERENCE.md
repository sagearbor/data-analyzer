# Docker API - Quick Reference Card

## Essential Commands

### Build & Run

```bash
# Build image
docker build -f Dockerfile.api -t data-analyzer-api:latest .

# Run production
docker run -d --name data-analyzer-api --env-file .env -p 8000:8000 data-analyzer-api:latest

# Run with docker-compose
docker-compose -f docker-compose.api.yml up -d

# Run development mode (hot reload)
docker-compose -f docker-compose.api.yml -f docker-compose.dev.yml up
```

### Manage Containers

```bash
# List running containers
docker ps

# Stop container
docker stop data-analyzer-api

# Start container
docker start data-analyzer-api

# Restart container
docker restart data-analyzer-api

# Remove container
docker rm data-analyzer-api

# Remove container (force)
docker rm -f data-analyzer-api
```

### Logs & Debugging

```bash
# View logs
docker logs data-analyzer-api

# Follow logs
docker logs -f data-analyzer-api

# Last 50 lines
docker logs --tail 50 data-analyzer-api

# Execute command in container
docker exec data-analyzer-api ls -la /app

# Interactive shell
docker exec -it data-analyzer-api bash
```

### Health & Status

```bash
# Check health status
docker ps --filter name=data-analyzer-api

# Health endpoint
curl http://localhost:8000/api/v1/health

# Inspect container
docker inspect data-analyzer-api

# Container stats
docker stats data-analyzer-api
```

### Volume Management

```bash
# List volumes
docker volume ls | grep data-analyzer

# Inspect volume
docker volume inspect data-analyzer-program-cache

# Backup volume
docker run --rm \
  -v data-analyzer-program-cache:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/cache-backup.tar.gz -C /data .

# Restore volume
docker run --rm \
  -v data-analyzer-program-cache:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/cache-backup.tar.gz -C /data

# Remove volume (WARNING: deletes data)
docker volume rm data-analyzer-program-cache
```

### Docker Compose

```bash
# Start services
docker-compose -f docker-compose.api.yml up -d

# Stop services
docker-compose -f docker-compose.api.yml down

# View logs
docker-compose -f docker-compose.api.yml logs -f

# Rebuild and restart
docker-compose -f docker-compose.api.yml up -d --build

# Stop and remove volumes
docker-compose -f docker-compose.api.yml down -v
```

## Environment Variables

Required in `.env` file:

```bash
# Azure OpenAI (required)
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-08-01-preview

# API Security (required)
DATA_ANALYZER_API_KEY=your-api-key
DATA_ANALYZER_ADMIN_PASSWORD=your-admin-password

# Optional
APP_ENV=prod  # dev, staging, prod
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

## API Endpoints

Base URL: `http://localhost:8000`

```bash
# Health check
curl http://localhost:8000/api/v1/health

# API documentation
open http://localhost:8000/api/v1/docs

# Test with API key
curl -H "X-API-Key: your-key" http://localhost:8000/api/v1/health
```

## Common Issues

### Port Already in Use
```bash
# Find what's using port 8000
lsof -i :8000
netstat -tlnp | grep 8000

# Use different port
docker run -p 9000:8000 data-analyzer-api:latest
```

### Container Exits Immediately
```bash
# Check logs
docker logs data-analyzer-api

# Run interactively to debug
docker run -it --rm data-analyzer-api:latest bash
```

### Permission Denied
```bash
# Check volume permissions
docker exec data-analyzer-api ls -la /app/data/

# Should be owned by apiuser (UID 1000)
```

### SSL Certificate Errors (WSL/Corporate)
Already fixed in Dockerfile with `--trusted-host` flags.

### Health Check Failing
```bash
# Test from inside container
docker exec data-analyzer-api curl http://localhost:8000/api/v1/health

# Check environment variables
docker exec data-analyzer-api env | grep AZURE_OPENAI
```

## Files & Locations

```
/home/scb2/PROJECTS/gitRepos-wsl/data-analyzer/
├── Dockerfile.api                    # Production Dockerfile
├── docker-compose.api.yml            # Production compose
├── docker-compose.dev.yml            # Development override
├── .dockerignore                     # Build exclusions
├── .env                              # Environment variables (not in git)
└── docker/
    ├── README.md                     # Full documentation
    ├── DOCKER_API_SUMMARY.md         # Implementation summary
    ├── QUICK_REFERENCE.md            # This file
    └── test-docker-api.sh            # Automated test script
```

## Testing

```bash
# Run automated tests
./docker/test-docker-api.sh

# Manual test
docker build -f Dockerfile.api -t data-analyzer-api:latest .
docker run -d --name test-api --env-file .env -p 9000:8000 data-analyzer-api:latest
sleep 10
curl http://localhost:9000/api/v1/health
docker rm -f test-api
```

## Image Info

- **Name**: `data-analyzer-api:latest`
- **Size**: ~761MB
- **Base**: `python:3.11-slim`
- **User**: `apiuser` (UID 1000, non-root)
- **Port**: 8000
- **Volume**: `/app/data` (program cache)
- **Health**: `/api/v1/health` (30s interval)

## Security

- ✓ Non-root user (apiuser)
- ✓ Minimal base image
- ✓ No secrets in image
- ✓ Health checks enabled
- ✓ Resource limits configured
- ✓ SSL compatibility

## Performance

- **CPU**: 2 cores max (prod), 1 core (dev)
- **Memory**: 2GB max (prod), 1GB (dev)
- **Workers**: 1 (increase for production load)
- **Startup**: ~40s before healthy

## More Information

See `docker/README.md` for:
- Detailed troubleshooting
- Multi-container deployment
- Production considerations
- Monitoring and scaling
- Backup strategies
