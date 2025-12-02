# NGINX Reverse Proxy Path Issue - Troubleshooting Log

**Date:** October 21, 2025
**Issue:** Streamlit app not loading through NGINX reverse proxy at `/sageapp02/`
**Status:** ✅ RESOLVED with helper NGINX container

---

## The Problem

### Main NGINX Configuration
Your main NGINX is configured with path-stripping:

```nginx
location = /sageapp02  { return 301 /sageapp02/; }
location /sageapp02/  { proxy_pass http://127.0.0.1:3002/; }
                                                    ↑ This trailing slash strips the prefix
```

**What happens:**
1. Browser requests: `https://aidemo.dcri.duke.edu/sageapp02/`
2. NGINX strips `/sageapp02/` and proxies to: `http://127.0.0.1:3002/` (root path)
3. Streamlit app expects to serve at: `/sageapp02/` (with prefix for WebSocket routing)
4. **Mismatch = WebSocket connection fails**

### Error Symptoms
- Page loads but appears frozen
- Browser console shows: `WebSocket connection to 'wss://aidemo.dcri.duke.edu/sageapp02/_stcore/stream' failed`
- WebSocket onclose errors repeating

---

## Solutions Attempted

### ❌ Solution 1: Empty BASE_URL_PATH (FAILED)
**Attempted:** Set `BASE_URL_PATH=` (empty) so app serves at root

**Why it failed:**
- Streamlit needs to know the external path to generate correct WebSocket URLs
- Even though NGINX strips the prefix internally, the browser still uses `/sageapp02/` in the URL
- WebSocket URLs were incorrect

### ❌ Solution 2: Ask IT to Fix NGINX (NOT PURSUED)
**Proposed:** Change main NGINX config from:
```nginx
location /sageapp02/  { proxy_pass http://127.0.0.1:3002/; }
```
to:
```nginx
location /sageapp02/  { proxy_pass http://127.0.0.1:3002; }  # Remove trailing slash
```

**Why not pursued:**
- Requires IT intervention
- May affect other apps
- User wanted self-contained solution

### ✅ Solution 3: Helper NGINX Container (WORKING)
**Implementation:** Added lightweight NGINX container to fix path routing

**Architecture:**
```
Browser: https://aidemo.dcri.duke.edu/sageapp02/
   ↓
Main NGINX: Strips /sageapp02/ → proxies to 127.0.0.1:3002/
   ↓
Helper NGINX (port 3002): Adds /sageapp02/ back → proxies to data-analyzer-prod:8002/sageapp02/
   ↓
Streamlit (port 8002): Serves at /sageapp02/ → Returns content + correct WebSocket URLs
```

---

## Current Working Configuration

### Files Added/Modified

**1. `nginx-helper.conf` (NEW)**
```nginx
server {
    listen 80;

    # Receive requests at root (from main NGINX after it strips /sageapp02)
    location / {
        # Add /sageapp02 prefix back before proxying to Streamlit
        rewrite ^/(.*)$ /sageapp02/$1 break;

        # Proxy to Streamlit container
        proxy_pass http://data-analyzer-prod:8002;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Streamlit-specific settings
        proxy_buffering off;
        proxy_read_timeout 86400;
    }
}
```

**2. `docker-compose.yml` (MODIFIED)**
Added `nginx-helper` service:
```yaml
services:
  data-analyzer:
    # No external port - accessed only via nginx-helper
    environment:
      - BASE_URL_PATH=/sageapp02
    networks:
      - app-network

  nginx-helper:
    image: nginx:alpine
    container_name: data-analyzer-nginx-helper
    restart: unless-stopped
    ports:
      - "127.0.0.1:3002:80"  # Exposed to main NGINX
    volumes:
      - ./nginx-helper.conf:/etc/nginx/conf.d/default.conf:ro
    depends_on:
      - data-analyzer
    networks:
      - app-network
```

**3. `entrypoint.sh` (MODIFIED)**
Added WebSocket compression disable for better reverse proxy compatibility

**4. `.streamlit/config.toml` (NEW)**
Streamlit-specific configuration for reverse proxy

**5. `Dockerfile` (MODIFIED)**
- Changed port from 8501 → 8002
- Uses entrypoint.sh instead of hardcoded CMD
- Copies .streamlit/ config directory

---

## Comparison: Port 3001 vs Port 3002

### Port 3001 (byod-synthetic-generator) - WORKING
**Docker Ports:** `0.0.0.0:3001->3001/tcp`
- Container exposed directly to all network interfaces
- App framework: Unknown (needs investigation)
- **TODO TOMORROW:** Check `/home/scb2/PROJECTS/gitRepos/` for byod project or find where it's running from

### Port 3002 (data-analyzer) - NOW WORKING
**Docker Ports:** `127.0.0.1:3002->80/tcp` (nginx-helper)
- Uses helper NGINX container
- App framework: Streamlit (requires special WebSocket handling)
- More complex setup but self-contained

### Key Differences to Investigate Tomorrow:
1. What framework is 3001 using? (FastAPI, Flask, etc.)
2. How does 3001's NGINX config differ?
3. Does 3001 have simpler WebSocket handling?
4. Can we simplify 3002 by mimicking 3001's approach?

---

## Testing & Verification

### Test Commands
```bash
# Check container status
docker compose ps

# Test health endpoint
curl -s http://127.0.0.1:3002/_stcore/health
# Should return: ok

# Test main page
curl -I http://127.0.0.1:3002/

# View logs
docker compose logs -f

# Check main NGINX config (requires sudo)
sudo nginx -T 2>/dev/null | grep -E "sageapp0[1-9]" -B 2 -A 2
```

### Browser Test
🌐 **https://aidemo.dcri.duke.edu/sageapp02/**
- Should load Streamlit interface
- No WebSocket errors in console (F12 → Console)
- App should be interactive

---

## Deployment to Other Environments

### For POC/Demo (with path prefix like /sageapp02)
**Use current setup** - Requires nginx-helper container

**docker-compose.yml:**
```yaml
environment:
  - BASE_URL_PATH=/sageapp02
# Include nginx-helper service
```

### For Dev/Val/Prod (dedicated domain, no path prefix)
**Remove nginx-helper**, expose app directly

**docker-compose.yml:**
```yaml
services:
  data-analyzer:
    ports:
      - "127.0.0.1:3002:8002"  # Direct exposure
    environment:
      - BASE_URL_PATH=  # Empty for root path
# Remove nginx-helper service entirely
```

**NGINX Configuration:**
```nginx
location / {
    proxy_pass http://127.0.0.1:3002;
    # Same proxy headers as in nginx-helper.conf
}
```

---

## Next Steps / TODO

### Tomorrow's Investigation:
1. ✅ **Find byod-synthetic-generator project**
   - Search for its docker-compose.yml or Dockerfile
   - Compare its NGINX/proxy configuration
   - Check if it uses a simpler approach

2. ✅ **Compare app frameworks**
   - byod: What framework? (FastAPI, Flask, vanilla?)
   - data-analyzer: Streamlit (WebSocket-heavy)
   - Determine if framework differences explain complexity

3. ✅ **Consider simplification**
   - Can we use byod's approach for data-analyzer?
   - Would require code changes to how app handles paths?
   - Trade-offs?

4. ✅ **Document final recommendation**
   - Which approach is better for future apps?
   - Create template docker-compose.yml for new projects

### Optional Improvements:
- Remove "version: '3.8'" from docker-compose.yml (obsolete warning)
- Add Azure Container Apps config for BASE_URL_PATH (already done in azure_config.yaml)
- Update DEPLOYMENT_VM.md with nginx-helper approach

---

## Files Modified This Session

### New Files:
- `entrypoint.sh` - Dynamic Streamlit startup script
- `.streamlit/config.toml` - Streamlit configuration
- `nginx-helper.conf` - Path-fixing reverse proxy config
- `nginx-prefix-fix.conf` - Alternative (not used)
- `TROUBLESHOOTING_NGINX_PATHS.md` - This file

### Modified Files:
- `Dockerfile` - Port 8002, entrypoint, config copy
- `docker-compose.yml` - Added nginx-helper service, networks
- `README.md` - Docker commands, BASE_URL_PATH docs
- `CLAUDE.md` - Updated port references
- `DEPLOYMENT_VM.md` - BASE_URL_PATH configuration examples
- `scripts/experimental/azure/azure_config.yaml` - Updated env vars and ports

### Git Status:
- Current branch: `dev`
- Last commits pushed to origin/dev
- Ready to merge to main when tested

---

## Quick Commands for Tomorrow

```bash
# Start the app
cd /home/scb2/PROJECTS/gitRepos/data-analyzer
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f

# Stop the app
docker compose down

# Investigate port 3001
docker inspect byod-synthetic-generator
docker port byod-synthetic-generator
# Find its project directory to compare configs
```

---

## Key Learnings

1. **NGINX trailing slash matters:**
   - `proxy_pass http://host/;` (with `/`) = strip location prefix
   - `proxy_pass http://host;` (no `/`) = keep location prefix

2. **Streamlit is picky about paths:**
   - Needs `--server.baseUrlPath` to match external URL
   - WebSocket URLs must match exactly
   - Can't just serve at root when external path differs

3. **Helper containers are valid solutions:**
   - Small NGINX Alpine image (~17MB)
   - Self-contained fix without touching main NGINX
   - Easier than coordinating with IT

4. **Environment-agnostic design:**
   - Use `BASE_URL_PATH` env var for flexibility
   - Same Docker image works in all environments
   - Just change compose file for different scenarios

---

**Status:** Working solution deployed, ready for testing tomorrow morning.
**Access:** https://aidemo.dcri.duke.edu/sageapp02/
