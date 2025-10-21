# Production Deployment on Unix VM with NGINX

This guide covers deploying the Data Analyzer to a Unix VM with Docker and NGINX reverse proxy, including auto-restart on reboot.

## Initial Setup on Unix VM

### 1. Clone Repository
```bash
cd /opt  # or your preferred location
git clone https://github.com/sagearbor/data-analyzer.git
cd data-analyzer
git checkout main  # or val for staging
```

### 2. Create Environment Configuration
```bash
# Create .env file
cat > .env << 'EOF'
# Production environment - no banner shown
APP_ENV=prod

# Azure OpenAI Configuration (if needed)
# AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
# AZURE_OPENAI_API_KEY=your-key-here
# AZURE_OPENAI_DEPLOYMENT=gpt-4
EOF

chmod 600 .env  # Protect secrets
```

### 3. Build Docker Image
```bash
docker build -t data-analyzer:latest .
```

### 4. Test Run (Optional)
```bash
# Test that it works
docker run -p 3002:8002 -e APP_ENV=prod data-analyzer:latest

# Access http://your-vm-ip:3002 to verify
# Press Ctrl+C to stop
```

---

## Option A: Docker Compose (Recommended)

### Create docker-compose.yml
```bash
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  data-analyzer:
    image: data-analyzer:latest
    container_name: data-analyzer-prod
    restart: unless-stopped
    ports:
      - "127.0.0.1:3002:8002"  # Only expose to localhost (NGINX will proxy)
    environment:
      - APP_ENV=prod
    env_file:
      - .env
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
EOF
```

### Start with Docker Compose
```bash
# Start in background
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop
docker-compose down

# Restart
docker-compose restart
```

### Auto-start on VM Reboot
Docker Compose with `restart: unless-stopped` will automatically start the container when Docker daemon starts on boot.

**Ensure Docker starts on boot:**
```bash
sudo systemctl enable docker
sudo systemctl start docker
```

---

## Option B: Systemd Service (Alternative)

If you prefer systemd instead of Docker Compose:

```bash
sudo cat > /etc/systemd/system/data-analyzer.service << 'EOF'
[Unit]
Description=Data Analyzer Streamlit Application
After=docker.service
Requires=docker.service

[Service]
Type=simple
WorkingDirectory=/opt/data-analyzer
Restart=always
RestartSec=10
ExecStartPre=-/usr/bin/docker stop data-analyzer-prod
ExecStartPre=-/usr/bin/docker rm data-analyzer-prod
ExecStart=/usr/bin/docker run --rm \
  --name data-analyzer-prod \
  -p 127.0.0.1:3002:8002 \
  -e APP_ENV=prod \
  --env-file /opt/data-analyzer/.env \
  data-analyzer:latest
ExecStop=/usr/bin/docker stop data-analyzer-prod
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable data-analyzer
sudo systemctl start data-analyzer

# Check status
sudo systemctl status data-analyzer

# View logs
sudo journalctl -u data-analyzer -f
```

---

## NGINX Reverse Proxy Configuration

### Create NGINX Site Configuration
```bash
sudo cat > /etc/nginx/sites-available/data-analyzer << 'EOF'
upstream data_analyzer {
    server 127.0.0.1:3002;
}

server {
    listen 80;
    server_name data-analyzer.yourdomain.com;  # Change this

    # Redirect to HTTPS (if using SSL)
    # return 301 https://$server_name$request_uri;

    # Or serve directly on HTTP
    location / {
        proxy_pass http://data_analyzer;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;

        # WebSocket support for Streamlit
        proxy_buffering off;
    }
}

# HTTPS Configuration (optional, recommended)
# server {
#     listen 443 ssl http2;
#     server_name data-analyzer.yourdomain.com;
#
#     ssl_certificate /etc/ssl/certs/your-cert.crt;
#     ssl_certificate_key /etc/ssl/private/your-key.key;
#
#     location / {
#         proxy_pass http://data_analyzer;
#         proxy_http_version 1.1;
#         proxy_set_header Upgrade $http_upgrade;
#         proxy_set_header Connection "upgrade";
#         proxy_set_header Host $host;
#         proxy_set_header X-Real-IP $remote_addr;
#         proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
#         proxy_set_header X-Forwarded-Proto $scheme;
#         proxy_read_timeout 86400;
#         proxy_buffering off;
#     }
# }
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/data-analyzer /etc/nginx/sites-enabled/

# Test NGINX configuration
sudo nginx -t

# Reload NGINX
sudo systemctl reload nginx
```

---

## Deployment Commands Summary

### On Initial Clone
```bash
# 1. Clone repo
git clone https://github.com/sagearbor/data-analyzer.git
cd data-analyzer
git checkout main

# 2. Create .env file
cp .env.example .env
# Edit .env to set APP_ENV=prod

# 3. Build image
docker build -t data-analyzer:latest .

# 4. Start with Docker Compose
docker-compose up -d

# 5. Configure NGINX (see above)

# 6. Enable Docker auto-start
sudo systemctl enable docker
```

### On Updates (git pull)
```bash
# Pull latest
git pull origin main

# Rebuild image
docker build -t data-analyzer:latest .

# Restart container
docker-compose down
docker-compose up -d

# Or if using systemd:
# sudo systemctl restart data-analyzer
```

---

## Auto-Restart Behavior

### With Docker Compose
- ✅ Container auto-restarts if it crashes
- ✅ Container auto-starts on VM reboot (if Docker daemon starts)
- ✅ `restart: unless-stopped` policy persists across reboots

### With Systemd
- ✅ Service auto-restarts if it crashes (`Restart=always`)
- ✅ Service auto-starts on VM reboot (`WantedBy=multi-user.target`)
- ✅ 10-second delay between restart attempts

---

## Monitoring & Logs

### Docker Compose
```bash
# View logs
docker-compose logs -f

# Check container health
docker-compose ps

# Inspect container
docker inspect data-analyzer-prod
```

### Systemd
```bash
# View logs
sudo journalctl -u data-analyzer -f

# Check status
sudo systemctl status data-analyzer

# Recent logs
sudo journalctl -u data-analyzer --since "1 hour ago"
```

### NGINX
```bash
# Access logs
sudo tail -f /var/log/nginx/access.log

# Error logs
sudo tail -f /var/log/nginx/error.log
```

---

## Troubleshooting

### Container Won't Start
```bash
# Check Docker logs
docker logs data-analyzer-prod

# Check port availability
sudo lsof -i :3002
ss -tulpn | grep 3002

# Rebuild image
docker build --no-cache -t data-analyzer:latest .
```

### NGINX 502 Bad Gateway
```bash
# Verify container is running
docker ps | grep data-analyzer

# Test upstream
curl http://127.0.0.1:3002/_stcore/health

# Check NGINX logs
sudo tail -f /var/log/nginx/error.log
```

### Updates Not Showing
```bash
# Clear Docker cache
docker system prune -a

# Rebuild
docker build --no-cache -t data-analyzer:latest .

# Force recreate
docker-compose up -d --force-recreate
```

---

## Security Notes

1. **Firewall**: Only expose port 80/443 (NGINX), not 3002
2. **SSL/TLS**: Use Let's Encrypt for production HTTPS
3. **Environment**: Keep `.env` file secure (chmod 600)
4. **Updates**: Regularly update Docker images and base OS
5. **Logs**: Rotate logs to prevent disk filling

---

## Quick Reference

**Start Everything:**
```bash
sudo systemctl start docker
docker-compose up -d
sudo systemctl reload nginx
```

**Stop Everything:**
```bash
docker-compose down
```

**View All Logs:**
```bash
docker-compose logs -f
```

**Update Deployment:**
```bash
git pull origin main
docker build -t data-analyzer:latest .
docker-compose up -d --force-recreate
```
