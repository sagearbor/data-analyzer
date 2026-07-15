# Setup Guide for Other Repos

Copy this configuration to enable reverse proxy subdirectory deployments and environment warning banners.

---

## ⚡ QUICK START

### Files to Copy:
1. **`src/web/static/js/env-banner.js`** → your repo's `static/js/` folder
2. This entire "Detailed Setup" section → your repo's `CLAUDE.md`

### Code Changes (3 lines):
```python
# 1. main.py (or app.py)
app = FastAPI(root_path=os.getenv("ROOT_PATH", ""))

# 2. docker-compose.yml under environment:
- ROOT_PATH=${ROOT_PATH:-}

# 3. All HTML files in <head>:
<script src="/static/js/env-banner.js"></script>
```

### Deploy:
```bash
# Most common: Comment out ROOT_PATH in .env (NGINX strips path)
docker compose down && docker compose build && docker compose up -d
```

**Test:** Visit `http://localhost:PORT/` → should see red environment banner

---

## 📋 DETAILED SETUP

### 1. Reverse Proxy Subdirectory Support

**Problem:** App deployed at `/sageapp01/` but static files fail with 404 errors.

**Solution:** Configure `ROOT_PATH` based on your NGINX config.

#### A. Check Your NGINX Configuration

**NGINX strips path (MOST COMMON):**
```nginx
location /sageapp01/ {
    proxy_pass http://localhost:3001/;  # ← Trailing slash = strips /sageapp01/
}
```
→ **Leave `ROOT_PATH` commented in `.env`**

**NGINX keeps path (LESS COMMON):**
```nginx
location /sageapp01/ {
    proxy_pass http://localhost:3001;  # ← No trailing slash = keeps /sageapp01/
}
```
→ **Set `ROOT_PATH=/sageapp01` in `.env`**

#### B. Update Application Code

**FastAPI/Flask:**
```python
import os
from fastapi import FastAPI

app = FastAPI(
    title="Your App",
    root_path=os.getenv("ROOT_PATH", "")  # Supports subdirectory deployments
)
```

**Docker Compose:**
```yaml
services:
  your-app:
    environment:
      - ROOT_PATH=${ROOT_PATH:-}
    restart: unless-stopped  # Auto-restart on failure/reboot
```

**HTML Templates:**
```html
<!-- Add to <head> section -->
<script>
    window.BASE_PATH = (() => {
        const path = window.location.pathname;
        const match = path.match(/^(\/[^\/]+)/);
        return match ? match[1] : '';
    })();
</script>

<!-- Use in static file references -->
<link rel="stylesheet" href="/static/css/style.css">
<!-- becomes (if using JavaScript injection): -->
<script>
    document.write('<link rel="stylesheet" href="' + window.BASE_PATH + '/static/css/style.css">');
</script>
```

**JavaScript API Calls:**
```javascript
const API_BASE = window.location.origin + window.BASE_PATH;
fetch(`${API_BASE}/api/endpoint`, {...});
```

#### C. Deploy
```bash
# Edit .env based on NGINX config above
# vim .env

# Rebuild and restart
docker compose down
docker compose build
docker compose up -d
```

---

### 2. Environment Warning Banner

**Purpose:** Prevent accidental use of dev/test environments with real/PHI data.

**Features:**
- Auto-detects environment from URL (`/dev/`, `/val/`, `localhost`)
- Red banner for dev/test/local/unknown (high alert)
- Yellow banner for val/staging (caution)
- No banner for production

#### Implementation

**Step 1: Copy banner script**
```bash
# From BYOD repo to your repo
cp src/web/static/js/env-banner.js /path/to/your-repo/static/js/
```

**Step 2: Add to all HTML pages**
```html
<!-- Add to <head> section, BEFORE other scripts -->
<script src="/static/js/env-banner.js"></script>
```

**Step 3: Test**
```bash
docker compose up -d

# Visit app - should see banner based on environment:
# http://localhost:3001/ → Red "Local development" banner
# https://domain.com/dev/ → Red "Development environment" banner
# https://domain.com/val/ → Yellow "Validation environment" banner
# https://domain.com (prod) → No banner
```

#### Manual Override (if needed)
```html
<script src="/static/js/env-banner.js" data-env="dev"></script>
```

#### Banner Colors by Environment

| Environment | Color | Security Level |
|-------------|-------|----------------|
| local, dev, test, unknown | 🚨 Red | Never use with real data |
| val, staging | ⚠️ Yellow | Controlled testing only |
| prod | ✅ None | Production |

---

### 3. Auto-Restart on VM Reboot

**Status:** ✅ Already configured in this repo!

#### Current Setup

**docker-compose.yml** (line 57):
```yaml
restart: unless-stopped
```

**Docker service** (enabled on boot):
```bash
# Verify Docker starts on boot
systemctl is-enabled docker
# → should output: enabled
```

#### How It Works

1. **VM boots** → Docker service starts automatically
2. **Docker starts** → Reads all containers with `restart: unless-stopped`
3. **Containers start** → Your app is live

**Important:** You must start containers ONCE manually after initial setup:
```bash
docker compose up -d
```

After that, they auto-restart forever (unless you explicitly stop them).

#### Verify Auto-Restart

**Test restart:**
```bash
# Stop containers
docker compose stop

# Start Docker service (simulates reboot)
sudo systemctl restart docker

# Wait 10 seconds, then check
docker ps
# → Should see your containers running
```

**Check logs after reboot:**
```bash
# Container logs
docker compose logs -f --tail=50

# System logs
journalctl -u docker -n 50
```

#### Optional: Dedicated Systemd Service

For better control and logging, create a systemd service:

**Create service file:**
```bash
sudo vim /etc/systemd/system/byod-app.service
```

**Add content:**
```ini
[Unit]
Description=BYOD Synthetic Data Generator
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/dcri/sasusers/home/scb2/azDevOps/Create-mockData-from-real-file
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
User=scb2

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable byod-app.service
sudo systemctl start byod-app.service

# Check status
sudo systemctl status byod-app.service
```

**Verify after reboot:**
```bash
# After VM reboot
sudo systemctl status byod-app.service
docker ps
```

---

## 🔍 Troubleshooting

### Reverse Proxy Issues

**Problem: 404 errors for CSS/JS files**
```bash
# Check what path Docker receives
docker logs byod-synthetic-generator | grep "GET /static"

# If shows "/static/..." → NGINX strips path (comment ROOT_PATH)
# If shows "/sageapp01/static/..." → NGINX keeps path (set ROOT_PATH)
```

**Problem: All pages return 404**
```bash
# Check NGINX logs
sudo tail -f /var/log/nginx/error.log

# Check if Docker is running
docker ps | grep byod
```

### Environment Banner Issues

**Problem: Banner not showing**
```bash
# Check browser console (F12) for errors
# Verify script loaded:
curl http://localhost:3001/static/js/env-banner.js

# Should return JavaScript code, not 404
```

**Problem: Wrong banner color**
- Check URL path (banner detects from URL)
- Use manual override: `data-env="dev"`
- Edit detection logic in `env-banner.js` if needed

### Auto-Restart Issues

**Problem: Containers don't start after reboot**
```bash
# 1. Check Docker is running
sudo systemctl status docker

# 2. Check restart policy
docker inspect byod-synthetic-generator | grep -i restart

# Should show: "RestartPolicy": {"Name": "unless-stopped"}

# 3. Manually start
docker compose up -d

# 4. Check logs
docker compose logs
```

**Problem: Container starts but crashes**
```bash
# Check container logs
docker compose logs -f

# Check health check
docker inspect byod-synthetic-generator | grep -A 10 Health
```

---

## ✅ Checklist for Each New Repo

### Reverse Proxy:
- [ ] Add `root_path=os.getenv("ROOT_PATH", "")` to app initialization
- [ ] Add `ROOT_PATH=${ROOT_PATH:-}` to docker-compose.yml
- [ ] Add `BASE_PATH` auto-detection to HTML
- [ ] Update static file refs to use `BASE_PATH`
- [ ] Configure `.env` based on NGINX (most: comment `ROOT_PATH`)
- [ ] Test: `docker compose up -d` and visit subdirectory URL

### Environment Banner:
- [ ] Copy `env-banner.js` to `static/js/`
- [ ] Add `<script src="/static/js/env-banner.js"></script>` to all HTML
- [ ] Test locally → should see red banner
- [ ] Test in each environment (dev, val, prod)

### Auto-Restart:
- [ ] Add `restart: unless-stopped` to docker-compose.yml
- [ ] Verify Docker enabled: `systemctl is-enabled docker`
- [ ] Start once: `docker compose up -d`
- [ ] Test reboot (if safe): `sudo reboot`
- [ ] Verify running after reboot: `docker ps`

---

## 📁 What to Copy to Other Repos

**This file contains everything you need!**
- **This guide** - Instructions and checklist
- **Complete env-banner.js code** - See Appendix below

**Just copy this one file (`SETUP_OTHER_REPOS.md`) to your new repo and you have everything.**

**Reference examples in BYOD repo:**
- `src/web/index.html` (lines 16-48) - BASE_PATH detection
- `main.py` - FastAPI root_path configuration
- `docker-compose.yml` - ROOT_PATH + restart policy

---

## 📎 APPENDIX: Complete env-banner.js Code

**Create this file:** `static/js/env-banner.js`

Copy everything below (all 238 lines) into your new file:

```javascript
/**
 * Environment Warning Banner
 *
 * Auto-detects deployment environment and displays warning banner for non-production environments.
 * Portable, self-contained, no dependencies.
 *
 * Detection Strategy (in priority order):
 * 1. URL path prefix (/dev/, /val/, /sageapp01/, etc.)
 * 2. Hostname patterns (localhost, *-dev.*, *-val.*, etc.)
 * 3. Manual override via data-env attribute on script tag
 * 4. Safe default: Show warning if environment cannot be determined
 *
 * Usage:
 *   <script src="/static/js/env-banner.js"></script>
 *   or with manual override:
 *   <script src="/static/js/env-banner.js" data-env="dev"></script>
 */

(function() {
    'use strict';

    /**
     * Detect environment from URL and hostname
     */
    function detectEnvironment() {
        const hostname = window.location.hostname.toLowerCase();
        const pathname = window.location.pathname.toLowerCase();

        // Check for manual override via script tag data attribute
        const scriptTag = document.currentScript;
        if (scriptTag && scriptTag.dataset.env) {
            return scriptTag.dataset.env.toLowerCase();
        }

        // Detection from localhost
        if (hostname === 'localhost' || hostname === '127.0.0.1' || hostname.startsWith('192.168.') || hostname.startsWith('10.')) {
            return 'local';
        }

        // Detection from URL path (common reverse proxy pattern)
        // Matches: /dev/, /val/, /prod/, /sageapp01/, /sageapp02/, etc.
        const pathMatch = pathname.match(/^\/([^\/]+)\//);
        if (pathMatch) {
            const pathPrefix = pathMatch[1];

            // Check if it's a known environment prefix
            if (pathPrefix === 'dev' || pathPrefix.includes('dev')) {
                return 'dev';
            }
            if (pathPrefix === 'val' || pathPrefix.includes('val') || pathPrefix.includes('validation')) {
                return 'val';
            }
            if (pathPrefix === 'prod' || pathPrefix === 'production') {
                return 'prod';
            }
            if (pathPrefix === 'test' || pathPrefix.includes('test')) {
                return 'test';
            }
            if (pathPrefix === 'staging' || pathPrefix.includes('staging')) {
                return 'staging';
            }

            // If it matches a pattern like /sageappXX/ assume it's a test/dev deployment
            if (/^sageapp\d+$/.test(pathPrefix)) {
                return 'test';
            }
        }

        // Detection from hostname patterns
        if (hostname.includes('-dev.') || hostname.includes('.dev.') || hostname.startsWith('dev-') || hostname.startsWith('dev.')) {
            return 'dev';
        }
        if (hostname.includes('-val.') || hostname.includes('.val.') || hostname.startsWith('val-') || hostname.startsWith('val.')) {
            return 'val';
        }
        if (hostname.includes('-test.') || hostname.includes('.test.') || hostname.startsWith('test-') || hostname.startsWith('test.')) {
            return 'test';
        }
        if (hostname.includes('-staging.') || hostname.includes('.staging.') || hostname.startsWith('staging-') || hostname.startsWith('staging.')) {
            return 'staging';
        }
        if (hostname.includes('-prod.') || hostname.includes('.prod.') || hostname.startsWith('prod-') || hostname.startsWith('prod.')) {
            return 'prod';
        }

        // If running on a well-known production domain (customize as needed)
        // if (hostname === 'yourdomain.com' || hostname === 'www.yourdomain.com') {
        //     return 'prod';
        // }

        // Default: unknown environment (safest - show warning)
        return 'unknown';
    }

    /**
     * Get banner configuration for each environment
     */
    function getBannerConfig(env) {
        const configs = {
            local: {
                show: true,
                bgColor: '#FF3B30',
                textColor: '#FFFFFF',
                message: 'Local development - do NOT use with real/PHI DATA - TESTING ONLY',
                icon: '🚨'
            },
            dev: {
                show: true,
                bgColor: '#FF3B30',
                textColor: '#FFFFFF',
                message: 'Development environment - do NOT use with real/PHI DATA - TESTING ONLY',
                icon: '🚨'
            },
            test: {
                show: true,
                bgColor: '#FF3B30',
                textColor: '#FFFFFF',
                message: 'Test environment - do NOT use with real/PHI DATA - TESTING ONLY',
                icon: '🚨'
            },
            val: {
                show: true,
                bgColor: '#FFCC00',
                textColor: '#000000',
                message: 'Validation environment - do NOT use with real/PHI DATA - TESTING ONLY',
                icon: '⚠️'
            },
            staging: {
                show: true,
                bgColor: '#FFCC00',
                textColor: '#000000',
                message: 'Staging environment - do NOT use with real/PHI DATA - TESTING ONLY',
                icon: '⚠️'
            },
            unknown: {
                show: true,
                bgColor: '#FF3B30',
                textColor: '#FFFFFF',
                message: 'Environment unknown - do NOT use with real/PHI DATA - TESTING ONLY',
                icon: '🚨'
            },
            prod: {
                show: false,  // No banner for production
                bgColor: '',
                textColor: '',
                message: '',
                icon: ''
            }
        };

        return configs[env] || configs.unknown;
    }

    /**
     * Create and inject banner into page
     */
    function createBanner(config) {
        if (!config.show) {
            return;  // No banner for production
        }

        // Create banner element
        const banner = document.createElement('div');
        banner.id = 'env-warning-banner';
        banner.setAttribute('role', 'alert');
        banner.setAttribute('aria-live', 'polite');

        // Set styles
        banner.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background-color: ${config.bgColor};
            color: ${config.textColor};
            padding: 8px 16px;
            text-align: center;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            font-size: 14px;
            font-weight: 600;
            z-index: 999999;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            line-height: 1.4;
        `;

        // Set content
        banner.innerHTML = `
            <span style="margin-right: 8px;">${config.icon}</span>
            <span>${config.message}</span>
        `;

        // Inject into page
        // Wait for DOM to be ready
        if (document.body) {
            document.body.insertBefore(banner, document.body.firstChild);
            adjustPageForBanner(banner);
        } else {
            document.addEventListener('DOMContentLoaded', function() {
                document.body.insertBefore(banner, document.body.firstChild);
                adjustPageForBanner(banner);
            });
        }
    }

    /**
     * Adjust page layout to account for fixed banner
     */
    function adjustPageForBanner(banner) {
        // Add padding to body to prevent content from being hidden under banner
        const bannerHeight = banner.offsetHeight;
        document.body.style.paddingTop = bannerHeight + 'px';

        // Re-adjust if window is resized (in case banner height changes)
        window.addEventListener('resize', function() {
            document.body.style.paddingTop = banner.offsetHeight + 'px';
        });
    }

    /**
     * Main execution
     */
    function init() {
        const env = detectEnvironment();
        const config = getBannerConfig(env);

        // Optional: Log to console for debugging
        console.log('[env-banner] Detected environment:', env);

        createBanner(config);

        // Expose environment to window for other scripts to use if needed
        window.APP_ENV = env;
    }

    // Execute immediately
    init();

})();
```

---

**Last Updated:** 2025-10-15
**Source:** Create-mockData-from-real-file (BYOD Synthetic Data Generator)
**One File = Everything You Need** ✅

