# Data Analyzer — Deployment & Security Brief for IT Review

**Prepared:** 2026-07-09 · **Branch:** `feature/enhancements` · **Target:** Azure Container Apps (FastAPI API image)

This document is a self-audit ahead of the deployment review. It lists what was
found, what has been fixed, and what remains open — deliberately including the
uncomfortable items, because you'll find them anyway and it's cheaper to lead with them.

---

## 1. TL;DR

- A public-repo secret leak was found and neutralized (dead key, file removed, gitignore hardened). **History scrub still owed** — see 3.1.
- The API server had a startup-blocking bug and several real security gaps (timing-unsafe auth, CORS wildcard+credentials, root container). **All fixed and verified today.**
- Both Docker images build clean; the FastAPI image runs non-root. Full test suite: **186+ passing** after test-fixture repairs.
- Three items need a decision/verification *before* go-live, not blockers for the conversation: history scrub, the `exec()`-based rule validator, and rate-limiting behind ACA ingress.

---

## 2. What was fixed (done, in-tree, committed on `feature/enhancements`)

| # | Area | Issue | Fix |
|---|------|-------|-----|
| 1 | API startup | `api_server.py` couldn't boot — one `try/except ImportError` wrapped both optional and required model imports, leaving Pydantic models undefined (`NameError` at route decoration). **The API could not have started as committed.** | Split imports so `src.api_models` loads unconditionally; each optional service degrades to 503 independently. |
| 2 | API auth | Credential check used `!=` (timing side-channel, byte-by-byte brute-forceable) | `secrets.compare_digest` for API key and admin password |
| 3 | API auth | Auth silently disabled if key env var unset — a prod footgun | Fail-closed: server refuses to start when `APP_ENV=prod` and `DATA_ANALYZER_API_KEY` is unset |
| 4 | API CORS | `allow_origins=["*"]` + `allow_credentials=True` → any site could make credentialed cross-origin calls | `ALLOWED_ORIGINS` env var; deny-all cross-origin by default in prod |
| 5 | API headers | No security response headers | Middleware adds `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Cache-Control: no-store` |
| 6 | API runtime | uvicorn hot-reload defaulted on (spawns file-watcher in the container) | `API_RELOAD` defaults to `false` |
| 7 | Secrets | `.env.bak` (from an unrelated project) committed to a **public** repo, containing a real Azure OpenAI key | File removed from index & disk; key **verified dead (HTTP 401)**; `.gitignore`/`.dockerignore` now block `.env.*` |
| 8 | Container | Streamlit `Dockerfile` ran as **root** with build tools in the final image | Non-root `appuser` (UID 1000); **built & run-verified** (`whoami`→`appuser`, health 200) |
| 9 | Web app | Dictionary disk cache used `pickle` (arbitrary code exec on crafted cache file) | Switched to JSON |
| 10 | Data load | No upload size cap (memory-exhaustion / zip-bomb surface) | `DataLoader` size cap, `DATA_ANALYZER_MAX_UPLOAD_BYTES` (100 MB default) |

The FastAPI image (`Dockerfile.api`) was already best-practice: multi-stage,
non-root, minimal runtime, HTTP healthcheck.

---

## 3. Open items — need a decision or verification before go-live

### 3.1 Purge the leaked secret from git history *(owner: repo admin)*
The key is dead, but `.env.bak` still exists in historical commits on the public
remote. Removing it from HEAD does not remove it from history.
**Action:** run `git filter-repo` (or BFG), force-push, have collaborators
re-clone. Also confirm GitHub push-protection/secret-scanning is on and add a
pre-commit secret scanner (gitleaks/detect-secrets). *Not done here because it
rewrites shared history and needs coordination.*

### 3.2 `exec()`-based logic validator *(owner: eng)*
`src/logic_engine.py` executes LLM-derived validation rules via `exec()` inside a
denylist sandbox. The dictionary content feeding it is user/LLM-influenced, and
the denylist has known bypass gaps (unsanitized f-string interpolation of field
names; `pd` module exposed, which itself wraps `pd.eval`/`read_pickle`).
**Recommendation:** replace generated-Python-then-exec with a restricted-grammar
interpreter, or run it in an isolated subprocess with CPU/memory/time limits as a
compensating control. This is the item a skeptical reviewer is most likely to
attack live — own it, name the gaps, lead with the remediation plan.

### 3.3 Rate limiting behind ACA ingress *(owner: eng, verify post-deploy)*
The limiter keys on `request.client.host` and does **not** trust
`X-Forwarded-For` (good — not spoofable). But behind ACA ingress that host may
resolve to the ingress IP, so all clients could share one bucket.
**Action:** after first deploy, confirm what IP the app sees; if it's the proxy,
key the limiter off `X-API-Key` for authenticated routes or configure ACA to
preserve client IP.

### 3.4 Lower-priority hardening
- Pin dependency versions with hashes (`pip-compile --generate-hashes`); currently floating `>=`.
- Replace `--trusted-host` pip flags with a corporate CA bundle bake-in.
- Excel/Parquet decompression cap (zip-bomb) beyond the raw byte check.
- Formula-injection escaping on Excel report export (`=+-@` prefixes).
- Standardize error responses to not echo library internals to key-holders (gate on `DEBUG`).
- Set explicit `server.maxUploadSize` in `.streamlit/config.toml` + proxy body cap.

---

## 4. Deployment artifact clarification

The `docker-compose.*.yml` files are the **on-prem/VM** pattern (Streamlit +
NGINX helper sidecar). They are **not** the ACA deployment. The ACA-native config
is `deploy/aca/containerapp.yaml` (single external HTTPS ingress on port 8000,
secrets from the ACA secret store, HTTP health probes, 1–3 replica autoscale).
See `deploy/aca/README.md`. Make sure the room is clear on which artifact is
"the deployment" — don't let the compose file get conflated with ACA.

---

## 5. Likely questions & straight answers

- **"Is the repo public? Was a real key exposed?"** Yes and yes. It's confirmed
  dead (401), removed from the working tree, and a history-scrub + push-protection
  plan is queued (3.1). Owning it plainly is the point.
- **"Does the web container run as root?"** No longer — fixed and run-verified today.
- **"Why does your app run AI-generated code?"** The rule validator execs
  LLM-derived logic; we've named the sandbox's gaps ourselves and have a
  re-architecture plan (3.2).
- **"What stops a huge upload from OOMing the container?"** A `DataLoader` size
  cap; ACA per-replica memory limit; proxy body cap is the next step.
- **"Timing attacks / CORS / rate limiting?"** Constant-time auth ✓, deny-by-default
  CORS ✓, rate limiting present with one post-deploy verification owed (3.3).
- **"Where does dictionary content go?"** To Azure OpenAI via `src/llm_client.py`.
  Confirm the Azure OpenAI resource's data-retention/BAA terms before go-live —
  relevant since dictionaries can describe research/clinical data.
