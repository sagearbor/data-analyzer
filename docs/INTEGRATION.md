# Integration Guide — Calling the Data Analyzer Service

The Data Analyzer exposes the **same core analysis logic** through three independent interfaces. Choose the interface that fits your client:

| Interface | When to use |
|-----------|-------------|
| **REST API** | Any HTTP client; easiest for ad-hoc scripting and CI pipelines |
| **MCP** | Claude Desktop, Cursor, or any Model Context Protocol host |
| **A2A** | Other DCRI agent-to-agent pipelines (`dcri-ct-graph` and siblings) |

All three transports delegate to `analysis_service.py`. An error in data parsing returns `{"error": "...", "type": "..."}` in the JSON body rather than an HTTP 5xx, so clients can inspect the payload uniformly.

---

## 1. REST API

### Running the API server

```bash
# Default port 8003
python api_server.py

# Custom port
API_PORT=9000 python api_server.py
```

The server listens on `0.0.0.0` so it is reachable from Docker networks and remote clients.

### Endpoints

#### `GET /health`

Liveness probe. Also reports which A2A runtime is active (real `dcri-a2a-core` or the bundled shim).

```bash
curl http://localhost:8003/health
```

```json
{
  "status": "ok",
  "service": "data-analyzer",
  "a2a_backend": "shim",
  "a2a_shim": true
}
```

#### `POST /analyze`

Run the full data-quality pipeline (type validation, range checks, missing values, duplicate detection).

```bash
curl -X POST http://localhost:8003/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "data_content": "name,age,score\nAlice,30,95.5\nBob,25,88.0",
    "file_format": "csv",
    "schema": {"age": "int", "score": "float"},
    "rules": {"age": {"min": 0, "max": 120}},
    "min_rows": 1
  }'
```

**Body fields:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data_content` | string | required | Raw CSV/JSON text, base64-encoded bytes, or `data:<mime>;base64,...` data-URL |
| `file_format` | string | `"csv"` | `"csv"`, `"json"`, `"excel"`, `"parquet"` |
| `schema` | object | `{}` | Column-type map: `{"col": "int\|float\|str\|bool\|datetime"}` |
| `rules` | object | `{}` | Validation rules: `{"col": {"min": 0, "max": 100, "allowed_values": [...]}}` |
| `min_rows` | int | `1` | Minimum expected row count |
| `encoding` | string | `"utf-8"` | Text encoding for decoding base64 content |

#### `POST /data-info`

Lightweight shape/column/sample info — no full quality pipeline, fast.

```bash
curl -X POST http://localhost:8003/data-info \
  -H "Content-Type: application/json" \
  -d '{"data_content": "name,age\nAlice,30\nBob,25", "sample_rows": 2}'
```

**Response shape:**
```json
{
  "format": "csv",
  "shape": {"rows": 2, "columns": 2},
  "columns": ["name", "age"],
  "dtypes": {"name": "object", "age": "int64"},
  "sample_data": [{"name": "Alice", "age": 30}],
  "missing_values": {"name": 0, "age": 0},
  "duplicate_rows": 0
}
```

---

## 2. MCP (Model Context Protocol)

The MCP server (`mcp_server.py`) exposes two tools: `analyze_data` and `get_data_info`. Any MCP host (Claude Desktop, Cursor, etc.) can call them.

### Running the MCP server

```bash
python mcp_server.py
```

The server communicates over stdio following the MCP protocol. Configure your MCP host to launch this command.

### Tool: `analyze_data`

Accepts the same fields as `POST /analyze`. The MCP tool schema lists them as required/optional properties.

### Tool: `get_data_info`

Accepts `data_content`, `file_format`, and `sample_rows`. Returns the same shape as `POST /data-info`.

Both tools delegate internally to `analysis_service.py`, so the output is byte-for-byte identical to the REST responses.

---

## 3. A2A (Agent-to-Agent)

A2A is the inter-agent wire protocol used by DCRI pipeline systems such as `dcri-ct-graph`.

### Agent identity

```
agent_id : data-analyzer
version  : 1.0.0
```

### AgentCard discovery

Every A2A-compliant agent exposes its capabilities at a well-known URL:

```
GET http://<host>:8003/.well-known/agent.json
```

Example response:
```json
{
  "name": "Data Analyzer",
  "description": "Runs data-quality checks ...",
  "version": "1.0.0",
  "url": "/a2a/data-analyzer",
  "capabilities": {"streaming": false, "pushNotifications": false},
  "methods": ["message/send", "tasks/get"],
  "x-dcri": {"agent_id": "data-analyzer", "internal_only": false}
}
```

Other agents (e.g. `dcri-ct-graph` pipeline nodes) point their `a2a_endpoint` configuration at:

```
https://<host>/a2a/data-analyzer
```

### Sending an A2A message

Encode your payload in the canonical A2A envelope using `encode_message`:

```python
from a2a_runtime import encode_message, decode_message, A2AClient

# In-process short-circuit (agent registered locally):
import a2a_agent
a2a_agent.register()

client = A2AClient()
result = await client.call(
    "data-analyzer",
    {
        "operation": "analyze",          # or "get_info"
        "data_content": "name,age\nAlice,30",
        "file_format": "csv",
    },
    correlation_id="my-request-id",
    cycle=1,
)
```

Over HTTP (remote agent):

```bash
# Build the envelope manually or use the SDK:
curl -X POST http://<host>:8003/a2a/data-analyzer \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "operation": "analyze",
      "data_content": "name,age\nAlice,30",
      "file_format": "csv"
    },
    "metadata": {
      "correlation_id": "abc-123",
      "cycle": 1,
      "agent_id": "data-analyzer"
    }
  }'
```

**Supported `operation` values:**

| Value | Description |
|-------|-------------|
| `"analyze"` (default) | Full quality pipeline |
| `"get_info"` or `"data_info"` | Lightweight shape/column info |

The response is itself an A2A envelope. Decode with `decode_message(response_json)` to extract the result dict.

---

## A2A runtime — shim vs real package

`a2a_runtime/` is a thin indirection layer:

- If `dcri-a2a-core` is installed (`pip install dcri-a2a-core`), the **real versioned wire contract** is used.
- Otherwise, the **bundled pure-stdlib shim** (`a2a_runtime/_shim.py`) provides identical names and call signatures.

The `GET /health` response and AgentCard `x-dcri` section both expose `a2a_backend` and `a2a_shim` so operators can see at a glance which runtime is active.

To upgrade to the real package:
```bash
pip install dcri-a2a-core
# No code changes required — a2a_runtime/__init__.py prefers the real package automatically.
```

---

## Running the API server

```bash
# Default port 8003
python api_server.py

# Override port
API_PORT=9000 python api_server.py
```

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `API_PORT` | `8003` | TCP port for the REST + A2A HTTP server |

---

## Authentication (production note)

The API server does not implement authentication itself — that is a **deployment concern**. In production, place the service behind your org's API gateway or reverse proxy (NGINX, Azure API Management, etc.) and enforce bearer-token validation there. The A2A envelope metadata section is a natural place to carry a token when the real `dcri-a2a-core` package adds its HTTP transport layer.

---

## Deploying to Google Cloud Run (public, unauthenticated)

The repository ships with a ready-to-use Cloud Run deployment path that builds a minimal API image (no Streamlit, no LLM packages) and deploys it as a fully public HTTPS endpoint.

### One-command deploy

```bash
PROJECT_ID=my-gcp-project ./deploy/cloudrun.sh
```

Optional overrides (set as environment variables before the command):

| Variable | Default | Description |
|----------|---------|-------------|
| `REGION` | `us-central1` | GCP region |
| `SERVICE` | `data-analyzer-api` | Cloud Run service name |
| `IMAGE` | derived | Full Artifact Registry image path |

The script will:
1. Ensure the `apps` Artifact Registry repository exists (idempotent — safe to re-run).
2. Submit a Cloud Build job (`cloudbuild.yaml`) to build `Dockerfile.api` and push the image.
3. Deploy the image to Cloud Run with `--allow-unauthenticated --port 8080`.
4. Print the resulting public HTTPS URL.

### GitHub Actions alternative

A `workflow_dispatch` workflow is provided at `.github/workflows/deploy-cloudrun.yml`. Trigger it from the GitHub Actions UI (Actions tab → "Deploy to Cloud Run" → Run workflow) and supply:
- **project_id** (required)
- **region** (default `us-central1`)
- **service** (default `data-analyzer-api`)

The public service URL is echoed into the job summary after a successful deploy.

**Required GitHub secret — `GCP_SA_KEY`:**
Create a GCP service account with the following IAM roles, generate a JSON key, and store it as the `GCP_SA_KEY` repository secret:

| IAM Role | Purpose |
|----------|---------|
| `roles/run.admin` | Deploy and manage Cloud Run services |
| `roles/cloudbuild.builds.editor` | Submit Cloud Build jobs |
| `roles/artifactregistry.admin` | Create repositories and push images |
| `roles/iam.serviceAccountUser` | Impersonate the Cloud Build service account |
| `roles/storage.admin` | Cloud Build source staging bucket |

**WORKLOAD IDENTITY FEDERATION is the more secure alternative.** WIF eliminates long-lived service account JSON keys by federating GitHub's OIDC token directly into GCP. See [google-github-actions/auth — Workload Identity Federation](https://github.com/google-github-actions/auth#setting-up-workload-identity-federation). Replace the `credentials_json` input with `workload_identity_provider` + `service_account` to adopt it.

### Grabbing the URL after deploy

```bash
gcloud run services describe data-analyzer-api \
    --region=us-central1 \
    --project=MY_PROJECT \
    --format='value(status.url)'
```

### End-to-end check

After deploy, run the live proof script against the public URL:

```bash
python3 scripts/e2e_a2a_check.py --url https://data-analyzer-xxxx.run.app
```

Or via environment variable:

```bash
E2E_URL=https://data-analyzer-xxxx.run.app python3 scripts/e2e_a2a_check.py
```

The script (stdlib-only, no pip installs) verifies:
1. `GET /health` returns `status: ok` with `a2a_backend` present.
2. `GET /.well-known/agent.json` card `url` equals `/a2a/data-analyzer`.
3. A base64-encoded CSV is sent via `call_agent_http()` and the response contains quality-pipeline keys (`checks`, `summary_stats`, etc.).

It prints a concise PASS/FAIL summary and exits non-zero on any failure.

### Security WARNING

`--allow-unauthenticated` makes the `/analyze`, `/data-info`, and `/a2a/data-analyzer` endpoints reachable by **anyone on the public internet**. This is acceptable for a temporary demo or internal proof-of-concept. Before processing real patient data or any sensitive information:

- Remove the `allUsers` IAM binding from the Cloud Run service.
- Add Cloud Run IAM invoker bindings for specific identities, or place an API Gateway in front with key/JWT enforcement.
- Consider VPC Service Controls for network-level isolation.

### Note on A2A wire framing compatibility

Our `/a2a/{agent_id}` endpoint speaks the **canonical DCRI JSON envelope** — a `{"payload": {...}, "metadata": {...}}` dict — which is what this repo's `a2a_runtime` shim produces and consumes. When the org installs the real `dcri-a2a-core` package (a2a-sdk based), verify whether its inbound route (`message/send`) expects the a2a-sdk JSON-RPC framing instead. The `a2a_runtime/__init__.py` indirection means no application code changes are needed — only the envelope shape at the HTTP boundary may need reconciling against the real package's router.
