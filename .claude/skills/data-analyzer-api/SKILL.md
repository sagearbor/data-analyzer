---
name: data-analyzer-api
description: Use this skill to analyze a data dictionary or check the deployed data-analyzer service's health via its REST API. Use when the user asks to parse/upload a data dictionary (REDCap CSV, JSON, TXT), retrieve a previously-parsed dictionary/program by ID or name, or verify the data-analyzer API is reachable. The API is VPN-only (Duke network) and requires an API key from the repo's .env file.
---

# Data Analyzer REST API

Call the deployed data-analyzer REST API to parse data dictionaries into
validation programs (schema + rules extraction via LLM) and retrieve
previously-saved programs.

**Base URL:** `https://data-analyzer-api.wonderfulground-6988c2db.eastus2.azurecontainerapps.io`

**Network requirement:** The API is VPN-only (Duke network). It is NOT
reachable from the open internet.

**Source of truth for routes:** `api_server.py` in this repo. If the deployed
behavior ever seems to disagree with this document, re-read that file — it is
the actual FastAPI route table, not `docs/API.md` (which documents an older,
separate stdio MCP server) and not the docstrings in `src/api_models.py`
(which describe some endpoints — `/api/v1/analyze`, `/api/v1/programs/*` —
that have Pydantic models defined but are **not currently wired up** as
routes in `api_server.py`). Only four routes actually exist today:

| Method | Path | Auth | Rate limit |
|---|---|---|---|
| GET | `/` | none | none |
| GET | `/api/v1/health` | none | 60/min |
| POST | `/api/v1/dictionary/parse` | `X-API-Key` | 5/min |
| GET | `/api/v1/dictionary/{dict_id}` | `X-API-Key` | 30/min |

Interactive Swagger UI (only useful when on VPN, in a browser):
`https://data-analyzer-api.wonderfulground-6988c2db.eastus2.azurecontainerapps.io/api/v1/docs`

## Step 0: Prerequisites (always check first)

1. **VPN connectivity** — run the health check before anything else:

   ```bash
   curl -s -m 10 -o /dev/null -w "HTTP %{http_code} in %{time_total}s\n" \
     https://data-analyzer-api.wonderfulground-6988c2db.eastus2.azurecontainerapps.io/api/v1/health
   ```

   - If this **times out** or returns curl error code `000` / `28` (connection
     timed out) / `6` (couldn't resolve host): the user is almost certainly
     off Duke VPN. Tell them to connect to the Duke VPN and retry. Do not
     proceed to authenticated calls until this succeeds.
   - If it returns `HTTP 200`, the service is reachable — proceed.

2. **API key present** — the key lives in the repo's gitignored `.env` file
   under the variable `DATA_ANALYZER_API_KEY` (confirmed in `api_server.py`,
   line ~218: `API_KEY = os.getenv("DATA_ANALYZER_API_KEY")`). Read it from
   the shell at call time — **never** hardcode it in a command you type out,
   never echo/print it, never paste it into chat:

   ```bash
   cd /path/to/data-analyzer   # repo root, wherever .env lives
   KEY=$(grep '^DATA_ANALYZER_API_KEY=' .env | cut -d= -f2-)
   if [ -z "$KEY" ]; then echo "DATA_ANALYZER_API_KEY not set in .env"; fi
   ```

   Use `$KEY` inside subsequent curl calls via `-H "X-API-Key: $KEY"`. Do not
   run `echo $KEY` or otherwise print it to output. If `DATA_ANALYZER_API_KEY`
   is absent from `.env`, tell the user — the deployed server may still be
   running with authentication effectively required (the deployment sets
   `APP_ENV=prod`, which makes the server refuse to start at all without this
   variable set server-side; your local `.env` copy must contain the matching
   key that was provisioned for this deployment, not an arbitrary value).

## Endpoint reference

### GET /api/v1/health

No auth. 60 requests/minute.

```bash
curl -s https://data-analyzer-api.wonderfulground-6988c2db.eastus2.azurecontainerapps.io/api/v1/health
```

Response shape (`HealthResponse` in `api_server.py`):

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2025-12-02T22:30:00.123456",
  "services": {
    "llm_client": true,
    "program_manager": true,
    "logic_validator": true,
    "mcp_server": true
  }
}
```

`status` is `"healthy"` only if both `program_manager` and `logic_validator`
are `true`; otherwise `"degraded"`. A degraded status with `llm_client: false`
means dictionary parsing will fail with 503 (Azure OpenAI not configured
server-side) even though the API itself is up.

### POST /api/v1/dictionary/parse

Requires `X-API-Key`. **5 requests/minute** (LLM calls are expensive — do not
retry-loop on 429).

Multipart form fields:
- `dictionary_file` (file, required) — one of `.csv`, `.json`, `.txt` (`.pdf`
  is accepted by the file-extension check but returns `501 Not Implemented`
  server-side; it is not actually supported yet).
- `save_program` (bool, form field, default `true`) — whether to cache the
  resulting validation program server-side.
- `program_name` (string, form field, optional) — custom name; auto-generated
  (timestamp + inferred name) if omitted.

```bash
curl -s -X POST \
  https://data-analyzer-api.wonderfulground-6988c2db.eastus2.azurecontainerapps.io/api/v1/dictionary/parse \
  -H "X-API-Key: $KEY" \
  -F "dictionary_file=@/path/to/dictionary.csv;type=text/csv" \
  -F "save_program=true" \
  -F "program_name=MyProgram_v1"
```

Response shape (`ParseDictionaryResponse`):

```json
{
  "program_id": "a1b2c3d4-...",
  "program_name": "20241202-143022-ClinicalTrial",
  "fields_extracted": 45,
  "rules_extracted": 23,
  "logic_rules_extracted": 8,
  "generated_code": "def validate_logic_rules(df):\n    ... (truncated to 500 chars, '...' appended if longer)",
  "schema": { "patient_id": {"type": "int", "required": true}, "...": "..." },
  "dictionary_format": "REDCap CSV",
  "generation_time_seconds": 3.5,
  "model_used": "gpt-5-nano"
}
```

Error cases to expect and how to interpret them:
- `400` — unsupported file extension, or a decode failure (file isn't valid
  UTF-8/Latin-1 text).
- `401` — missing `X-API-Key` header.
- `403` — `X-API-Key` present but wrong.
- `429` — rate limit exceeded (5/min); wait and retry, do not hammer it.
- `501` — you uploaded a `.pdf`; convert to CSV/JSON/TXT first.
- `503` — `program_manager`/LLM client not initialized server-side; check
  `/api/v1/health` — this is a server-side config issue, not something the
  caller can fix by retrying.

### GET /api/v1/dictionary/{dict_id}

Requires `X-API-Key`. 30 requests/minute. `{dict_id}` may be a program UUID,
its auto-generated name, or a user-assigned alias.

```bash
curl -s https://data-analyzer-api.wonderfulground-6988c2db.eastus2.azurecontainerapps.io/api/v1/dictionary/<dict_id> \
  -H "X-API-Key: $KEY"
```

Response shape (`ProgramDetail`) — includes everything from the parse
response plus full (untruncated) `generated_code`, `aliases`,
`conditional_rules`, `created_by`, `created_at`, `last_used`, `use_count`,
`status` (`active`/`deleted`), `version`. Returns `404` if not found or if the
program has been soft-deleted.

## Worked example: parse a tiny dictionary

1. Create a minimal REDCap-style CSV dictionary:

   ```bash
   cat > /tmp/mini_dictionary.csv <<'EOF'
   Variable / Field Name,Field Type,Field Label,Text Validation Type OR Show Slider Number,Text Validation Min,Text Validation Max
   patient_id,text,Patient ID,integer,1,999999
   age,text,Age at enrollment,integer,0,120
   enrollment_date,text,Enrollment Date,date_ymd,,
   EOF
   ```

2. Check prerequisites (VPN + key), per Step 0 above.

3. Submit it:

   ```bash
   curl -s -X POST \
     https://data-analyzer-api.wonderfulground-6988c2db.eastus2.azurecontainerapps.io/api/v1/dictionary/parse \
     -H "X-API-Key: $KEY" \
     -F "dictionary_file=@/tmp/mini_dictionary.csv;type=text/csv" \
     -F "save_program=false" \
     | tee /tmp/parse_response.json
   ```

   `save_program=false` is a good default for a one-off test — it parses and
   returns the result without cluttering the server's saved-program list.

4. Interpret the response:
   - `fields_extracted` should be `3` (patient_id, age, enrollment_date).
   - `schema` will show inferred types/constraints per field (e.g. `age` as
     an int with range 0-120).
   - `rules_extracted` / `logic_rules_extracted` count basic and conditional
     validation rules the LLM inferred beyond simple type checks.
   - If `save_program=true` had been used, note the returned `program_id` or
     `program_name` — that's what you'd pass to `GET /api/v1/dictionary/{dict_id}`
     to retrieve it later.

5. Clean up the scratch file when done: `rm /tmp/mini_dictionary.csv /tmp/parse_response.json`.

## Notes for the calling agent

- Never print, log, or echo the contents of `$KEY`. Only use it inside
  `-H "X-API-Key: $KEY"`.
- Respect the 5/min rate limit on `/dictionary/parse` — space out repeated
  test calls; don't loop.
- If a call fails with a connection error partway through a session (not at
  Step 0), re-run the Step 0 health check before concluding it's an API bug —
  VPN connections can drop mid-session.
- This skill only covers the four routes that actually exist in
  `api_server.py`. If a task requires `/api/v1/analyze`,
  `/api/v1/analyze/with-program`, or `/api/v1/programs/*` (listing, aliasing,
  deleting programs), those are defined as Pydantic models in
  `src/api_models.py` but have no live route yet — check with the user/repo
  maintainer rather than assuming they work.
