# Integration Guide - Calling the Data Analyzer Service

This guide describes what the **released** service exposes. Earlier versions of this document
described `/a2a/*` and `/.well-known/agent.json` routes and a public, unauthenticated Cloud Run
deployment. The released service has none of those; they have been removed from this guide.

## What is released

| Interface | Status |
|-----------|--------|
| **REST API** (`api_server.py`, `/api/v1/*`) | Released. Requires the `X-API-Key` header on analysis and dictionary routes. |
| **Streamlit web UI** (`web_app.py`) | Released; calls the REST API. |
| **MCP server** (`mcp_server.py`, stdio) | Optional local adapter. Not in the released images. |
| **A2A adapter** (`a2a_agent.py`, `a2a_runtime/`, `analysis_service.py`) | Optional and experimental. Not in the released images; the API registers no `/a2a/*` and no `/.well-known/agent.json` route. |

The MCP and A2A adapters accept caller-supplied content and must **never be exposed to a network
without authentication in front of them**. Do not deploy them as public or unauthenticated
endpoints.

## REST API

Routes, request bodies, curl examples, size limits and the error format are documented in the
"Calling the REST API from another program" section of the repository [README](../README.md) and in
[API_COMPREHENSIVE.md](API_COMPREHENSIVE.md) / [API_ANALYZE_ENDPOINTS.md](API_ANALYZE_ENDPOINTS.md).

Summary of the released routes:

| Route | Purpose | API key |
|---|---|---|
| `GET /api/v1/health` | Service status and version | no |
| `GET /api/v1/version` | Name and version | no |
| `POST /api/v1/analyze` | Analyze an uploaded file (multipart) | `X-API-Key` |
| `POST /api/v1/analyze/json` | Analyze a dataset sent inline as JSON | `X-API-Key` |
| `POST /api/v1/dictionary/parse` | Parse a data dictionary (may call the Azure OpenAI model) | `X-API-Key` |
| `GET /api/v1/dictionary/{dict_id}` | Retrieve a parsed dictionary/program | `X-API-Key` |

The OpenAPI/Swagger pages (`/api/v1/docs`, `/api/v1/redoc`, `/api/v1/openapi.json`) are open
without an API key by design; they describe the schema only.

## Authentication

In deployed environments (`APP_ENV=prod`, or any Azure Container Apps runtime) the API refuses to
start unless `DATA_ANALYZER_API_KEY` is set. Clients send that value in `X-API-Key`. Ingress is
internal-network only on the supported Azure deployment; see
[deploy/aca/README.md](../deploy/aca/README.md).

## Dictionary parsing limits

Dictionary parsing may send dictionary text to the configured Azure OpenAI deployment. Each parse
is bounded by environment variables (defaults in parentheses): `LLM_MAX_INPUT_CHARS` (500000)
and `LLM_MAX_CHUNKS` (20); a dictionary over either limit is rejected with an explicit error,
never silently truncated. Requests ask the service not to retain the exchange
(`LLM_RESPONSES_STORE=false`; set `omit` only if a model rejects the `store` field). The model
endpoint must be https (`LLM_ALLOW_INSECURE_ENDPOINT=1` is a local-development override only) and
redirects from it are treated as failures.

## Alternative deployment paths

The only supported production target is Azure Container Apps (see `deploy/aca/`). The Cloud Run
script and workflow (`deploy/cloudrun.sh`, `.github/workflows/deploy-cloudrun.yml`) are **not used
for the release** and are pending a retire-or-keep decision; they require IAM-authenticated
ingress and an API-key secret binding and must not be used to publish an unauthenticated service.
