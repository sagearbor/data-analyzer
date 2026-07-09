# Azure Container Apps (ACA) Deployment — Data Analyzer API

This directory holds the **ACA-native** deployment definition for the FastAPI
service (`api_server.py`). It is intentionally separate from the repo's
`docker-compose.*.yml` files, which describe the on-prem / VM deployment (a
Streamlit app behind an NGINX helper sidecar) and are **not** used by ACA.

| Artifact | Used for |
|---|---|
| `deploy/aca/containerapp.yaml` | Azure Container Apps (this doc) |
| `Dockerfile.api` | The image ACA runs (FastAPI + uvicorn, non-root, port 8000) |
| `docker-compose.api.yml` / `docker-compose.dev.yml` | Local dev / VM only |
| `Dockerfile` + `entrypoint.sh` | Streamlit web app (separate deployment) |

## Prerequisites

- An Azure Container Registry (ACR) and an ACA managed environment (IT provides these).
- `az` CLI logged in, with the `containerapp` extension: `az extension add --name containerapp`.

## 1. Build & push the image

```bash
ACR=<your-acr-name>
az acr login --name "$ACR"
docker build -f Dockerfile.api -t "$ACR.azurecr.io/data-analyzer-api:latest" .
docker push "$ACR.azurecr.io/data-analyzer-api:latest"
```

## 2. Set secrets (never commit these)

```bash
RG=<resource-group>
APP=data-analyzer-api
az containerapp secret set --name "$APP" --resource-group "$RG" --secrets \
  data-analyzer-api-key="$(openssl rand -hex 32)" \
  azure-openai-api-key="<your-azure-openai-key>"
```

The generated `data-analyzer-api-key` is what API clients must send in the
`X-API-Key` header. Store it in your secret manager and share only with
authorized callers.

## 3. Deploy

Fill in the `<...>` placeholders in `containerapp.yaml` (subscription, RG,
environment id, ACR name, Azure OpenAI endpoint, frontend origin), then:

```bash
az containerapp create --name "$APP" --resource-group "$RG" \
  --environment <ACA_ENVIRONMENT> --yaml deploy/aca/containerapp.yaml
```

Updates use the same file: `az containerapp update --name "$APP" --resource-group "$RG" --yaml deploy/aca/containerapp.yaml`.

## Security-relevant configuration (what IT will ask about)

- **Auth is fail-closed in prod.** `APP_ENV=prod` makes `api_server.py` refuse
  to start if `DATA_ANALYZER_API_KEY` is unset — no silent open endpoint.
- **Credentials are compared in constant time** (`secrets.compare_digest`).
- **CORS is deny-by-default in prod.** Set `ALLOWED_ORIGINS` to the exact
  frontend origin(s); leaving it empty blocks all cross-origin browser calls.
- **Container runs as non-root** (`apiuser`, UID 1000) with a read-only-friendly
  layout and a minimal runtime image (multi-stage build).
- **Secrets are never in the image or git** — they come from the ACA secret
  store (or Key Vault references) at runtime.
- **Health probe** is HTTP `GET /api/v1/health` (ACA uses this, not the
  Dockerfile `HEALTHCHECK`, which ACA ignores).
- **Rate limiting** is per-client-IP via slowapi. Verify what client IP the app
  sees behind ACA ingress after first deploy (see meeting brief open item).

## Verify after deploy

```bash
FQDN=$(az containerapp show --name "$APP" --resource-group "$RG" \
  --query properties.configuration.ingress.fqdn -o tsv)
curl -fsS "https://$FQDN/api/v1/health"                 # -> {"status":"healthy",...}
curl -fsS -H "X-API-Key: <key>" "https://$FQDN/api/v1/dictionary/parse" # authed route
```
