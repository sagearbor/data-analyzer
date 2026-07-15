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

- **Duke VPN connected** — the ACA environment is VNET-internal (no public ingress); nothing below resolves off-VPN.
- Real infra IDs live in the gitignored **`azure-environment.md`** at the repo root — plug those into `containerapp.yaml`'s `<PLACEHOLDER>`s. (Kept out of git because the repo is public.)
- `az` CLI logged in to the dev subscription, with the `containerapp` extension: `az extension add --name containerapp`.

## 1. Build & push the image (linux/amd64)

The ACA nodes are **amd64** — build for that platform explicitly (matters if you ever build from an ARM Mac; fine natively on x86 WSL). Use the ACR login server from `azure-environment.md`.

```bash
ACR=<acr-login-server>       # from azure-environment.md
az acr login --name "${ACR%%.*}"
docker build --platform linux/amd64 -f Dockerfile.api -t "$ACR/data-analyzer-api:latest" .
docker push "$ACR/data-analyzer-api:latest"
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

## ⚠️ Program-cache persistence (functional gotcha, read before demo)

`src/program_cache.py` stores saved validation programs in a **SQLite file on
the container's local disk** (`~/.data_analyzer/programs.db`). The Dockerfile
`VOLUME` is ignored by ACA. With the IT-specified scale (min replicas **0**,
max **2**):

- Scale-to-zero → cold start loses the SQLite file (saved programs vanish).
- Two replicas → each has its own SQLite file (inconsistent cache).

For a **live demo tomorrow**, deploy with `minReplicas: 1, maxReplicas: 1` so a
single replica stays warm and data survives within the session (flag this as a
temporary deviation from IT's 0/2 guidance). The **real fix** is migrating the
program cache to the provided Postgres (a dedicated `data_analyzer` DB, AAD-token
auth — host in `azure-environment.md`) or an Azure Files mount — tracked as an
open item, not done yet.

## Postgres (when you migrate off SQLite)

Auth uses an **Entra ID access token via the user-assigned managed identity**,
not the `psqladmin` password (that's break-glass, Key-Vault-only). Create a
dedicated DB + per-app user — do **not** use the `postgres` system DB. Details
in `azure-environment.md`.

## Verify after deploy

```bash
FQDN=$(az containerapp show --name "$APP" --resource-group "$RG" \
  --query properties.configuration.ingress.fqdn -o tsv)
curl -fsS "https://$FQDN/api/v1/health"                 # -> {"status":"healthy",...}
curl -fsS -H "X-API-Key: <key>" "https://$FQDN/api/v1/dictionary/parse" # authed route
```
