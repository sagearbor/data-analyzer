#!/usr/bin/env bash
# deploy/cloudrun.sh — Manual deploy script for the data-analyzer API on Cloud Run.
#
# ######################################################################
# # NOT USED FOR THE RELEASE. The supported production target is Azure #
# # Container Apps (deploy/aca/). This Cloud Run path is kept only     #
# # pending an owner retire/keep decision (ADP-81). If it is kept it   #
# # must deploy with IAM-authenticated ingress, APP_ENV=prod and an    #
# # API-key secret binding - the script now refuses to do otherwise.   #
# ######################################################################
#
# USAGE
#   PROJECT_ID=my-gcp-project ./deploy/cloudrun.sh
#
# REQUIRED ENVIRONMENT VARIABLES
#   PROJECT_ID   GCP project id (no default — script exits if unset).
#   API_KEY_SECRET  Name of an existing Secret Manager secret holding the API key
#                (bound as DATA_ANALYZER_API_KEY). Script refuses to deploy without it.
#
# OPTIONAL ENVIRONMENT VARIABLES (with sensible defaults)
#   REGION       GCP region          (default: us-central1)
#   SERVICE      Cloud Run service   (default: data-analyzer-api)
#   IMAGE        Full image path     (default: derived from REGION/PROJECT_ID/SERVICE)
#
# PREREQUISITES
#   - gcloud CLI authenticated:  gcloud auth login  (or use a service account)
#   - gcloud application-default credentials configured (for Cloud Build):
#       gcloud auth application-default login
#   - The caller needs these IAM roles on the project:
#       roles/run.admin
#       roles/cloudbuild.builds.editor
#       roles/artifactregistry.admin
#       roles/iam.serviceAccountUser          (to act as the Cloud Build SA)
#       roles/storage.admin                   (Cloud Build source staging bucket)
#
# WHAT THIS SCRIPT DOES
#   1. Validates PROJECT_ID is set.
#   2. Ensures the Artifact Registry repository 'apps' exists (idempotent).
#   3. Runs Cloud Build (cloudbuild.yaml) to build and push the image.
#   4. Deploys the image to Cloud Run with IAM-authenticated ingress
#      (--no-allow-unauthenticated), APP_ENV=prod and the API key bound from
#      Secret Manager. Callers need roles/run.invoker AND the X-API-Key header.
#   5. Prints the service URL.
#
# SECURITY NOTE
#   The service is never made public by this script. Do not add
#   --allow-unauthenticated.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

: "${PROJECT_ID:?ERROR: PROJECT_ID environment variable must be set.}"
: "${API_KEY_SECRET:?ERROR: API_KEY_SECRET (Secret Manager secret holding the API key) must be set; refusing to deploy without an API key binding.}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-data-analyzer-api}"
IMAGE="${IMAGE:-${REGION}-docker.pkg.dev/${PROJECT_ID}/apps/${SERVICE}:latest}"

echo "================================================================"
echo "  Deploying data-analyzer API to Cloud Run"
echo "  Project  : ${PROJECT_ID}"
echo "  Region   : ${REGION}"
echo "  Service  : ${SERVICE}"
echo "  Image    : ${IMAGE}"
echo "================================================================"

# ---------------------------------------------------------------------------
# Step 1: Ensure Artifact Registry repository 'apps' exists
# ---------------------------------------------------------------------------
echo ""
echo "[1/3] Ensuring Artifact Registry repository 'apps' exists..."
gcloud artifacts repositories create apps \
    --repository-format=docker \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --description="Container images for deployed services" \
    2>&1 || true   # 'already exists' is not an error

# Configure Docker authentication for the registry (idempotent).
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet

# ---------------------------------------------------------------------------
# Step 2: Build and push the image via Cloud Build
# ---------------------------------------------------------------------------
echo ""
echo "[2/3] Building and pushing image via Cloud Build..."
# Run from the repo root (parent of this script's directory).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
gcloud builds submit \
    --config="${REPO_ROOT}/cloudbuild.yaml" \
    --substitutions="_IMAGE=${IMAGE}" \
    --project="${PROJECT_ID}" \
    "${REPO_ROOT}"

# ---------------------------------------------------------------------------
# Step 3: Deploy to Cloud Run
# ---------------------------------------------------------------------------
echo ""
echo "[3/3] Deploying to Cloud Run..."
gcloud run deploy "${SERVICE}" \
    --image="${IMAGE}" \
    --region="${REGION}" \
    --platform=managed \
    --no-allow-unauthenticated \
    --port=8000 \
    --set-env-vars="APP_ENV=prod" \
    --set-secrets="DATA_ANALYZER_API_KEY=${API_KEY_SECRET}:latest" \
    --project="${PROJECT_ID}"

# ---------------------------------------------------------------------------
# Print the public URL
# ---------------------------------------------------------------------------
echo ""
SERVICE_URL="$(gcloud run services describe "${SERVICE}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --format='value(status.url)')"

echo "================================================================"
echo "  Deployment complete!"
echo "  Service URL (IAM-authenticated, not public) : ${SERVICE_URL}"
echo ""
echo "  Verify with an identity token and the API key header:"
echo "    curl -H \"Authorization: Bearer \$(gcloud auth print-identity-token)\" ${SERVICE_URL}/api/v1/health"
echo "================================================================"
