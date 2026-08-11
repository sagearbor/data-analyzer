#!/usr/bin/env bash
# ============================================================
# OPTIONAL / NON-DEFAULT — Google Cloud Run path.
# The DEFAULT deploy target for this project is AZURE
# (Azure Container Apps): see deploy/azure-containerapp.sh and
# .github/workflows/deploy-azure.yml. These GCP files exist only
# for an optional public demo; do not use unless you are
# explicitly deploying to Google Cloud.
# ============================================================
# deploy/cloudrun.sh — Manual deploy script for the data-analyzer API on Cloud Run.
#
# USAGE
#   PROJECT_ID=my-gcp-project ./deploy/cloudrun.sh
#
# REQUIRED ENVIRONMENT VARIABLES
#   PROJECT_ID   GCP project id (no default — script exits if unset).
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
#   4. Deploys the image to Cloud Run as a public unauthenticated service.
#   5. Prints the resulting public URL.
#
# SECURITY NOTE
#   --allow-unauthenticated makes the endpoint public. Suitable for demos.
#   Add authentication (Cloud Run IAM or an API gateway) before exposing
#   real sensitive data.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

: "${PROJECT_ID:?ERROR: PROJECT_ID environment variable must be set.}"
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
    --allow-unauthenticated \
    --port=8080 \
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
echo "  Public URL : ${SERVICE_URL}"
echo ""
echo "  Verify with:"
echo "    curl ${SERVICE_URL}/health"
echo ""
echo "  Run the end-to-end check:"
echo "    python3 scripts/e2e_a2a_check.py --url ${SERVICE_URL}"
echo "================================================================"
