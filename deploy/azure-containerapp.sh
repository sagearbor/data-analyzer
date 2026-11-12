#!/usr/bin/env bash
# deploy/azure-containerapp.sh — DEFAULT deploy target for the data-analyzer project.
#
# This script deploys the data-analyzer API to Azure Container Apps, which is the
# primary and default deployment target for this project.
# For the optional Google Cloud Run path, see deploy/cloudrun.sh.
#
# USAGE
#   ACR_NAME=myregistry123 ./deploy/azure-containerapp.sh
#
# REQUIRED ENVIRONMENT VARIABLES
#   ACR_NAME   Azure Container Registry name (REQUIRED — no default).
#              Must be globally unique across all of Azure.
#              Constraints: 5-50 characters, alphanumeric only (no hyphens or underscores).
#              Example: ACR_NAME=dataanalyzer42acr
#
# OPTIONAL ENVIRONMENT VARIABLES (with sensible defaults)
#   RESOURCE_GROUP   Azure resource group name    (default: rg-data-analyzer)
#   LOCATION         Azure region                 (default: eastus)
#   ACA_ENV          Container Apps environment   (default: aca-env-data-analyzer)
#   APP_NAME         Container App name           (default: data-analyzer-api)
#   IMAGE_TAG        Image tag to build/deploy    (default: latest)
#
# PREREQUISITES
#   - Azure CLI authenticated:  az login
#   - Sufficient permissions on the subscription/resource group:
#       Contributor or Owner on the resource group (for resource creation)
#       AcrPush on the ACR (for image push via acr build)
#
# WHAT THIS SCRIPT DOES
#   1. Creates the resource group (idempotent).
#   2. Creates the Azure Container Registry with Basic SKU (idempotent).
#   3. Enables ACR admin credentials (used for Container Apps pull authentication).
#   4. Builds the image in the cloud via `az acr build` (no local Docker required).
#   5. Ensures the Container Apps extension and environment exist (idempotent).
#   6. Creates or updates the Container App with external ingress on port 8080.
#   7. Prints the public HTTPS URL and reminds you to run the e2e check.
#
# AUTHENTICATION NOTE — ACR admin credentials
#   This script uses ACR admin credentials (username/password) for the Container App
#   to pull images. Admin credentials are simple and sufficient for demos and small teams.
#   For production or team environments consider using a managed identity instead:
#     az containerapp create ... --registry-identity system
#   and grant the app's system identity AcrPull on the ACR.
#
# SECURITY NOTE
#   External ingress is public/unauthenticated — anyone on the internet can call
#   /analyze, /data-info, and /a2a/data-analyzer. This is acceptable for demos and
#   proofs-of-concept. Before processing real or sensitive data, add authentication:
#     - Azure API Management in front with key/JWT enforcement
#     - Microsoft Entra ID (formerly Azure AD) authentication on the Container App
#       (az containerapp auth update --enabled true)
#     - VNet integration with private ingress only

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

: "${ACR_NAME:?ERROR: ACR_NAME environment variable must be set. It must be globally unique, 5-50 alphanumeric characters (no hyphens). Example: ACR_NAME=dataanalyzer42acr}"

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-data-analyzer}"
LOCATION="${LOCATION:-eastus}"
ACA_ENV="${ACA_ENV:-aca-env-data-analyzer}"
APP_NAME="${APP_NAME:-data-analyzer-api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "================================================================"
echo "  Deploying data-analyzer API to Azure Container Apps (DEFAULT)"
echo "  Resource Group : ${RESOURCE_GROUP}"
echo "  Location       : ${LOCATION}"
echo "  ACR Name       : ${ACR_NAME}"
echo "  ACA Environment: ${ACA_ENV}"
echo "  App Name       : ${APP_NAME}"
echo "  Image Tag      : ${IMAGE_TAG}"
echo "================================================================"

# ---------------------------------------------------------------------------
# Step 1: Ensure the resource group exists
# ---------------------------------------------------------------------------
echo ""
echo "[1/6] Ensuring resource group '${RESOURCE_GROUP}' exists..."
az group create \
    --name "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --output none
echo "  Resource group ready."

# ---------------------------------------------------------------------------
# Step 2: Ensure the Azure Container Registry exists
# ---------------------------------------------------------------------------
echo ""
echo "[2/6] Ensuring Azure Container Registry '${ACR_NAME}' exists..."
az acr create \
    --name "${ACR_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --sku Basic \
    --output none \
    2>&1 || true   # 'already exists' is not an error

# Enable admin credentials so the Container App can authenticate to pull images.
# (See AUTHENTICATION NOTE in the header for a managed-identity alternative.)
echo "  Enabling ACR admin credentials..."
az acr update \
    --name "${ACR_NAME}" \
    --admin-enabled true \
    --output none
echo "  ACR ready."

# ---------------------------------------------------------------------------
# Step 3: Build the image in the cloud via ACR Tasks (no local Docker needed)
# ---------------------------------------------------------------------------
echo ""
echo "[3/6] Building image '${APP_NAME}:${IMAGE_TAG}' via az acr build..."
# Run from the repo root (parent of this script's directory).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
az acr build \
    --registry "${ACR_NAME}" \
    --image "${APP_NAME}:${IMAGE_TAG}" \
    --file "${REPO_ROOT}/Dockerfile.api" \
    "${REPO_ROOT}"
echo "  Image build and push complete."

# ---------------------------------------------------------------------------
# Step 4: Ensure the Container Apps extension and environment exist
# ---------------------------------------------------------------------------
echo ""
echo "[4/6] Ensuring Container Apps extension and environment exist..."
az extension add --name containerapp --upgrade --output none 2>&1 || true

az containerapp env create \
    --name "${ACA_ENV}" \
    --resource-group "${RESOURCE_GROUP}" \
    --location "${LOCATION}" \
    --output none \
    2>&1 || true   # 'already exists' is not an error
echo "  Container Apps environment ready."

# ---------------------------------------------------------------------------
# Step 5: Derive ACR login server and credentials
# ---------------------------------------------------------------------------
echo ""
echo "[5/6] Retrieving ACR credentials..."
LOGIN_SERVER="$(az acr show --name "${ACR_NAME}" --query loginServer --output tsv)"
ACR_USERNAME="$(az acr credential show --name "${ACR_NAME}" --query username --output tsv)"
ACR_PASSWORD="$(az acr credential show --name "${ACR_NAME}" --query 'passwords[0].value' --output tsv)"
echo "  ACR login server: ${LOGIN_SERVER}"

# ---------------------------------------------------------------------------
# Step 6: Create or update the Container App
# ---------------------------------------------------------------------------
echo ""
echo "[6/6] Creating or updating Container App '${APP_NAME}'..."

# Attempt to create the app. If it already exists, fall back to update.
if az containerapp create \
    --name "${APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --environment "${ACA_ENV}" \
    --image "${LOGIN_SERVER}/${APP_NAME}:${IMAGE_TAG}" \
    --target-port 8080 \
    --ingress external \
    --registry-server "${LOGIN_SERVER}" \
    --registry-username "${ACR_USERNAME}" \
    --registry-password "${ACR_PASSWORD}" \
    --query properties.configuration.ingress.fqdn \
    --output tsv \
    2>/dev/null; then
    echo "  Container App created."
else
    echo "  Container App already exists — updating image..."
    az containerapp update \
        --name "${APP_NAME}" \
        --resource-group "${RESOURCE_GROUP}" \
        --image "${LOGIN_SERVER}/${APP_NAME}:${IMAGE_TAG}" \
        --output none
    echo "  Container App updated."
fi

# ---------------------------------------------------------------------------
# Print the public URL
# ---------------------------------------------------------------------------
echo ""
FQDN="$(az containerapp show \
    --name "${APP_NAME}" \
    --resource-group "${RESOURCE_GROUP}" \
    --query properties.configuration.ingress.fqdn \
    --output tsv)"

PUBLIC_URL="https://${FQDN}"

echo "================================================================"
echo "  Deployment complete!"
echo "  Public URL : ${PUBLIC_URL}"
echo ""
echo "  Verify with:"
echo "    curl ${PUBLIC_URL}/health"
echo ""
echo "  Run the end-to-end check:"
echo "    python3 scripts/e2e_a2a_check.py --url ${PUBLIC_URL}"
echo ""
echo "  SECURITY REMINDER: External ingress is currently public/"
echo "  unauthenticated. Add Entra ID authentication or an API"
echo "  gateway before exposing real/sensitive data."
echo "================================================================"
