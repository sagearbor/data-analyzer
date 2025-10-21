#!/bin/bash

# Azure Data Analyzer Deployment Script
# This script deploys the data analyzer to Azure Container Apps

set -e

# Configuration
APP_NAME="data-analyzer"
RESOURCE_GROUP="rg-${APP_NAME}"
LOCATION="eastus"
CONTAINER_REGISTRY="${APP_NAME}registry"
IMAGE_NAME="data-analyzer"
IMAGE_TAG="latest"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Azure CLI is installed
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check if logged into Azure
    if ! az account show &> /dev/null; then
        log_error "Not logged into Azure. Please run 'az login' first."
        exit 1
    fi
    
    log_info "Prerequisites check passed!"
}

create_resource_group() {
    log_info "Creating resource group: $RESOURCE_GROUP"
    
    if az group show --name $RESOURCE_GROUP &> /dev/null; then
        log_warn "Resource group $RESOURCE_GROUP already exists"
    else
        az group create --name $RESOURCE_GROUP --location $LOCATION
        log_info "Resource group created successfully"
    fi
}

create_container_registry() {
    log_info "Creating Azure Container Registry: $CONTAINER_REGISTRY"
    
    if az acr show --name $CONTAINER_REGISTRY --resource-group $RESOURCE_GROUP &> /dev/null; then
        log_warn "Container registry $CONTAINER_REGISTRY already exists"
    else
        az acr create \
            --resource-group $RESOURCE_GROUP \
            --name $CONTAINER_REGISTRY \
            --sku Basic \
            --admin-enabled true
        log_info "Container registry created successfully"
    fi
    
    # Enable admin user
    az acr update --name $CONTAINER_REGISTRY --admin-enabled true
}

build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    # Get ACR login server
    ACR_LOGIN_SERVER=$(az acr show --name $CONTAINER_REGISTRY --resource-group $RESOURCE_GROUP --query loginServer --output tsv)
    
    # Login to ACR
    az acr login --name $CONTAINER_REGISTRY
    
    # Build image
    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
    
    # Tag image for ACR
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}
    
    # Push image
    docker push ${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}
    
    log_info "Image pushed successfully to $ACR_LOGIN_SERVER"
}

create_container_app_environment() {
    log_info "Creating Container Apps environment..."
    
    # Install containerapp extension if not already installed
    az extension add --name containerapp --upgrade &> /dev/null || true
    
    # Register providers
    az provider register --namespace Microsoft.App --wait
    az provider register --namespace Microsoft.OperationalInsights --wait
    
    # Create log analytics workspace
    LOG_ANALYTICS_WORKSPACE="${APP_NAME}-logs"
    
    if az monitor log-analytics workspace show --resource-group $RESOURCE_GROUP --workspace-name $LOG_ANALYTICS_WORKSPACE &> /dev/null; then
        log_warn "Log Analytics workspace already exists"
    else
        az monitor log-analytics workspace create \
            --resource-group $RESOURCE_GROUP \
            --workspace-name $LOG_ANALYTICS_WORKSPACE \
            --location $LOCATION
    fi
    
    # Get workspace credentials
    LOG_ANALYTICS_WORKSPACE_CLIENT_ID=$(az monitor log-analytics workspace show --query customerId --output tsv --resource-group $RESOURCE_GROUP --workspace-name $LOG_ANALYTICS_WORKSPACE)
    LOG_ANALYTICS_WORKSPACE_CLIENT_SECRET=$(az monitor log-analytics workspace get-shared-keys --query primarySharedKey --output tsv --resource-group $RESOURCE_GROUP --workspace-name $LOG_ANALYTICS_WORKSPACE)
    
    # Create container app environment
    ENVIRONMENT_NAME="${APP_NAME}-env"
    
    if az containerapp env show --name $ENVIRONMENT_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
        log_warn "Container app environment already exists"
    else
        az containerapp env create \
            --name $ENVIRONMENT_NAME \
            --resource-group $RESOURCE_GROUP \
            --location $LOCATION \
            --logs-workspace-id $LOG_ANALYTICS_WORKSPACE_CLIENT_ID \
            --logs-workspace-key $LOG_ANALYTICS_WORKSPACE_CLIENT_SECRET
        log_info "Container app environment created successfully"
    fi
}

deploy_container_app() {
    log_info "Deploying container app..."
    
    # Get ACR credentials
    ACR_LOGIN_SERVER=$(az acr show --name $CONTAINER_REGISTRY --resource-group $RESOURCE_GROUP --query loginServer --output tsv)
    ACR_USERNAME=$(az acr credential show --name $CONTAINER_REGISTRY --resource-group $RESOURCE_GROUP --query username --output tsv)
    ACR_PASSWORD=$(az acr credential show --name $CONTAINER_REGISTRY --resource-group $RESOURCE_GROUP --query passwords[0].value --output tsv)
    
    CONTAINER_APP_NAME="${APP_NAME}-web"
    ENVIRONMENT_NAME="${APP_NAME}-env"
    
    # Deploy or update container app
    if az containerapp show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP &> /dev/null; then
        log_info "Updating existing container app..."
        az containerapp update \
            --name $CONTAINER_APP_NAME \
            --resource-group $RESOURCE_GROUP \
            --image "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}"
    else
        log_info "Creating new container app..."
        az containerapp create \
            --name $CONTAINER_APP_NAME \
            --resource-group $RESOURCE_GROUP \
            --environment $ENVIRONMENT_NAME \
            --image "${ACR_LOGIN_SERVER}/${IMAGE_NAME}:${IMAGE_TAG}" \
            --registry-server $ACR_LOGIN_SERVER \
            --registry-username $ACR_USERNAME \
            --registry-password $ACR_PASSWORD \
            --target-port 8501 \
            --ingress external \
            --cpu 1.0 \
            --memory 2Gi \
            --min-replicas 1 \
            --max-replicas 10 \
            --env-vars STREAMLIT_SERVER_PORT=8501 STREAMLIT_SERVER_ADDRESS=0.0.0.0
    fi
    
    # Get the app URL
    APP_URL=$(az containerapp show --name $CONTAINER_APP_NAME --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn --output tsv)
    
    log_info "Container app deployed successfully!"
    log_info "Access your application at: https://$APP_URL"
}

cleanup() {
    read -p "Do you want to delete all resources? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Deleting resource group: $RESOURCE_GROUP"
        az group delete --name $RESOURCE_GROUP --yes --no-wait
        log_info "Resource group deletion initiated"
    else
        log_info "Cleanup cancelled"
    fi
}

show_deployment_info() {
    log_info "Deployment Information:"
    echo "=========================="
    echo "Resource Group: $RESOURCE_GROUP"
    echo "Location: $LOCATION"
    echo "Container Registry: $CONTAINER_REGISTRY"
    echo "Container App: ${APP_NAME}-web"
    echo "Environment: ${APP_NAME}-env"
    
    # Get app URL if it exists
    if az containerapp show --name "${APP_NAME}-web" --resource-group $RESOURCE_GROUP &> /dev/null; then
        APP_URL=$(az containerapp show --name "${APP_NAME}-web" --resource-group $RESOURCE_GROUP --query properties.configuration.ingress.fqdn --output tsv)
        echo "Application URL: https://$APP_URL"
    fi
    echo "=========================="
}

# Main deployment function
deploy() {
    log_info "Starting Data Analyzer deployment to Azure..."
    
    check_prerequisites
    create_resource_group
    create_container_registry
    build_and_push_image
    create_container_app_environment
    deploy_container_app
    
    log_info "Deployment completed successfully!"
    show_deployment_info
}

# Script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  deploy    Deploy the data analyzer to Azure"
    echo "  cleanup   Delete all Azure resources"
    echo "  info      Show deployment information"
    echo "  help      Show this help message"
    echo ""
    echo "Environment Variables (optional):"
    echo "  APP_NAME           Application name (default: data-analyzer)"
    echo "  RESOURCE_GROUP     Resource group name (default: rg-\$APP_NAME)"
    echo "  LOCATION           Azure region (default: eastus)"
    echo "  CONTAINER_REGISTRY Registry name (default: \${APP_NAME}registry)"
    echo ""
    echo "Examples:"
    echo "  $0 deploy                    # Deploy with defaults"
    echo "  APP_NAME=myapp $0 deploy     # Deploy with custom app name"
    echo "  $0 cleanup                   # Delete all resources"
    echo "  $0 info                      # Show current deployment info"
}

# Main script logic
case "${1:-help}" in
    deploy)
        deploy
        ;;
    cleanup)
        cleanup
        ;;
    info)
        show_deployment_info
        ;;
    help)
        usage
        ;;
    *)
        log_error "Unknown command: $1"
        usage
        exit 1
        ;;
esac
