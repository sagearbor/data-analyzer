# Azure CLI Deployment - Reference Only

⚠️ **These files are for reference and experimentation only**

This directory contains Azure deployment scripts created during development. They are intentionally excluded from production deployment workflows.

## Purpose

These scripts were developed to:
- Provide a reference implementation for Azure Container Apps deployment
- Enable testing in personal Azure subscriptions
- Serve as learning material for Azure infrastructure

## Status

- **Production Status**: Not used in organizational deployment
- **Development Status**: Potentially functional for personal Azure subscriptions, but untested.
- **Maintenance**: Preserved as-is for reference

## Contents

- `deploy.sh` - Azure Container Apps deployment automation
- `azure_config.yaml` - Azure resource configuration templates

## Usage

These scripts are available for personal experimentation or as reference material. For organizational deployment, consult your DevOps team's approved procedures.

## Personal Use

If you want to test this in your own Azure subscription:

### Prerequisites
- Azure CLI installed and configured
- Docker installed
- Personal Azure subscription
- Appropriate permissions

### Deployment
```bash
# From this directory
./deploy.sh deploy

# With custom app name
APP_NAME=myapp ./deploy.sh deploy

# View info
./deploy.sh info

# Cleanup
./deploy.sh cleanup
```

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `APP_NAME` | Application name | `data-analyzer` |
| `RESOURCE_GROUP` | Azure resource group | `rg-{APP_NAME}` |
| `LOCATION` | Azure region | `eastus` |
| `CONTAINER_REGISTRY` | ACR name | `{APP_NAME}registry` |

### Resources Created
- Resource Group
- Container Registry (ACR)
- Log Analytics Workspace
- Container Apps Environment
- Container App (web application)

## Troubleshooting

**Deployment fails with ACR permissions**
```bash
az acr update --name $CONTAINER_REGISTRY --admin-enabled true
```

**Container app won't start**
```bash
az containerapp logs show --name data-analyzer-web --resource-group rg-data-analyzer
```

**Not logged into Azure**
```bash
az login
```

---

**Note:** For production deployment, follow your organization's established procedures and infrastructure guidelines.
