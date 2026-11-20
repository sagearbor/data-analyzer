Deploy release 1.0 to the development environment (Azure Container Apps) and check the deployed site.

# Acceptance criteria
1. Both parts (the page and the API service) run the same released code version. Today they do not: the API service is on an older build than the page.
2. The health check answers on both, and the browser test passes against the deployed address.
3. The model list is set on the server.
4. The version that is running is written down in the release documents.
