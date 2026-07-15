# Data Analyzer REST API - Comprehensive Reference

**Version:** 1.0.0
**Base URL:** `http://localhost:8000/api/v1`
**Documentation:** http://localhost:8000/api/v1/docs

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URL & Endpoints](#base-url--endpoints)
4. [Rate Limits](#rate-limits)
5. [Request/Response Format](#requestresponse-format)
6. [Error Handling](#error-handling)
7. [Endpoints Reference](#endpoints-reference)
   - [Health Check](#health-check)
   - [Dictionary Management](#dictionary-management)
   - [Program Management](#program-management)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)
10. [Code Examples](#code-examples)

---

## Overview

The Data Analyzer REST API provides programmatic access to data quality analysis and validation program management. Key features include:

- **Data Quality Analysis**: Upload CSV, JSON, Excel, or Parquet files for quality checks
- **Dictionary Parsing**: Extract schemas and validation rules from data dictionaries using AI
- **Logic Validation**: Execute conditional validation rules (e.g., "if gender=male, skip pregnancy questions")
- **Program Management**: Save, retrieve, and reuse validation programs with user-friendly aliases
- **Rate Limiting**: Built-in protection against abuse
- **Comprehensive Error Handling**: Detailed error messages for debugging

### Key Capabilities

- Parse data dictionaries in multiple formats (REDCap CSV, FHIR JSON, generic formats)
- Generate Python validation code from dictionary specifications
- Cache validation programs for repeated use
- Create user-friendly aliases (e.g., "johnDoesFav01") for programs
- Track program usage statistics
- Admin-controlled program management

---

## Authentication

The API uses header-based authentication with two levels:

### API Key Authentication

**Purpose:** Protect regular API endpoints
**Header:** `X-API-Key`
**Configuration:** Set `DATA_ANALYZER_API_KEY` environment variable

**Example:**
```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/api/v1/dictionary/parse
```

### Admin Password Authentication

**Purpose:** Protect administrative operations (delete programs, system config)
**Header:** `X-Admin-Password`
**Configuration:** Set `DATA_ANALYZER_ADMIN_PASSWORD` environment variable

**Example:**
```bash
curl -X DELETE \
  -H "X-Admin-Password: your-admin-password" \
  http://localhost:8000/api/v1/programs/abc123
```

### Development Mode

If authentication variables are not set, the API runs in **development mode** with authentication disabled. This is for local development only.

**Security Warning:** Always enable authentication in production environments.

### Generating Secure Credentials

```bash
# Generate API Key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate Admin Password
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

---

## Base URL & Endpoints

### Development
```
http://localhost:8000/api/v1
```

### Production
```
https://your-domain.com/api/v1
```

### Available Endpoints

| Endpoint | Method | Auth | Rate Limit | Description |
|----------|--------|------|------------|-------------|
| `/health` | GET | No | 60/min | Health check |
| `/dictionary/parse` | POST | API Key | 5/min | Parse dictionary |
| `/dictionary/{dict_id}` | GET | API Key | 30/min | Get dictionary details |
| `/programs` | GET | API Key | 30/min | List programs *(planned)* |
| `/programs/{id}` | GET | API Key | 30/min | Get program details *(planned)* |
| `/programs/{id}/alias` | POST | API Key | 10/min | Create alias *(planned)* |
| `/programs/{id}` | DELETE | Admin | 10/min | Delete program *(planned)* |
| `/programs/{id}/restore` | POST | Admin | 10/min | Restore program *(planned)* |
| `/analyze` | POST | API Key | 10/min | Analyze data *(planned)* |
| `/analyze/with-program` | POST | API Key | 10/min | Analyze with program *(planned)* |

---

## Rate Limits

Rate limits are enforced per IP address and returned in response headers:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1701552000
```

### Rate Limit Guidelines

| Endpoint Category | Limit | Reason |
|-------------------|-------|--------|
| Health check | 60/min | Monitoring systems |
| Dictionary parsing | 5/min | LLM calls are expensive |
| Dictionary retrieval | 30/min | Database reads |
| Data analysis | 10/min | CPU-intensive operations |
| Admin operations | 10/min | Sensitive operations |

### Rate Limit Exceeded

**Status Code:** `429 Too Many Requests`

**Response:**
```json
{
  "error": "Rate limit exceeded",
  "detail": "60 per 1 minute",
  "timestamp": "2025-12-02T22:30:00.123456"
}
```

**Solution:** Wait for the rate limit window to reset (check `X-RateLimit-Reset` header).

---

## Request/Response Format

### Request Headers

```http
Content-Type: application/json
X-API-Key: your-api-key
X-Request-ID: optional-tracking-id
```

For file uploads:
```http
Content-Type: multipart/form-data
X-API-Key: your-api-key
```

### Response Format

All responses are JSON with consistent structure:

**Success Response:**
```json
{
  "program_id": "uuid",
  "program_name": "20241202-143022-ClinicalTrial",
  "fields_extracted": 45,
  ...
}
```

**Error Response:**
```json
{
  "error": "Error message",
  "detail": "Detailed information",
  "timestamp": "2025-12-02T22:30:00.123456",
  "request_id": "optional-id"
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Invalid input, unsupported format |
| 401 | Unauthorized | Missing authentication credentials |
| 403 | Forbidden | Invalid credentials |
| 404 | Not Found | Resource doesn't exist |
| 413 | Payload Too Large | File exceeds size limits |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error |
| 501 | Not Implemented | Feature not yet available |
| 503 | Service Unavailable | Service temporarily unavailable |

### Error Response Schema

```json
{
  "error": "string",           // High-level error message
  "detail": "string | null",   // Detailed error information
  "timestamp": "string",        // ISO 8601 timestamp
  "request_id": "string | null" // Optional tracking ID
}
```

### Common Error Scenarios

#### Missing Authentication
```json
{
  "error": "API key authentication required",
  "detail": "Provide valid credentials in X-API-Key header",
  "timestamp": "2025-12-02T22:30:00.123456"
}
```

#### Invalid File Format
```json
{
  "error": "Unsupported file format: .xlsx",
  "detail": "Supported formats: .pdf, .csv, .json, .txt",
  "timestamp": "2025-12-02T22:30:00.123456"
}
```

#### Program Not Found
```json
{
  "error": "Program not found",
  "detail": "No program found with identifier: abc123",
  "timestamp": "2025-12-02T22:30:00.123456"
}
```

#### LLM Service Unavailable
```json
{
  "error": "Dictionary parsing service not available",
  "detail": "LLM client may not be configured. Check AZURE_OPENAI_* environment variables",
  "timestamp": "2025-12-02T22:30:00.123456"
}
```

---

## Endpoints Reference

### Health Check

**Endpoint:** `GET /api/v1/health`
**Authentication:** None
**Rate Limit:** 60 requests/minute

#### Description
Check API health and service availability. Use this endpoint for monitoring and readiness probes.

#### Request
```bash
curl http://localhost:8000/api/v1/health
```

#### Response (200 OK)
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

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Overall status: "healthy" or "degraded" |
| `version` | string | API version |
| `timestamp` | string | Current server time (ISO 8601) |
| `services` | object | Status of each service component |

#### Service Status Details

- **llm_client**: Azure OpenAI client for dictionary parsing
- **program_manager**: Program database and management
- **logic_validator**: Conditional validation engine
- **mcp_server**: Data quality analysis pipeline

---

### Dictionary Management

### Parse Dictionary

**Endpoint:** `POST /api/v1/dictionary/parse`
**Authentication:** API Key required
**Rate Limit:** 5 requests/minute (LLM calls are expensive)

#### Description
Parse a data dictionary file using AI to extract field definitions, validation rules, and conditional logic. Optionally save the generated validation program for reuse.

#### Request

**Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `dictionary_file` | file | Yes | Dictionary file (PDF, CSV, JSON, TXT) |
| `save_program` | boolean | No | Save as cached program (default: true) |
| `program_name` | string | No | Custom name override (auto-generated if not provided) |

**Supported Formats:**
- **CSV**: REDCap data dictionaries, custom CSV formats
- **JSON**: FHIR questionnaires, custom JSON schemas
- **TXT**: Plain text dictionary descriptions
- **PDF**: Not yet implemented (returns 501)

**File Size Limits:**
- Dictionary files: 10 MB maximum

#### Example Request (cURL)
```bash
curl -X POST http://localhost:8000/api/v1/dictionary/parse \
  -H "X-API-Key: your-api-key" \
  -F "dictionary_file=@clinical_trial_dict.csv" \
  -F "save_program=true" \
  -F "program_name=ClinicalTrialV1"
```

#### Example Request (Python)
```python
import requests

url = "http://localhost:8000/api/v1/dictionary/parse"
headers = {"X-API-Key": "your-api-key"}

files = {
    "dictionary_file": open("clinical_trial_dict.csv", "rb")
}

data = {
    "save_program": True,
    "program_name": "ClinicalTrialV1"
}

response = requests.post(url, headers=headers, files=files, data=data)
print(response.json())
```

#### Response (200 OK)
```json
{
  "program_id": "550e8400-e29b-41d4-a716-446655440000",
  "program_name": "20241202-143022-ClinicalTrial",
  "fields_extracted": 45,
  "rules_extracted": 23,
  "logic_rules_extracted": 8,
  "generated_code": "def validate_logic_rules(df: pd.DataFrame)...",
  "schema": {
    "patient_id": {
      "field_type": "integer",
      "required": true,
      "description": "Unique patient identifier"
    },
    "gender": {
      "field_type": "categorical",
      "required": true,
      "allowed_values": ["male", "female", "other"],
      "description": "Patient gender"
    }
  },
  "dictionary_format": "redcap_csv",
  "generation_time_seconds": 12.5,
  "model_used": "gpt-4o"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `program_id` | string | Unique program identifier (UUID) |
| `program_name` | string | Program name (auto-generated or custom) |
| `fields_extracted` | integer | Number of fields found in dictionary |
| `rules_extracted` | integer | Number of validation rules extracted |
| `logic_rules_extracted` | integer | Number of conditional logic rules found |
| `generated_code` | string | Python validation code (truncated if > 500 chars) |
| `schema` | object | Field definitions with types and constraints |
| `dictionary_format` | string | Detected dictionary format |
| `generation_time_seconds` | number | Time taken to parse dictionary |
| `model_used` | string | LLM model used for parsing |

#### Error Responses

**400 Bad Request - Unsupported Format:**
```json
{
  "error": "Unsupported file format: .xlsx",
  "detail": "Supported formats: .pdf, .csv, .json, .txt",
  "timestamp": "2025-12-02T22:30:00.123456"
}
```

**501 Not Implemented - PDF Parsing:**
```json
{
  "error": "PDF parsing not yet implemented",
  "detail": "Please convert to CSV or JSON",
  "timestamp": "2025-12-02T22:30:00.123456"
}
```

**503 Service Unavailable - LLM Client:**
```json
{
  "error": "Dictionary parsing service not available",
  "detail": "LLM client may not be configured",
  "timestamp": "2025-12-02T22:30:00.123456"
}
```

**500 Internal Server Error - Parsing Failed:**
```json
{
  "error": "Dictionary parsing failed",
  "detail": "Unable to extract field definitions from dictionary",
  "timestamp": "2025-12-02T22:30:00.123456"
}
```

---

### Get Dictionary

**Endpoint:** `GET /api/v1/dictionary/{dict_id}`
**Authentication:** API Key required
**Rate Limit:** 30 requests/minute

#### Description
Retrieve a saved validation program (parsed dictionary) by ID, name, or alias.

#### Request

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `dict_id` | string | Program ID (UUID), program name, or alias |

#### Example Requests

**By Program ID:**
```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/api/v1/dictionary/550e8400-e29b-41d4-a716-446655440000
```

**By Program Name:**
```bash
curl -H "X-API-Key: your-api-key" \
  "http://localhost:8000/api/v1/dictionary/20241202-143022-ClinicalTrial"
```

**By Alias:**
```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/api/v1/dictionary/johnDoesFav01
```

#### Response (200 OK)
```json
{
  "program_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "20241202-143022-ClinicalTrial",
  "aliases": ["johnDoesFav01", "clinicalV1"],
  "dictionary_source": "clinical_trial_dict.csv",
  "dictionary_format": "redcap_csv",
  "dictionary_hash": "sha256:abc123...",
  "schema": {
    "patient_id": {
      "field_type": "integer",
      "required": true,
      "description": "Unique patient identifier"
    },
    "gender": {
      "field_type": "categorical",
      "required": true,
      "allowed_values": ["male", "female", "other"],
      "description": "Patient gender"
    }
  },
  "rules": [
    {
      "rule_id": "rule_001",
      "rule_type": "skip_if",
      "condition": "gender == 'male'",
      "action": "skip",
      "affected_fields": ["pregnancy_status"],
      "description": "Skip pregnancy questions for male patients",
      "severity": "error"
    }
  ],
  "generated_code": "def validate_logic_rules(df: pd.DataFrame):\n    violations = []\n    ...\n    return violations",
  "num_fields": 45,
  "num_basic_rules": 23,
  "num_logic_rules": 8,
  "metadata": {
    "description": "Clinical trial validation program",
    "version": "1.0"
  },
  "created_by": "api_user",
  "created_at": "2024-12-02T14:30:22Z",
  "last_used": "2024-12-02T15:45:10Z",
  "use_count": 15,
  "status": "active",
  "model_used": "gpt-4o"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `program_id` | string | Unique program identifier |
| `name` | string | Program name |
| `aliases` | array | User-friendly aliases |
| `dictionary_source` | string | Original dictionary filename |
| `dictionary_format` | string | Detected format |
| `dictionary_hash` | string | SHA-256 hash of dictionary |
| `schema` | object | Field definitions |
| `rules` | array | Conditional validation rules |
| `generated_code` | string | Python validation function |
| `num_fields` | integer | Number of fields |
| `num_basic_rules` | integer | Number of basic rules |
| `num_logic_rules` | integer | Number of conditional rules |
| `metadata` | object | Additional program metadata |
| `created_by` | string | Creator username/ID |
| `created_at` | string | Creation timestamp |
| `last_used` | string | Last usage timestamp |
| `use_count` | integer | Number of times used |
| `status` | string | "active" or "deleted" |
| `model_used` | string | LLM model used |

#### Error Responses

**404 Not Found:**
```json
{
  "error": "Program not found",
  "detail": "No program found with identifier: abc123",
  "timestamp": "2025-12-02T22:30:00.123456"
}
```

**404 Not Found - Deleted Program:**
```json
{
  "error": "Program has been deleted",
  "detail": "Program abc123 was deleted by administrator",
  "timestamp": "2025-12-02T22:30:00.123456"
}
```

**503 Service Unavailable:**
```json
{
  "error": "Program management service not available",
  "detail": "Database connection failed",
  "timestamp": "2025-12-02T22:30:00.123456"
}
```

---

### Program Management

*The following endpoints are planned for future implementation (api_5):*

#### List Programs

**Endpoint:** `GET /api/v1/programs`
**Status:** Planned

List all validation programs with optional search and filters.

**Query Parameters:**
- `search`: Text search in name, source, description
- `dictionary_source`: Filter by source file
- `created_by`: Filter by creator
- `status`: "active", "deleted", or "all"
- `limit`: Pagination limit (default: 50)
- `offset`: Pagination offset (default: 0)

#### Get Program Details

**Endpoint:** `GET /api/v1/programs/{id_or_alias}`
**Status:** Planned

Get detailed program information by ID or alias.

#### Create Alias

**Endpoint:** `POST /api/v1/programs/{id}/alias`
**Status:** Planned

Create a user-friendly alias for a program.

**Request Body:**
```json
{
  "alias": "johnDoesFav01"
}
```

#### Delete Program

**Endpoint:** `DELETE /api/v1/programs/{id}`
**Status:** Planned
**Authentication:** Admin password required

Soft delete a validation program (admin only).

**Request Body:**
```json
{
  "reason": "Contains errors in validation logic"
}
```

#### Restore Program

**Endpoint:** `POST /api/v1/programs/{id}/restore`
**Status:** Planned
**Authentication:** Admin password required

Restore a deleted program (admin only).

---

## Best Practices

### 1. Authentication

- **Never commit credentials** to version control
- Use environment variables for API keys
- **Rotate credentials regularly** (every 90 days)
- Generate cryptographically secure random keys
- Use different credentials for development and production

### 2. Rate Limiting

- **Implement retry logic** with exponential backoff
- **Cache responses** when possible
- Monitor `X-RateLimit-Remaining` header
- For high-volume usage, contact administrators for increased limits

### 3. Error Handling

- **Check HTTP status codes** before parsing response
- **Log errors** with request_id for debugging
- Implement graceful degradation for service unavailability
- Display user-friendly error messages

### 4. File Uploads

- **Validate file sizes** before upload
- **Check file extensions** match content
- Use streaming for large files
- Clean up temporary files after processing

### 5. Program Management

- **Use aliases** for frequently used programs
- **Track program usage** to identify popular validations
- **Test validation programs** with sample data before production use
- **Document program purpose** in metadata

### 6. Performance

- **Reuse validation programs** instead of re-parsing dictionaries
- **Batch operations** when possible
- **Use pagination** for large result sets
- **Monitor generation times** for dictionary parsing

### 7. Security

- **Use HTTPS** in production
- **Validate all inputs** before sending to API
- **Don't log sensitive data** (PII, credentials)
- **Implement request timeouts** to prevent hanging connections
- **Use unique request IDs** for tracing

---

## Troubleshooting

### Common Issues

#### 1. "Address already in use" Error

**Problem:** Port 8000 is already in use.

**Solutions:**
```bash
# Option 1: Kill existing process
lsof -ti:8000 | xargs kill -9

# Option 2: Use different port
API_PORT=8001 python api_server.py

# Option 3: Use uvicorn with custom port
uvicorn api_server:app --port 8001
```

#### 2. LLM Client Not Configured

**Problem:** Azure OpenAI credentials missing.

**Solution:**
```bash
# Add to .env file
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# Restart server
python api_server.py
```

#### 3. Rate Limit Exceeded

**Problem:** Too many requests in short time.

**Solution:**
```python
import time
import requests

def make_request_with_retry(url, headers, max_retries=3):
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)

        if response.status_code == 429:
            # Get retry-after from headers
            retry_after = int(response.headers.get('Retry-After', 60))
            print(f"Rate limited. Waiting {retry_after} seconds...")
            time.sleep(retry_after)
            continue

        return response

    raise Exception("Max retries exceeded")
```

#### 4. Dictionary Parsing Failed

**Problem:** LLM unable to extract fields from dictionary.

**Possible Causes:**
- Dictionary format not recognized
- Insufficient information in dictionary
- LLM timeout or error

**Solutions:**
1. **Check dictionary format** - Ensure it follows REDCap, FHIR, or recognizable patterns
2. **Add more context** - Include field descriptions and examples
3. **Use standard formats** - REDCap CSV and FHIR JSON are best supported
4. **Check logs** - Review API server logs for detailed errors

#### 5. Program Not Found

**Problem:** Cannot retrieve saved program.

**Solutions:**
```bash
# Check program exists in database
sqlite3 ~/.data_analyzer/programs.db "SELECT program_id, name FROM programs LIMIT 10;"

# Check for typos in name/alias
# Names are case-sensitive: "ClinicalTrial" ≠ "clinicaltrial"

# Use program ID (UUID) instead of name for reliability
```

#### 6. File Upload Failed

**Problem:** File upload returns 400 or 413 error.

**Solutions:**
- **Check file size** (max 10MB for dictionaries)
- **Verify file extension** (.pdf, .csv, .json, .txt)
- **Check file encoding** (UTF-8 recommended)
- **Ensure file is not corrupted**

---

## Code Examples

### Python with Requests

```python
import requests
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
API_KEY = "your-api-key"

headers = {
    "X-API-Key": API_KEY
}

# 1. Health check
response = requests.get(f"{API_BASE_URL}/health")
print("Health:", response.json())

# 2. Parse dictionary
with open("clinical_trial_dict.csv", "rb") as f:
    files = {"dictionary_file": f}
    data = {
        "save_program": True,
        "program_name": "ClinicalTrialV1"
    }

    response = requests.post(
        f"{API_BASE_URL}/dictionary/parse",
        headers=headers,
        files=files,
        data=data
    )

    result = response.json()
    program_id = result["program_id"]
    print(f"Created program: {program_id}")

# 3. Get dictionary details
response = requests.get(
    f"{API_BASE_URL}/dictionary/{program_id}",
    headers=headers
)

program = response.json()
print(f"Program: {program['name']}")
print(f"Fields: {program['num_fields']}")
print(f"Logic rules: {program['num_logic_rules']}")
```

### JavaScript with Fetch

```javascript
const API_BASE_URL = 'http://localhost:8000/api/v1';
const API_KEY = 'your-api-key';

const headers = {
  'X-API-Key': API_KEY
};

// 1. Health check
async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  const data = await response.json();
  console.log('Health:', data);
}

// 2. Parse dictionary
async function parseDictionary(file) {
  const formData = new FormData();
  formData.append('dictionary_file', file);
  formData.append('save_program', 'true');
  formData.append('program_name', 'ClinicalTrialV1');

  const response = await fetch(`${API_BASE_URL}/dictionary/parse`, {
    method: 'POST',
    headers: headers,
    body: formData
  });

  const result = await response.json();
  console.log('Program created:', result.program_id);
  return result.program_id;
}

// 3. Get dictionary details
async function getDictionary(programId) {
  const response = await fetch(
    `${API_BASE_URL}/dictionary/${programId}`,
    { headers: headers }
  );

  const program = await response.json();
  console.log(`Program: ${program.name}`);
  console.log(`Fields: ${program.num_fields}`);
  return program;
}

// Usage
await checkHealth();
const programId = await parseDictionary(fileInput.files[0]);
const program = await getDictionary(programId);
```

### cURL Examples

```bash
# Set variables
API_BASE_URL="http://localhost:8000/api/v1"
API_KEY="your-api-key"

# 1. Health check
curl -s ${API_BASE_URL}/health | python -m json.tool

# 2. Parse dictionary
curl -X POST ${API_BASE_URL}/dictionary/parse \
  -H "X-API-Key: ${API_KEY}" \
  -F "dictionary_file=@clinical_trial_dict.csv" \
  -F "save_program=true" \
  -F "program_name=ClinicalTrialV1" \
  | python -m json.tool

# 3. Get dictionary details
PROGRAM_ID="550e8400-e29b-41d4-a716-446655440000"
curl -s ${API_BASE_URL}/dictionary/${PROGRAM_ID} \
  -H "X-API-Key: ${API_KEY}" \
  | python -m json.tool

# 4. Get dictionary by name (URL encode spaces)
PROGRAM_NAME="20241202-143022-ClinicalTrial"
curl -s "${API_BASE_URL}/dictionary/${PROGRAM_NAME}" \
  -H "X-API-Key: ${API_KEY}" \
  | python -m json.tool

# 5. Get dictionary by alias
curl -s ${API_BASE_URL}/dictionary/johnDoesFav01 \
  -H "X-API-Key: ${API_KEY}" \
  | python -m json.tool
```

---

## Additional Resources

- **Interactive API Documentation**: http://localhost:8000/api/v1/docs
- **Authentication Guide**: [docs/AUTHENTICATION.md](./AUTHENTICATION.md)
- **Quick Start Guide**: [docs/API_QUICK_START.md](./API_QUICK_START.md)
- **Code Examples**: [docs/API_EXAMPLES.md](./API_EXAMPLES.md)
- **Postman Collection**: [postman_collection.json](../postman_collection.json)

---

## Changelog

### Version 1.0.0 (2025-12-02)

**Implemented:**
- Health check endpoint
- Dictionary parsing endpoint
- Dictionary retrieval endpoint
- API key authentication
- Rate limiting
- Comprehensive error handling
- OpenAPI documentation

**Planned:**
- Program management endpoints (list, create alias, delete, restore)
- Data analysis endpoints (analyze, analyze with program)
- Admin operations
- Webhooks for program updates
- Batch operations

---

## Support

For issues, questions, or feature requests:

1. Check the [Troubleshooting](#troubleshooting) section
2. Review the [Interactive API Docs](http://localhost:8000/api/v1/docs)
3. Check `developer_checklist.yaml` for implementation status
4. Review server logs: `/tmp/api_server.log`

---

**Last Updated:** 2025-12-02
**API Version:** 1.0.0
**Documentation Version:** 1.0.0
