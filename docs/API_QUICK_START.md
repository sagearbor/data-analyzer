# Data Analyzer API - Quick Start Guide

Get up and running with the Data Analyzer REST API in 5 minutes.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Get Your API Key](#get-your-api-key)
4. [Make Your First Request](#make-your-first-request)
5. [Common Workflows](#common-workflows)
6. [Next Steps](#next-steps)

---

## Prerequisites

- Python 3.10+ installed
- Basic familiarity with REST APIs
- A data dictionary file (CSV, JSON, or TXT format)
- Optional: Postman or cURL for testing

---

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd data-analyzer
```

### 2. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install API dependencies
pip install -r requirements.txt -r api_requirements.txt
```

### 3. Configure Environment

Create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and add Azure OpenAI credentials:

```bash
# Azure OpenAI Configuration (required for dictionary parsing)
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# API Authentication (optional for development)
DATA_ANALYZER_API_KEY=dev-test-key-12345
DATA_ANALYZER_ADMIN_PASSWORD=admin-password-67890

# App Configuration
APP_ENV=dev
API_HOST=0.0.0.0
API_PORT=8000
```

### 4. Start the API Server

```bash
# Option 1: Direct Python
python api_server.py

# Option 2: Using uvicorn
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:api_server:Data Analyzer REST API starting up
INFO:api_server:Version: 1.0.0
```

---

## Get Your API Key

### Development Mode (Quick Testing)

If you didn't set `DATA_ANALYZER_API_KEY` in `.env`, the API runs in **development mode** with authentication disabled. This is fine for local testing.

### Production Mode (Secure)

Generate a secure API key:

```bash
python -c "import secrets; print('API Key:', secrets.token_urlsafe(32))"
```

Add to `.env`:

```bash
DATA_ANALYZER_API_KEY=your-generated-key-here
```

Restart the server:

```bash
python api_server.py
```

---

## Make Your First Request

### Test 1: Health Check (No Auth Required)

```bash
curl http://localhost:8000/api/v1/health
```

**Expected Response:**

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

✅ **Success!** Your API is running.

### Test 2: Parse a Dictionary (Auth Required)

Create a simple test dictionary (`test_dict.csv`):

```csv
field_name,field_type,required,description
patient_id,integer,yes,Unique patient identifier
age,integer,yes,Patient age in years
gender,categorical,yes,Patient gender (male/female/other)
```

Parse it:

```bash
curl -X POST http://localhost:8000/api/v1/dictionary/parse \
  -H "X-API-Key: your-api-key" \
  -F "dictionary_file=@test_dict.csv" \
  -F "save_program=true"
```

**Expected Response:**

```json
{
  "program_id": "550e8400-e29b-41d4-a716-446655440000",
  "program_name": "20241202-143022-PatientData",
  "fields_extracted": 3,
  "rules_extracted": 2,
  "logic_rules_extracted": 0,
  "generated_code": "def validate_logic_rules(df: pd.DataFrame)...",
  "schema": {
    "patient_id": {
      "field_type": "integer",
      "required": true,
      "description": "Unique patient identifier"
    },
    ...
  },
  "dictionary_format": "generic_csv",
  "generation_time_seconds": 8.2,
  "model_used": "gpt-4o"
}
```

✅ **Success!** Your dictionary has been parsed and saved.

### Test 3: Retrieve the Program

Copy the `program_id` from the response above and use it:

```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/api/v1/dictionary/550e8400-e29b-41d4-a716-446655440000
```

Or use the program name:

```bash
curl -H "X-API-Key: your-api-key" \
  "http://localhost:8000/api/v1/dictionary/20241202-143022-PatientData"
```

**Expected Response:**

```json
{
  "program_id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "20241202-143022-PatientData",
  "aliases": [],
  "dictionary_source": "test_dict.csv",
  "schema": { ... },
  "rules": [ ... ],
  "generated_code": "def validate_logic_rules(df: pd.DataFrame)...",
  "num_fields": 3,
  "num_basic_rules": 2,
  "num_logic_rules": 0,
  "created_at": "2024-12-02T14:30:22Z",
  "last_used": "2024-12-02T14:30:22Z",
  "use_count": 1,
  "status": "active"
}
```

✅ **Success!** You've completed the basic workflow.

---

## Common Workflows

### Workflow 1: Parse Dictionary and Save

**Use Case:** You have a data dictionary and want to create a reusable validation program.

```bash
# Step 1: Parse dictionary
curl -X POST http://localhost:8000/api/v1/dictionary/parse \
  -H "X-API-Key: your-api-key" \
  -F "dictionary_file=@my_dictionary.csv" \
  -F "save_program=true" \
  -F "program_name=MyValidationProgram"

# Step 2: Note the program_id from response
# Example: 550e8400-e29b-41d4-a716-446655440000

# Step 3: Retrieve program details
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/api/v1/dictionary/550e8400-e29b-41d4-a716-446655440000
```

### Workflow 2: Find and Reuse Existing Program

**Use Case:** You want to use a previously saved validation program.

```bash
# Option 1: Search by name
curl -H "X-API-Key: your-api-key" \
  "http://localhost:8000/api/v1/dictionary/MyValidationProgram"

# Option 2: Search by alias (if you created one)
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/api/v1/dictionary/myAlias01

# Option 3: List all programs (planned feature)
# curl -H "X-API-Key: your-api-key" \
#   "http://localhost:8000/api/v1/programs?search=validation"
```

### Workflow 3: Test with Postman

**Use Case:** You prefer a GUI for testing APIs.

1. Import the Postman collection:
   - Open Postman
   - Click "Import" → "File"
   - Select `postman_collection.json`

2. Set environment variables:
   - BASE_URL: `http://localhost:8000/api/v1`
   - API_KEY: `your-api-key`

3. Run the "Health Check" request to verify setup

4. Run "Parse Dictionary" with your test file

5. Run "Get Dictionary by ID" to retrieve results

### Workflow 4: Python Script Integration

**Use Case:** Automate dictionary parsing in a Python script.

```python
import requests

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
API_KEY = "your-api-key"

def parse_dictionary(file_path, program_name=None):
    """Parse a dictionary and save as program"""
    url = f"{API_BASE_URL}/dictionary/parse"
    headers = {"X-API-Key": API_KEY}

    with open(file_path, "rb") as f:
        files = {"dictionary_file": f}
        data = {"save_program": True}

        if program_name:
            data["program_name"] = program_name

        response = requests.post(url, headers=headers, files=files, data=data)
        response.raise_for_status()
        return response.json()

def get_program(program_id_or_name):
    """Retrieve program details"""
    url = f"{API_BASE_URL}/dictionary/{program_id_or_name}"
    headers = {"X-API-Key": API_KEY}

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

# Usage
result = parse_dictionary("my_dictionary.csv", "MyProgram")
print(f"Created program: {result['program_id']}")

program = get_program(result['program_id'])
print(f"Fields: {program['num_fields']}")
print(f"Logic rules: {program['num_logic_rules']}")
```

---

## Next Steps

### Explore Interactive Documentation

Visit the auto-generated Swagger UI:

```
http://localhost:8000/api/v1/docs
```

This provides:
- Interactive API testing
- Request/response schemas
- Example requests
- Authentication setup

### Read Comprehensive Documentation

- **Full API Reference**: [docs/API_COMPREHENSIVE.md](./API_COMPREHENSIVE.md)
- **Code Examples**: [docs/API_EXAMPLES.md](./API_EXAMPLES.md)
- **Authentication Guide**: [docs/AUTHENTICATION.md](./AUTHENTICATION.md)

### Try Advanced Features

1. **Conditional Logic Rules**
   - Upload a REDCap dictionary with branching logic
   - See conditional rules extracted automatically

2. **Program Aliases** (planned)
   - Create user-friendly aliases like "clinicalV1"
   - Access programs by alias instead of UUID

3. **Data Analysis** (planned)
   - Upload data files for quality analysis
   - Use saved programs to validate data
   - Get detailed validation reports

### Monitor API Health

Set up monitoring using the health endpoint:

```bash
# Simple health check script
while true; do
  curl -s http://localhost:8000/api/v1/health | python -m json.tool
  sleep 60
done
```

### Join the Development

Check `developer_checklist.yaml` for:
- Current implementation status
- Planned features
- Known issues

---

## Troubleshooting

### Issue: "Connection refused"

**Cause:** API server not running

**Solution:**
```bash
python api_server.py
```

### Issue: "LLM client may not be configured"

**Cause:** Missing Azure OpenAI credentials

**Solution:**
```bash
# Add to .env
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o

# Restart server
python api_server.py
```

### Issue: "Rate limit exceeded"

**Cause:** Too many requests in short time

**Solution:**
- Wait 60 seconds for rate limit to reset
- Check `X-RateLimit-Reset` header for exact time
- Implement retry logic with exponential backoff

### Issue: "Invalid credentials provided"

**Cause:** Wrong API key

**Solution:**
- Check `.env` file for correct `DATA_ANALYZER_API_KEY`
- Ensure API key matches in requests
- Restart server after changing `.env`

---

## Quick Reference

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication Header
```bash
-H "X-API-Key: your-api-key"
```

### Key Endpoints
```bash
# Health check (no auth)
GET /health

# Parse dictionary (auth required)
POST /dictionary/parse

# Get dictionary (auth required)
GET /dictionary/{id_or_name_or_alias}
```

### Rate Limits
- Health: 60/minute
- Parse: 5/minute
- Get: 30/minute

### File Size Limits
- Dictionary files: 10 MB
- Data files: 50 MB

---

## Support

- **Interactive Docs**: http://localhost:8000/api/v1/docs
- **Implementation Status**: Check `developer_checklist.yaml`
- **Logs**: `/tmp/api_server.log`

---

**Last Updated:** 2025-12-02
**API Version:** 1.0.0
