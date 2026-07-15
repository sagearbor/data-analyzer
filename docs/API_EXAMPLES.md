# Data Analyzer API - Code Examples

Comprehensive code examples for all common operations in multiple programming languages.

## Table of Contents

1. [Python Examples](#python-examples)
2. [JavaScript Examples](#javascript-examples)
3. [cURL Examples](#curl-examples)
4. [Error Handling Examples](#error-handling-examples)
5. [Advanced Usage](#advanced-usage)

---

## Python Examples

### Setup and Configuration

```python
import requests
import json
from pathlib import Path
from typing import Dict, Any, Optional
import time

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
API_KEY = "your-api-key"
ADMIN_PASSWORD = "your-admin-password"

# Headers
headers_api = {"X-API-Key": API_KEY}
headers_admin = {"X-Admin-Password": ADMIN_PASSWORD}
```

### Example 1: Health Check

```python
def check_health() -> Dict[str, Any]:
    """Check API health and service availability"""
    url = f"{API_BASE_URL}/health"

    response = requests.get(url)
    response.raise_for_status()

    data = response.json()
    print(f"API Status: {data['status']}")
    print(f"Version: {data['version']}")
    print(f"Services: {json.dumps(data['services'], indent=2)}")

    return data

# Usage
health = check_health()
```

**Output:**
```
API Status: healthy
Version: 1.0.0
Services: {
  "llm_client": true,
  "program_manager": true,
  "logic_validator": true,
  "mcp_server": true
}
```

### Example 2: Parse Dictionary and Save Program

```python
def parse_dictionary(
    file_path: str,
    save_program: bool = True,
    program_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Parse a data dictionary file and optionally save as program

    Args:
        file_path: Path to dictionary file (CSV, JSON, TXT)
        save_program: Whether to save as cached program
        program_name: Optional custom name override

    Returns:
        Dictionary with program details
    """
    url = f"{API_BASE_URL}/dictionary/parse"

    # Prepare file
    with open(file_path, "rb") as f:
        files = {"dictionary_file": f}

        # Prepare form data
        data = {"save_program": str(save_program).lower()}
        if program_name:
            data["program_name"] = program_name

        # Make request
        response = requests.post(url, headers=headers_api, files=files, data=data)
        response.raise_for_status()

    result = response.json()

    print(f"Program ID: {result['program_id']}")
    print(f"Program Name: {result['program_name']}")
    print(f"Fields Extracted: {result['fields_extracted']}")
    print(f"Basic Rules: {result['rules_extracted']}")
    print(f"Logic Rules: {result['logic_rules_extracted']}")
    print(f"Generation Time: {result['generation_time_seconds']:.2f}s")

    return result

# Usage
result = parse_dictionary(
    file_path="clinical_trial_dict.csv",
    save_program=True,
    program_name="ClinicalTrialV1"
)

program_id = result["program_id"]
```

**Output:**
```
Program ID: 550e8400-e29b-41d4-a716-446655440000
Program Name: 20241202-143022-ClinicalTrial
Fields Extracted: 45
Basic Rules: 23
Logic Rules: 8
Generation Time: 12.45s
```

### Example 3: Get Program Details

```python
def get_program(identifier: str) -> Dict[str, Any]:
    """
    Retrieve program by ID, name, or alias

    Args:
        identifier: Program ID (UUID), name, or alias

    Returns:
        Complete program details
    """
    url = f"{API_BASE_URL}/dictionary/{identifier}"

    response = requests.get(url, headers=headers_api)
    response.raise_for_status()

    program = response.json()

    print(f"Program: {program['name']}")
    print(f"Source: {program['dictionary_source']}")
    print(f"Format: {program['dictionary_format']}")
    print(f"Fields: {program['num_fields']}")
    print(f"Logic Rules: {program['num_logic_rules']}")
    print(f"Created: {program['created_at']}")
    print(f"Last Used: {program['last_used']}")
    print(f"Use Count: {program['use_count']}")
    print(f"Status: {program['status']}")

    return program

# Usage - by ID
program = get_program("550e8400-e29b-41d4-a716-446655440000")

# Usage - by name
program = get_program("20241202-143022-ClinicalTrial")

# Usage - by alias
program = get_program("johnDoesFav01")
```

**Output:**
```
Program: 20241202-143022-ClinicalTrial
Source: clinical_trial_dict.csv
Format: redcap_csv
Fields: 45
Logic Rules: 8
Created: 2024-12-02T14:30:22Z
Last Used: 2024-12-02T15:45:10Z
Use Count: 15
Status: active
```

### Example 4: Complete Workflow with Error Handling

```python
def complete_workflow(dictionary_file: str, program_name: str):
    """
    Complete workflow: parse dictionary, save program, retrieve details

    Includes comprehensive error handling and logging
    """
    try:
        # Step 1: Check API health
        print("Step 1: Checking API health...")
        health = check_health()

        if health["status"] != "healthy":
            print("Warning: API is in degraded state")

        if not health["services"]["llm_client"]:
            print("Error: LLM client not available - dictionary parsing will fail")
            return None

        # Step 2: Parse dictionary
        print(f"\nStep 2: Parsing dictionary: {dictionary_file}")
        result = parse_dictionary(
            file_path=dictionary_file,
            save_program=True,
            program_name=program_name
        )

        program_id = result["program_id"]
        print(f"✓ Program created: {program_id}")

        # Step 3: Retrieve and verify program
        print(f"\nStep 3: Retrieving program details...")
        program = get_program(program_id)

        print(f"✓ Program retrieved: {program['name']}")

        # Step 4: Display schema
        print(f"\nStep 4: Program Schema:")
        for field_name, field_def in program["schema"].items():
            print(f"  - {field_name}: {field_def['field_type']}")
            if field_def.get("required"):
                print(f"      Required: Yes")
            if field_def.get("allowed_values"):
                print(f"      Allowed: {field_def['allowed_values']}")

        # Step 5: Display logic rules
        if program["rules"]:
            print(f"\nStep 5: Conditional Logic Rules:")
            for rule in program["rules"]:
                print(f"  - Rule: {rule['description']}")
                print(f"    Type: {rule['rule_type']}")
                print(f"    Condition: {rule['condition']}")
                print(f"    Affects: {', '.join(rule['affected_fields'])}")

        print(f"\n✓ Workflow complete!")
        return program

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e.response.status_code}")
        print(f"Details: {e.response.json()}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None

# Usage
program = complete_workflow(
    dictionary_file="clinical_trial_dict.csv",
    program_name="ClinicalTrialV1"
)
```

### Example 5: Batch Processing Multiple Dictionaries

```python
def batch_parse_dictionaries(dictionary_files: list) -> list:
    """
    Parse multiple dictionaries in batch

    Args:
        dictionary_files: List of file paths

    Returns:
        List of program results
    """
    results = []

    for i, file_path in enumerate(dictionary_files, 1):
        print(f"\nProcessing {i}/{len(dictionary_files)}: {file_path}")

        try:
            # Parse dictionary
            result = parse_dictionary(
                file_path=file_path,
                save_program=True
            )

            results.append({
                "file": file_path,
                "status": "success",
                "program_id": result["program_id"],
                "program_name": result["program_name"],
                "fields": result["fields_extracted"]
            })

            print(f"✓ Success: {result['program_name']}")

            # Rate limiting: Wait 12 seconds between requests (5/minute limit)
            if i < len(dictionary_files):
                print("Waiting 12 seconds (rate limit)...")
                time.sleep(12)

        except requests.exceptions.HTTPError as e:
            error_detail = e.response.json().get("detail", str(e))
            results.append({
                "file": file_path,
                "status": "error",
                "error": error_detail
            })
            print(f"✗ Error: {error_detail}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Batch Processing Summary")
    print(f"{'='*60}")
    print(f"Total: {len(results)}")
    print(f"Success: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'error')}")

    return results

# Usage
files = [
    "clinical_trial_dict.csv",
    "employee_records_dict.json",
    "survey_questions_dict.txt"
]

results = batch_parse_dictionaries(files)
```

---

## JavaScript Examples

### Setup and Configuration

```javascript
const API_BASE_URL = 'http://localhost:8000/api/v1';
const API_KEY = 'your-api-key';
const ADMIN_PASSWORD = 'your-admin-password';

const headersAPI = {
  'X-API-Key': API_KEY
};

const headersAdmin = {
  'X-Admin-Password': ADMIN_PASSWORD
};
```

### Example 1: Health Check

```javascript
async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  const data = await response.json();

  console.log(`API Status: ${data.status}`);
  console.log(`Version: ${data.version}`);
  console.log('Services:', JSON.stringify(data.services, null, 2));

  return data;
}

// Usage
checkHealth()
  .then(health => console.log('Health check complete'))
  .catch(error => console.error('Error:', error));
```

### Example 2: Parse Dictionary

```javascript
async function parseDictionary(file, saveProgram = true, programName = null) {
  const formData = new FormData();
  formData.append('dictionary_file', file);
  formData.append('save_program', saveProgram.toString());

  if (programName) {
    formData.append('program_name', programName);
  }

  const response = await fetch(`${API_BASE_URL}/dictionary/parse`, {
    method: 'POST',
    headers: headersAPI,
    body: formData
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Parse failed: ${error.detail}`);
  }

  const result = await response.json();

  console.log(`Program ID: ${result.program_id}`);
  console.log(`Program Name: ${result.program_name}`);
  console.log(`Fields Extracted: ${result.fields_extracted}`);
  console.log(`Logic Rules: ${result.logic_rules_extracted}`);

  return result;
}

// Usage with file input
document.getElementById('fileInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];

  if (file) {
    try {
      const result = await parseDictionary(file, true, 'MyProgram');
      console.log('Parse complete:', result);
    } catch (error) {
      console.error('Error:', error);
    }
  }
});
```

### Example 3: Get Program Details

```javascript
async function getProgram(identifier) {
  const response = await fetch(
    `${API_BASE_URL}/dictionary/${encodeURIComponent(identifier)}`,
    { headers: headersAPI }
  );

  if (!response.ok) {
    const error = await response.json();
    throw new Error(`Get program failed: ${error.detail}`);
  }

  const program = await response.json();

  console.log(`Program: ${program.name}`);
  console.log(`Source: ${program.dictionary_source}`);
  console.log(`Fields: ${program.num_fields}`);
  console.log(`Logic Rules: ${program.num_logic_rules}`);
  console.log(`Use Count: ${program.use_count}`);

  return program;
}

// Usage
getProgram('550e8400-e29b-41d4-a716-446655440000')
  .then(program => console.log('Program retrieved'))
  .catch(error => console.error('Error:', error));
```

### Example 4: Complete Workflow with React

```javascript
import React, { useState } from 'react';

function DictionaryParser() {
  const [file, setFile] = useState(null);
  const [programName, setProgramName] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setError(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file) {
      setError('Please select a file');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      // Parse dictionary
      const result = await parseDictionary(file, true, programName || null);
      setResult(result);

      // Retrieve program details
      const program = await getProgram(result.program_id);
      setResult(program);

    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Dictionary Parser</h2>

      <form onSubmit={handleSubmit}>
        <div>
          <label>
            Dictionary File:
            <input
              type="file"
              accept=".csv,.json,.txt"
              onChange={handleFileChange}
            />
          </label>
        </div>

        <div>
          <label>
            Program Name (optional):
            <input
              type="text"
              value={programName}
              onChange={(e) => setProgramName(e.target.value)}
              placeholder="e.g., ClinicalTrialV1"
            />
          </label>
        </div>

        <button type="submit" disabled={loading}>
          {loading ? 'Parsing...' : 'Parse Dictionary'}
        </button>
      </form>

      {error && (
        <div style={{ color: 'red' }}>
          Error: {error}
        </div>
      )}

      {result && (
        <div>
          <h3>Program Details</h3>
          <p><strong>Name:</strong> {result.name}</p>
          <p><strong>Fields:</strong> {result.num_fields}</p>
          <p><strong>Logic Rules:</strong> {result.num_logic_rules}</p>
          <p><strong>Use Count:</strong> {result.use_count}</p>

          <h4>Schema</h4>
          <ul>
            {Object.entries(result.schema).map(([name, def]) => (
              <li key={name}>
                {name}: {def.field_type}
                {def.required && ' (required)'}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default DictionaryParser;
```

---

## cURL Examples

### Example 1: Health Check

```bash
# Basic health check
curl -s http://localhost:8000/api/v1/health | python -m json.tool

# With formatted output using jq
curl -s http://localhost:8000/api/v1/health | jq '.'

# Check specific service
curl -s http://localhost:8000/api/v1/health | jq '.services.llm_client'
```

### Example 2: Parse Dictionary

```bash
# Set variables
API_BASE_URL="http://localhost:8000/api/v1"
API_KEY="your-api-key"
DICT_FILE="clinical_trial_dict.csv"

# Parse dictionary with default settings
curl -X POST "${API_BASE_URL}/dictionary/parse" \
  -H "X-API-Key: ${API_KEY}" \
  -F "dictionary_file=@${DICT_FILE}" \
  -F "save_program=true" \
  | python -m json.tool

# Parse with custom program name
curl -X POST "${API_BASE_URL}/dictionary/parse" \
  -H "X-API-Key: ${API_KEY}" \
  -F "dictionary_file=@${DICT_FILE}" \
  -F "save_program=true" \
  -F "program_name=ClinicalTrialV1" \
  | python -m json.tool

# Save program_id for later use
PROGRAM_ID=$(curl -s -X POST "${API_BASE_URL}/dictionary/parse" \
  -H "X-API-Key: ${API_KEY}" \
  -F "dictionary_file=@${DICT_FILE}" \
  -F "save_program=true" \
  | jq -r '.program_id')

echo "Program ID: ${PROGRAM_ID}"
```

### Example 3: Get Program Details

```bash
# By program ID
curl -s "${API_BASE_URL}/dictionary/${PROGRAM_ID}" \
  -H "X-API-Key: ${API_KEY}" \
  | python -m json.tool

# By program name (URL encode spaces if needed)
PROGRAM_NAME="20241202-143022-ClinicalTrial"
curl -s "${API_BASE_URL}/dictionary/${PROGRAM_NAME}" \
  -H "X-API-Key: ${API_KEY}" \
  | python -m json.tool

# By alias
curl -s "${API_BASE_URL}/dictionary/johnDoesFav01" \
  -H "X-API-Key: ${API_KEY}" \
  | python -m json.tool

# Extract specific fields using jq
curl -s "${API_BASE_URL}/dictionary/${PROGRAM_ID}" \
  -H "X-API-Key: ${API_KEY}" \
  | jq '{name, num_fields, num_logic_rules, use_count}'
```

### Example 4: Complete Workflow Script

```bash
#!/bin/bash
# complete_workflow.sh - Complete dictionary parsing workflow

set -e  # Exit on error

# Configuration
API_BASE_URL="http://localhost:8000/api/v1"
API_KEY="your-api-key"
DICT_FILE="$1"

if [ -z "$DICT_FILE" ]; then
    echo "Usage: $0 <dictionary_file>"
    exit 1
fi

if [ ! -f "$DICT_FILE" ]; then
    echo "Error: File not found: $DICT_FILE"
    exit 1
fi

echo "============================================"
echo "Data Analyzer API - Complete Workflow"
echo "============================================"

# Step 1: Health check
echo ""
echo "Step 1: Checking API health..."
HEALTH=$(curl -s "${API_BASE_URL}/health")
STATUS=$(echo "$HEALTH" | jq -r '.status')

if [ "$STATUS" != "healthy" ]; then
    echo "Warning: API status is $STATUS"
fi

echo "✓ API is $STATUS"

# Step 2: Parse dictionary
echo ""
echo "Step 2: Parsing dictionary: $DICT_FILE"
PARSE_RESULT=$(curl -s -X POST "${API_BASE_URL}/dictionary/parse" \
  -H "X-API-Key: ${API_KEY}" \
  -F "dictionary_file=@${DICT_FILE}" \
  -F "save_program=true")

PROGRAM_ID=$(echo "$PARSE_RESULT" | jq -r '.program_id')
PROGRAM_NAME=$(echo "$PARSE_RESULT" | jq -r '.program_name')
FIELDS=$(echo "$PARSE_RESULT" | jq -r '.fields_extracted')
LOGIC_RULES=$(echo "$PARSE_RESULT" | jq -r '.logic_rules_extracted')

echo "✓ Program created:"
echo "  ID: $PROGRAM_ID"
echo "  Name: $PROGRAM_NAME"
echo "  Fields: $FIELDS"
echo "  Logic Rules: $LOGIC_RULES"

# Step 3: Retrieve program details
echo ""
echo "Step 3: Retrieving program details..."
PROGRAM=$(curl -s "${API_BASE_URL}/dictionary/${PROGRAM_ID}" \
  -H "X-API-Key: ${API_KEY}")

USE_COUNT=$(echo "$PROGRAM" | jq -r '.use_count')
CREATED=$(echo "$PROGRAM" | jq -r '.created_at')

echo "✓ Program retrieved:"
echo "  Use Count: $USE_COUNT"
echo "  Created: $CREATED"

# Step 4: Display schema
echo ""
echo "Step 4: Field Schema:"
echo "$PROGRAM" | jq -r '.schema | to_entries[] | "  - \(.key): \(.value.field_type)"'

# Step 5: Display logic rules (if any)
if [ "$LOGIC_RULES" != "0" ]; then
    echo ""
    echo "Step 5: Conditional Logic Rules:"
    echo "$PROGRAM" | jq -r '.rules[] | "  - \(.description)\n    Condition: \(.condition)"'
fi

echo ""
echo "============================================"
echo "✓ Workflow complete!"
echo "============================================"
```

**Usage:**
```bash
chmod +x complete_workflow.sh
./complete_workflow.sh clinical_trial_dict.csv
```

---

## Error Handling Examples

### Python: Comprehensive Error Handling

```python
import requests
from typing import Optional, Dict, Any

class DataAnalyzerAPIError(Exception):
    """Base exception for API errors"""
    pass

class AuthenticationError(DataAnalyzerAPIError):
    """Authentication failed"""
    pass

class RateLimitError(DataAnalyzerAPIError):
    """Rate limit exceeded"""
    pass

class NotFoundError(DataAnalyzerAPIError):
    """Resource not found"""
    pass

class ValidationError(DataAnalyzerAPIError):
    """Request validation failed"""
    pass

def handle_response(response: requests.Response) -> Dict[str, Any]:
    """
    Handle API response with comprehensive error checking

    Args:
        response: requests Response object

    Returns:
        Parsed JSON response

    Raises:
        Appropriate API error based on status code
    """
    try:
        response_data = response.json()
    except ValueError:
        response_data = {"error": "Invalid JSON response"}

    if response.status_code == 200:
        return response_data

    elif response.status_code == 400:
        raise ValidationError(f"Validation error: {response_data.get('detail')}")

    elif response.status_code == 401:
        raise AuthenticationError(f"Authentication required: {response_data.get('detail')}")

    elif response.status_code == 403:
        raise AuthenticationError(f"Invalid credentials: {response_data.get('detail')}")

    elif response.status_code == 404:
        raise NotFoundError(f"Not found: {response_data.get('detail')}")

    elif response.status_code == 429:
        retry_after = response.headers.get('Retry-After', '60')
        raise RateLimitError(f"Rate limit exceeded. Retry after {retry_after} seconds")

    elif response.status_code == 500:
        raise DataAnalyzerAPIError(f"Server error: {response_data.get('detail')}")

    elif response.status_code == 503:
        raise DataAnalyzerAPIError(f"Service unavailable: {response_data.get('detail')}")

    else:
        raise DataAnalyzerAPIError(f"HTTP {response.status_code}: {response_data.get('detail')}")

def safe_parse_dictionary(file_path: str, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Parse dictionary with retry logic and comprehensive error handling

    Args:
        file_path: Path to dictionary file
        max_retries: Maximum number of retry attempts

    Returns:
        Parse result or None if failed
    """
    url = f"{API_BASE_URL}/dictionary/parse"

    for attempt in range(1, max_retries + 1):
        try:
            with open(file_path, "rb") as f:
                files = {"dictionary_file": f}
                data = {"save_program": "true"}

                response = requests.post(url, headers=headers_api, files=files, data=data)
                result = handle_response(response)

                print(f"✓ Success on attempt {attempt}")
                return result

        except RateLimitError as e:
            print(f"⚠ Attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                wait_time = 60  # Wait 60 seconds
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print("✗ Max retries exceeded")
                return None

        except AuthenticationError as e:
            print(f"✗ Authentication error: {e}")
            print("Check your API key in .env file")
            return None

        except ValidationError as e:
            print(f"✗ Validation error: {e}")
            print("Check file format and content")
            return None

        except NotFoundError as e:
            print(f"✗ Not found: {e}")
            return None

        except DataAnalyzerAPIError as e:
            print(f"⚠ Attempt {attempt}/{max_retries}: {e}")
            if attempt < max_retries:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print("✗ Max retries exceeded")
                return None

        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            return None

    return None

# Usage
result = safe_parse_dictionary("clinical_trial_dict.csv")
if result:
    print(f"Program created: {result['program_id']}")
else:
    print("Failed to parse dictionary")
```

### JavaScript: Error Handling with Async/Await

```javascript
class DataAnalyzerAPIError extends Error {
  constructor(message, statusCode, details) {
    super(message);
    this.name = 'DataAnalyzerAPIError';
    this.statusCode = statusCode;
    this.details = details;
  }
}

async function handleResponse(response) {
  const data = await response.json();

  if (!response.ok) {
    const message = data.detail || data.error || 'Unknown error';

    switch (response.status) {
      case 400:
        throw new DataAnalyzerAPIError('Validation Error', 400, message);
      case 401:
      case 403:
        throw new DataAnalyzerAPIError('Authentication Error', response.status, message);
      case 404:
        throw new DataAnalyzerAPIError('Not Found', 404, message);
      case 429:
        throw new DataAnalyzerAPIError('Rate Limit Exceeded', 429, message);
      case 500:
      case 503:
        throw new DataAnalyzerAPIError('Server Error', response.status, message);
      default:
        throw new DataAnalyzerAPIError('API Error', response.status, message);
    }
  }

  return data;
}

async function safeParseDictionary(file, maxRetries = 3) {
  const url = `${API_BASE_URL}/dictionary/parse`;

  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const formData = new FormData();
      formData.append('dictionary_file', file);
      formData.append('save_program', 'true');

      const response = await fetch(url, {
        method: 'POST',
        headers: headersAPI,
        body: formData
      });

      const result = await handleResponse(response);
      console.log(`✓ Success on attempt ${attempt}`);
      return result;

    } catch (error) {
      console.error(`⚠ Attempt ${attempt}/${maxRetries}: ${error.message}`);

      if (error.statusCode === 429 && attempt < maxRetries) {
        // Rate limit - wait 60 seconds
        console.log('Waiting 60 seconds before retry...');
        await new Promise(resolve => setTimeout(resolve, 60000));
      } else if (error.statusCode >= 500 && attempt < maxRetries) {
        // Server error - exponential backoff
        const waitTime = Math.pow(2, attempt) * 1000;
        console.log(`Waiting ${waitTime/1000} seconds before retry...`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
      } else if (error.statusCode === 401 || error.statusCode === 403) {
        // Authentication error - don't retry
        console.error('✗ Authentication failed. Check your API key.');
        throw error;
      } else if (attempt === maxRetries) {
        console.error('✗ Max retries exceeded');
        throw error;
      }
    }
  }

  throw new Error('Failed to parse dictionary after all retries');
}

// Usage
try {
  const result = await safeParseDictionary(file);
  console.log('Program created:', result.program_id);
} catch (error) {
  console.error('Error:', error.message);
}
```

---

## Advanced Usage

### Python: API Client Class

```python
class DataAnalyzerClient:
    """
    Complete API client for Data Analyzer

    Features:
    - Automatic authentication
    - Retry logic with exponential backoff
    - Rate limit handling
    - Comprehensive error handling
    - Request logging
    """

    def __init__(self, base_url: str, api_key: str, admin_password: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.admin_password = admin_password
        self.session = requests.Session()
        self.session.headers.update({'X-API-Key': api_key})

    def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        response = self.session.request(method, url, **kwargs)
        return handle_response(response)

    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        return self._make_request('GET', '/health')

    def parse_dictionary(
        self,
        file_path: str,
        save_program: bool = True,
        program_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Parse dictionary file"""
        with open(file_path, 'rb') as f:
            files = {'dictionary_file': f}
            data = {'save_program': str(save_program).lower()}

            if program_name:
                data['program_name'] = program_name

            return self._make_request('POST', '/dictionary/parse', files=files, data=data)

    def get_program(self, identifier: str) -> Dict[str, Any]:
        """Get program by ID, name, or alias"""
        return self._make_request('GET', f'/dictionary/{identifier}')

    def list_programs(
        self,
        search: Optional[str] = None,
        status: str = 'active',
        limit: int = 50,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List programs (when implemented)"""
        params = {
            'status': status,
            'limit': limit,
            'offset': offset
        }

        if search:
            params['search'] = search

        return self._make_request('GET', '/programs', params=params)

# Usage
client = DataAnalyzerClient(
    base_url="http://localhost:8000/api/v1",
    api_key="your-api-key"
)

# Check health
health = client.health_check()
print(f"API Status: {health['status']}")

# Parse dictionary
result = client.parse_dictionary(
    file_path="clinical_trial_dict.csv",
    program_name="ClinicalTrialV1"
)
print(f"Program created: {result['program_id']}")

# Get program
program = client.get_program(result['program_id'])
print(f"Fields: {program['num_fields']}")
```

---

**Last Updated:** 2025-12-02
**API Version:** 1.0.0
