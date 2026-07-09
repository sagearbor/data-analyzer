# Data Analysis API Endpoints

## Overview

The data analysis endpoints provide programmatic access to data quality analysis with optional conditional logic validation. These endpoints are designed for integration with external systems, CI/CD pipelines, and automated data validation workflows.

## Endpoints

### 1. POST /api/v1/analyze

Analyze data file with optional data dictionary.

**URL**: `/api/v1/analyze`

**Method**: `POST`

**Authentication**: Required (X-API-Key header)

**Rate Limit**: 10 requests per minute

**Content-Type**: `multipart/form-data`

#### Request Parameters

##### Form Data (multipart/form-data)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data_file` | File | Yes | Data file to analyze (CSV, JSON, Excel, Parquet) |
| `dictionary_file` | File | No | Optional data dictionary file (CSV, JSON, TXT) |
| `data_format` | String | No | Data file format: `csv`, `json`, `excel`, `parquet` (default: `csv`) |
| `validate_logic` | Boolean | No | Whether to validate conditional logic rules (default: `true`) |
| `return_format` | String | No | Response format: `json`, `html`, `excel` (default: `json`) |

##### Headers

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `X-API-Key` | String | Yes | API key for authentication |

#### Response

**Status Code**: `200 OK`

**Content-Type**: `application/json`

**Response Schema**: `AnalyzeResponse`

```json
{
  "analysis_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "timestamp": "2025-12-02T14:30:00Z",
  "summary": {
    "total_rows": 1000,
    "total_columns": 25,
    "issues_found": 42,
    "logic_violations": 8,
    "execution_time_seconds": 2.5
  },
  "field_violations": [
    {
      "field_name": "age",
      "row_index": 42,
      "violation_type": "range_violation",
      "expected": "0 <= age <= 120",
      "actual": 150,
      "severity": "error"
    }
  ],
  "logic_violations": [
    {
      "rule_id": "skip_if_001",
      "rule_description": "If gender=1, then pregnancy_status should be blank",
      "row_index": 15,
      "affected_fields": ["gender", "pregnancy_status"],
      "actual_values": {"gender": 1, "pregnancy_status": "Yes"},
      "expected_behavior": "pregnancy_status should be blank when gender=1",
      "severity": "error"
    }
  ],
  "recommendations": [
    "Review data types - some columns have type mismatches",
    "Review 8 conditional logic violations"
  ],
  "program_used": "20241202-143022-ClinicalTrial"
}
```

#### Workflow

1. **Load Data File**: Parse and validate the uploaded data file
2. **Parse Dictionary** (if provided): Extract schema and validation rules using LLM
3. **Create/Find Program** (if dictionary provided): Save or retrieve cached validation program
4. **Run Quality Checks**: Execute basic quality checks (types, ranges, allowed values)
5. **Run Logic Validation** (if enabled): Execute conditional logic validation
6. **Return Results**: Comprehensive analysis with violations and recommendations

#### Error Responses

| Status Code | Description |
|-------------|-------------|
| `400` | Bad Request - Invalid file format or data |
| `401` | Unauthorized - Missing API key |
| `403` | Forbidden - Invalid API key |
| `413` | Payload Too Large - File exceeds size limit |
| `500` | Internal Server Error - Processing error |
| `503` | Service Unavailable - Required services not initialized |

#### File Size Limits

- **Data File**: 50 MB maximum
- **Dictionary File**: 10 MB maximum

#### Supported Data Formats

##### CSV
- Automatic encoding detection
- Handles various delimiters
- Supports quoted fields

##### JSON
- Nested structure flattening
- Array of objects format
- Single object format

##### Excel
- .xlsx and .xls formats
- First sheet by default
- Preserves data types

##### Parquet
- Compressed columnar format
- High performance for large datasets
- Schema preservation

#### Example Usage

##### cURL

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "X-API-Key: your-api-key" \
  -F "data_file=@sample_data.csv" \
  -F "dictionary_file=@data_dictionary.csv" \
  -F "data_format=csv" \
  -F "validate_logic=true" \
  -F "return_format=json"
```

##### Python (requests)

```python
import requests

url = "http://localhost:8000/api/v1/analyze"
headers = {"X-API-Key": "your-api-key"}
files = {
    "data_file": open("sample_data.csv", "rb"),
    "dictionary_file": open("data_dictionary.csv", "rb")
}
data = {
    "data_format": "csv",
    "validate_logic": True,
    "return_format": "json"
}

response = requests.post(url, headers=headers, files=files, data=data)
result = response.json()
print(f"Analysis ID: {result['analysis_id']}")
print(f"Issues found: {result['summary']['issues_found']}")
```

---

### 2. POST /api/v1/analyze/with-program

Analyze data using a cached validation program.

**URL**: `/api/v1/analyze/with-program`

**Method**: `POST`

**Authentication**: Required (X-API-Key header)

**Rate Limit**: 10 requests per minute

**Content-Type**: `multipart/form-data`

#### Request Parameters

##### Form Data (multipart/form-data)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data_file` | File | Yes | Data file to analyze |
| `program` | String | Yes | Program name, ID, or alias |
| `data_format` | String | No | Data file format (default: `csv`) |
| `return_format` | String | No | Response format (default: `json`) |

##### Headers

| Header | Type | Required | Description |
|--------|------|----------|-------------|
| `X-API-Key` | String | Yes | API key for authentication |

#### Program Lookup

The `program` parameter can be one of:

- **Program Name**: Auto-generated name (e.g., `"20241202-143022-ClinicalTrial"`)
- **Program ID**: UUID (e.g., `"a1b2c3d4-e5f6-7890-abcd-ef1234567890"`)
- **Program Alias**: User-friendly alias (e.g., `"johnDoesFav01"`)

#### Response

**Status Code**: `200 OK`

**Content-Type**: `application/json`

**Response Schema**: `AnalyzeResponse` (same as `/api/v1/analyze`)

#### Workflow

1. **Load Validation Program**: Retrieve program by name, ID, or alias from database
2. **Check Program Status**: Verify program is active (not deleted)
3. **Load Data File**: Parse and validate the uploaded data file
4. **Run Quality Checks**: Execute checks using program's schema and rules
5. **Run Logic Validation**: Execute program's generated validation code
6. **Update Usage Tracking**: Increment program execution count and last_used timestamp
7. **Return Results**: Comprehensive analysis with violations and recommendations

#### Error Responses

| Status Code | Description |
|-------------|-------------|
| `400` | Bad Request - Invalid file format or data |
| `401` | Unauthorized - Missing API key |
| `403` | Forbidden - Invalid API key |
| `404` | Not Found - Program not found or deleted |
| `413` | Payload Too Large - File exceeds size limit |
| `500` | Internal Server Error - Processing error |
| `503` | Service Unavailable - Required services not initialized |

#### Example Usage

##### cURL

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/with-program" \
  -H "X-API-Key: your-api-key" \
  -F "data_file=@sample_data.csv" \
  -F "program=johnDoesFav01" \
  -F "data_format=csv" \
  -F "return_format=json"
```

##### Python (requests)

```python
import requests

url = "http://localhost:8000/api/v1/analyze/with-program"
headers = {"X-API-Key": "your-api-key"}
files = {"data_file": open("sample_data.csv", "rb")}
data = {
    "program": "johnDoesFav01",
    "data_format": "csv",
    "return_format": "json"
}

response = requests.post(url, headers=headers, files=files, data=data)
result = response.json()
print(f"Program used: {result['program_used']}")
print(f"Logic violations: {result['summary']['logic_violations']}")
```

---

## Common Response Fields

### AnalysisSummary

| Field | Type | Description |
|-------|------|-------------|
| `total_rows` | Integer | Number of rows analyzed |
| `total_columns` | Integer | Number of columns in dataset |
| `issues_found` | Integer | Total field-level validation issues |
| `logic_violations` | Integer | Total conditional logic violations |
| `execution_time_seconds` | Float | Analysis execution time |

### FieldViolation

| Field | Type | Description |
|-------|------|-------------|
| `field_name` | String | Name of the field with violation |
| `row_index` | Integer | Row number (0-indexed) |
| `violation_type` | String | Type: `type_mismatch`, `range_violation`, etc. |
| `expected` | String | Expected value or constraint |
| `actual` | Any | Actual value that violated constraint |
| `severity` | String | Severity: `error`, `warning`, `info` |

### LogicViolation

| Field | Type | Description |
|-------|------|-------------|
| `rule_id` | String | Unique rule identifier |
| `rule_description` | String | Human-readable rule description |
| `row_index` | Integer | Row number (0-indexed) |
| `affected_fields` | Array[String] | Fields involved in violation |
| `actual_values` | Object | Actual field values |
| `expected_behavior` | String | Expected behavior per rule |
| `severity` | String | Severity: `error`, `warning` |

---

## Best Practices

### 1. Program Reuse

For repeated analysis of similar datasets:
1. Upload dictionary once via `/api/v1/dictionary/parse`
2. Create a user-friendly alias for the program
3. Use `/api/v1/analyze/with-program` for subsequent analyses
4. This is faster and more consistent than re-parsing the dictionary

### 2. Error Handling

Always check the response status code:
- `200`: Analysis successful
- `4xx`: Client error (fix request and retry)
- `5xx`: Server error (retry with exponential backoff)

### 3. Rate Limiting

Respect rate limits:
- Monitor `X-RateLimit-*` headers in responses
- Implement exponential backoff on 429 errors
- Consider batch processing for large datasets

### 4. File Size Management

For large files:
- Split into smaller chunks if possible
- Use Parquet format for better compression
- Consider streaming analysis for very large datasets

### 5. Security

- Store API keys securely (environment variables, secrets manager)
- Never commit API keys to version control
- Rotate keys regularly
- Use HTTPS in production

---

## Testing

### Test Suite

Run the comprehensive test suite:

```bash
# Set API key
export DATA_ANALYZER_API_KEY="your-api-key"

# Start API server
uvicorn api_server:app --reload &

# Run tests
python test_analyze_endpoints.py
```

### Manual Testing

Test basic analysis:

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "X-API-Key: test-key" \
  -F "data_file=@tests/test_data/sample.csv" \
  -F "data_format=csv"
```

Test with program:

```bash
curl -X POST "http://localhost:8000/api/v1/analyze/with-program" \
  -H "X-API-Key: test-key" \
  -F "data_file=@tests/test_data/sample.csv" \
  -F "program=my-program-alias"
```

---

## Integration Guide

### Integration into api_server.py

The endpoints are ready for integration. Follow these steps:

1. **Verify Imports**: Ensure these imports exist in `api_server.py`:
   ```python
   from src.api_models import (
       AnalyzeResponse, AnalysisSummary, FieldViolation, LogicViolation,
       SeverityEnum, DataFormatEnum, ReturnFormatEnum
   )
   import mcp_server
   import time
   import uuid as uuid_lib
   import tempfile
   import pandas as pd
   ```

2. **Add Endpoints**: Copy both functions from `api_endpoints_analyze.py` to `api_server.py` in the "Data Analysis Endpoints" section (after health check, before dictionary endpoints)

3. **Test**: Run the test suite to verify integration

4. **Deploy**: Follow standard deployment procedures

---

## Troubleshooting

### Common Issues

#### 1. Service Unavailable (503)

**Cause**: Required services not initialized

**Solutions**:
- Check LLM client configuration (Azure OpenAI credentials)
- Verify mcp_server module is loaded
- Check logs for initialization errors

#### 2. Program Not Found (404)

**Cause**: Invalid program identifier or program deleted

**Solutions**:
- List available programs: `GET /api/v1/programs`
- Verify program name/ID/alias spelling
- Check if program was soft-deleted
- Create new program if needed

#### 3. File Too Large (413)

**Cause**: File exceeds size limits

**Solutions**:
- Reduce file size (filter rows, remove columns)
- Use compressed format (Parquet)
- Split into multiple smaller files

#### 4. Invalid File Format (400)

**Cause**: Unsupported format or corrupted file

**Solutions**:
- Verify file format matches `data_format` parameter
- Check file is not corrupted
- Ensure CSV uses supported encoding (UTF-8, Latin-1)
- Validate JSON structure

---

## Performance Considerations

### Execution Time

Typical execution times (1000 rows, 50 columns):
- **Basic analysis only**: 0.5-1.0 seconds
- **With dictionary parsing**: 3-5 seconds (first time)
- **With cached program**: 1.0-2.0 seconds
- **With logic validation**: 2-4 seconds

### Optimization Tips

1. **Use Cached Programs**: Parse dictionary once, reuse many times
2. **Choose Appropriate Format**: Parquet is fastest for large datasets
3. **Minimize Dictionary Size**: Only include necessary fields
4. **Batch Similar Files**: Group files with same schema
5. **Parallel Processing**: Use multiple API clients for independent files

---

## Security

### Authentication

- All endpoints require API key authentication
- API key passed via `X-API-Key` header
- Keys configured in environment variables
- Failed attempts are logged (without exposing keys)

### Data Privacy

- Files are processed in memory when possible
- Temporary files deleted immediately after use
- No data stored persistently unless explicitly saved (programs only)
- Analysis results contain only metadata, not raw data

### Input Validation

- File size limits enforced
- File format validation
- Schema validation for structured data
- AST-based code validation for generated logic
- Sandboxed execution for validation code

---

## Support

For issues or questions:
1. Check this documentation
2. Review logs in API server console
3. Run test suite to identify configuration issues
4. Consult `developer_checklist.yaml` for implementation status
5. Contact your administrator for API keys or deployment issues

---

**Last Updated**: 2025-12-02

**Version**: 1.0.0

**Status**: Implementation Complete, Ready for Integration
