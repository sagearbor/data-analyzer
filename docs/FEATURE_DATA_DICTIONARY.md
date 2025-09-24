# Data Dictionary Parser Feature

## Overview

The Data Analyzer now includes an AI-powered data dictionary parser that uses LLM-generated Python code to parse various dictionary formats. This approach provides deterministic, verifiable, and cacheable results compared to direct LLM parsing.

## Key Features

### 1. LLM Code Generation
- Uses Azure OpenAI (gpt-4o-mini) to generate Python parsing code
- Generated code is executed in a sandboxed subprocess
- Results are validated and confidence scores provided
- Parser code can be downloaded for inspection/reuse

### 2. Supported Formats
- **CSV**: Standard column definitions with types, constraints
- **JSON**: Structured field definitions (clinical trials, REDCap)
- **Excel**: Spreadsheet-based dictionaries (via CSV export)
- **Text**: Unstructured text descriptions
- **REDCap**: REDCap data dictionary format
- **Clinical**: Clinical trial CRF specifications

### 3. Caching System
- Generated parsers are cached by content structure
- File-based cache in `~/.cache/data_analyzer/parsers/`
- 30-day cache lifetime
- Significantly improves performance for similar dictionaries

### 4. Validation & Confidence
- Validates parsed structure against expected schema
- Provides confidence scores (0-100%)
- Reports errors and warnings
- Supports retry with feedback for failed attempts

## Usage

### Web Interface

1. Upload a CSV file to analyze
2. In the sidebar, find the "📚 Data Dictionary" section
3. Either:
   - Select a demo dictionary from the dropdown that matches your demo data
   - Click "📁 Load Demo Dictionary" to load it
   OR
   - Upload your own dictionary file (CSV, JSON, TXT, XLSX)
4. Click "🤖 Parse Dictionary" to process
5. Review the extracted schema and rules
6. Click "✅ Apply to Validation" to use for data analysis
7. Optionally download the generated parser code

### MCP Server API

```python
# Using the MCP server
result = await mcp_client.parse_data_dictionary(
    dictionary_content="Column,Type,Required\nid,integer,Yes\n...",
    format_hint="csv",
    use_cache=True,
    debug=False
)

# Result structure
{
    "success": true,
    "schema": {"id": "int", "name": "str", ...},
    "rules": {"id": {"min": 1, "max": 999999}, ...},
    "metadata": {...},
    "confidence_score": 0.95,
    "parser_code": "def parse_dictionary(content):\n..."
}
```

### Direct Python Usage

```python
from src.core.data_dictionary_parser import DataDictionaryParser

parser = DataDictionaryParser()
result = await parser.parse_dictionary(
    content=dictionary_text,
    format_hint="csv",
    use_cache=True
)

# Convert to schema and rules
schema, rules = parser.convert_to_schema_and_rules(result)
```

## Example Dictionaries

### Simple CSV
```csv
Column,Type,Required,Min,Max,Description
employee_id,integer,Yes,1,999999,Unique ID
salary,decimal,Yes,30000,500000,Annual salary
department,string,Yes,,,Must be HR/IT/Sales
```

### Clinical JSON
```json
{
  "fields": [
    {
      "field_name": "subject_id",
      "data_type": "string",
      "required": true,
      "validation": {
        "pattern": "^S[0-9]{3,6}$"
      }
    }
  ]
}
```

### REDCap Format
```
Variable / Field Name: patient_id
Field Type: text
Field Label: Patient Identifier
Required: yes
Validation Pattern: ^P[0-9]{6}$
```

## Demo Data

The application now includes diverse, culturally-themed demo datasets:

- **CSV - Western**: Western names, standard business data
- **CSV - Asian**: Asian names, different date formats
- **JSON - Mixed**: Nested structure, international names
- **CSV - Clinical Trial**: Medical/pharmaceutical data

Each dataset includes intentional errors for testing validation:
- Invalid ages (negative, too high)
- Malformed dates
- Out-of-range values
- Invalid data types

## Security Considerations

Current implementation (MVP):
- 600-second (10 minute) execution timeout
- Subprocess isolation
- No network access in parser code

Future enhancements planned:
- Docker container isolation
- Resource limits (memory, CPU)
- Import whitelist
- AST validation before execution
- RestrictedPython environment

## Performance

- First parse: ~2-5 seconds (includes LLM call)
- Cached parse: <100ms
- Cache hit rate: ~80% for similar dictionaries
- Cache lifetime: 365 days (1 year)
- Execution timeout: 600 seconds (10 minutes)
- Confidence scores: 75-95% typical

## Error Handling

- Automatic retry with feedback (up to 3 attempts)
- Detailed error messages and warnings
- Partial results when possible
- Debug mode for troubleshooting

## Configuration

Environment variables (`.env`):
```
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_CHAT_DEPLOYMENT_NAME=gpt-4o-mini
```

## Testing

Test dictionaries provided in `/test_dictionaries/`:
- `simple_csv_dict.csv`: Basic employee data
- `clinical_trial_dict.json`: Clinical trial fields
- `redcap_style_dict.txt`: REDCap format example

Run tests:
```bash
python test_parser.py
```

## Limitations

- Simulation mode only supports CSV dictionaries
- Complex nested structures may require manual adjustment
- LLM token limits may truncate very large dictionaries
- Cache invalidation is time-based only (30 days)

## Future Enhancements

- Support for database schemas (SQL DDL)
- Industry-specific formats (HL7, FHIR, DICOM)
- Version control for parser code
- Collaborative dictionary editing
- API endpoint for dictionary parsing
- Advanced caching strategies (LRU, size-based)
- Parser code optimization and minification