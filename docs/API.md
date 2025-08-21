# Data Analyzer MCP Server API Documentation

## Overview

The Data Analyzer MCP (Model Context Protocol) server provides standardized tools for analyzing structured data files. Currently supports CSV format with an extensible architecture for future data formats.

## Server Information

- **Server Name**: `data-analyzer`
- **Version**: `1.0.0`
- **Protocol**: MCP (Model Context Protocol)
- **Transport**: stdio

## Available Tools

### 1. analyze_data

Performs comprehensive data quality analysis on structured data.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "data_content": {
      "type": "string",
      "description": "Data content as string or base64 encoded data",
      "required": true
    },
    "file_format": {
      "type": "string",
      "description": "Data format (csv, future: json, excel)",
      "default": "csv"
    },
    "schema": {
      "type": "object",
      "description": "Expected data types for columns",
      "additionalProperties": {"type": "string"},
      "example": {
        "id": "int",
        "name": "str",
        "age": "int",
        "country": "str"
      }
    },
    "rules": {
      "type": "object",
      "description": "Validation rules for columns",
      "additionalProperties": {"type": "object"},
      "example": {
        "age": {"min": 0, "max": 120},
        "country": {"allowed": ["USA", "CAN", "MEX"]}
      }
    },
    "min_rows": {
      "type": "integer",
      "description": "Minimum required number of rows",
      "default": 1
    },
    "encoding": {
      "type": "string",
      "description": "Data encoding (utf-8, latin1, etc.)",
      "default": "utf-8"
    }
  }
}
```

#### Response Format

```json
{
  "timestamp": "2025-01-15T10:30:00.000Z",
  "file_format": "csv",
  "summary_stats": {
    "shape": {"rows": 1000, "columns": 5},
    "columns": ["id", "name", "age", "country", "salary"],
    "dtypes": {
      "id": "int64",
      "name": "object",
      "age": "int64",
      "country": "object",
      "salary": "float64"
    },
    "missing_values": {"id": 0, "name": 3, "age": 0, "country": 1, "salary": 2},
    "duplicate_rows": 5,
    "memory_usage_mb": 0.85,
    "numeric_summary": {
      "age": {"min": 18, "max": 65, "mean": 35.2, "std": 12.4},
      "salary": {"min": 30000, "max": 150000, "mean": 75000, "std": 25000}
    }
  },
  "checks": {
    "row_count": {
      "check": "row_count",
      "passed": true,
      "row_count": 1000,
      "min_required": 1,
      "message": "Found 1000 rows (minimum: 1)"
    },
    "data_types": {
      "check": "data_types",
      "passed": false,
      "total_columns_checked": 5,
      "issues_found": 1,
      "issues": [
        {
          "column": "age",
          "issue": "type_mismatch",
          "expected_type": "int",
          "actual_type": "object",
          "sample_values": ["25", "thirty", "35"]
        }
      ],
      "message": "Type validation: 1 issues found"
    },
    "value_ranges": {
      "check": "value_ranges",
      "passed": false,
      "issues_found": 2,
      "issues": [
        {
          "column": "age",
          "rule": "max <= 120",
          "violation_count": 3,
          "violating_rows": [45, 67, 89]
        },
        {
          "column": "country",
          "rule": "allowed_values: ['USA', 'CAN', 'MEX']",
          "invalid_values": ["INVALID", "UK"],
          "violation_count": 8,
          "violating_rows": [12, 34, 56, 78, 90, 123, 145, 167]
        }
      ],
      "message": "Range validation: 2 issues found"
    }
  },
  "overall_passed": false,
  "total_issues": 3,
  "issues": [
    {
      "column": "age",
      "issue": "type_mismatch",
      "expected_type": "int",
      "actual_type": "object"
    },
    {
      "column": "age",
      "rule": "max <= 120",
      "violation_count": 3
    },
    {
      "column": "country",
      "rule": "allowed_values",
      "invalid_values": ["INVALID", "UK"],
      "violation_count": 8
    }
  ]
}
```

### 2. get_data_info

Retrieves basic information about data structure without performing validation.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "data_content": {
      "type": "string",
      "description": "Data content as string or base64 encoded data",
      "required": true
    },
    "file_format": {
      "type": "string",
      "description": "Data format (csv, future: json, excel)",
      "default": "csv"
    },
    "sample_rows": {
      "type": "integer",
      "description": "Number of sample rows to return",
      "default": 5
    }
  }
}
```

#### Response Format

```json
{
  "format": "csv",
  "shape": {"rows": 1000, "columns": 5},
  "columns": ["id", "name", "age", "country", "salary"],
  "dtypes": {
    "id": "int64",
    "name": "object",
    "age": "int64",
    "country": "object",
    "salary": "float64"
  },
  "sample_data": [
    {"id": 1, "name": "John Doe", "age": 25, "country": "USA", "salary": 50000},
    {"id": 2, "name": "Jane Smith", "age": 30, "country": "CAN", "salary": 55000},
    {"id": 3, "name": "Bob Johnson", "age": 35, "country": "MEX", "salary": 60000},
    {"id": 4, "name": "Alice Brown", "age": 28, "country": "USA", "salary": 52000},
    {"id": 5, "name": "Charlie Wilson", "age": 45, "country": "CAN", "salary": 75000}
  ],
  "missing_values": {"id": 0, "name": 3, "age": 0, "country": 1, "salary": 2},
  "duplicate_rows": 5
}
```

## Data Type Validation

### Supported Types

| Type | Description | Validation |
|------|-------------|------------|
| `int` | Integer numbers | Converts to numeric, checks for whole numbers |
| `float` | Floating point numbers | Converts to numeric |
| `str` | String/text data | No conversion needed |
| `bool` | Boolean values | Checks for true/false values |
| `datetime` | Date/time values | Attempts datetime parsing |

### Type Conversion Rules

- **int**: `pd.to_numeric()` with integer validation
- **float**: `pd.to_numeric()` allowing decimals  
- **str**: No conversion (pandas object type)
- **bool**: Boolean type checking
- **datetime**: `pd.to_datetime()` parsing

## Validation Rules

### Range Rules

For numeric columns, specify min/max bounds:

```json
{
  "column_name": {
    "min": 0,
    "max": 100
  }
}
```

### Allowed Values Rules

For categorical columns, specify allowed value lists:

```json
{
  "column_name": {
    "allowed": ["value1", "value2", "value3"]
  }
}
```

### Combined Rules

You can combine multiple rule types:

```json
{
  "age": {
    "min": 18,
    "max": 65
  },
  "status": {
    "allowed": ["active", "inactive", "pending"]
  },
  "score": {
    "min": 0.0,
    "max": 100.0
  }
}
```

## Input Formats

### Plain Text Data

```json
{
  "data_content": "id,name,age\n1,John,25\n2,Jane,30"
}
```

### Base64 Encoded Data

```json
{
  "data_content": "aWQsbmFtZSxhZ2UKMSxKb2huLDI1CjIsSmFuZSwzMA=="
}
```

### Data URLs

```json
{
  "data_content": "data:text/csv;base64,aWQsbmFtZSxhZ2UKMSxKb2huLDI1CjIsSmFuZSwzMA=="
}
```

## Error Handling

### Analysis Errors

```json
{
  "error": "Failed to load CSV: Error message details",
  "type": "analysis_error"
}
```

### Info Errors

```json
{
  "error": "Unsupported file format: xlsx",
  "type": "info_error"
}
```

### Common Error Types

| Error Type | Description | Solution |
|------------|-------------|----------|
| `analysis_error` | Data processing failed | Check data format and encoding |
| `info_error` | Info retrieval failed | Verify input parameters |
| `format_error` | Unsupported file format | Use supported format (CSV) |
| `encoding_error` | Character encoding issue | Try different encoding |

## Usage Examples

### Basic CSV Analysis

```python
# MCP tool call
{
  "tool": "analyze_data",
  "arguments": {
    "data_content": "id,name,age,country\n1,John,25,USA\n2,Jane,30,CAN",
    "file_format": "csv",
    "min_rows": 2
  }
}
```

### Schema Validation

```python
{
  "tool": "analyze_data", 
  "arguments": {
    "data_content": "id,name,age,salary\n1,John,25,50000\n2,Jane,30,55000",
    "schema": {
      "id": "int",
      "name": "str", 
      "age": "int",
      "salary": "float"
    }
  }
}
```

### Rules Validation

```python
{
  "tool": "analyze_data",
  "arguments": {
    "data_content": "id,age,status\n1,25,active\n2,30,pending",
    "rules": {
      "age": {"min": 18, "max": 65},
      "status": {"allowed": ["active", "inactive", "pending"]}
    }
  }
}
```

### Get Basic Info

```python
{
  "tool": "get_data_info",
  "arguments": {
    "data_content": "id,name,age\n1,John,25\n2,Jane,30\n3,Bob,35",
    "sample_rows": 2
  }
}
```

## Performance Considerations

- **Memory Usage**: Large files are processed in-memory
- **File Size Limits**: Recommended < 200MB for optimal performance
- **Processing Time**: Linear with file size and complexity of validation rules
- **Concurrent Requests**: Server handles async processing

## Future Extensibility

The API is designed to support additional data formats:

### Planned Formats

- **JSON**: Structured data analysis
- **Excel**: Multi-sheet support
- **Parquet**: Columnar data format
- **XML**: Hierarchical data structures

### Extension Points

- `file_format` parameter accepts new format types
- `DataLoader.load_data()` method is extensible
- Validation logic applies to any tabular data structure

## Integration Notes

### MCP Client Libraries

Use standard MCP client libraries to interact with this server:

- **Python**: `mcp-client-python`
- **TypeScript**: `@modelcontextprotocol/client`
- **Custom**: Implement stdio communication

### Authentication

Current version uses stdio transport with no authentication. For production deployments, consider:

- API key authentication
- OAuth2 integration  
- Network-based MCP transport

### Monitoring

The server provides structured logging for:

- Request/response timing
- Error tracking
- Performance metrics
- Data processing statistics
