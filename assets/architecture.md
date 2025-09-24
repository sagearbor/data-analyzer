# System Architecture

## Data Flow Overview

The Data Analyzer system processes data through multiple stages:

### 1. Data Input Stage
- **CSV Files**: Primary data format (currently supported)
- **JSON/Excel/Text**: Planned formats for future support
- **Upload Interface**: Web-based file upload

### 2. Dictionary Processing (AI-Powered)
- **Dictionary Sources**:
  - Manual upload (CSV, JSON, REDCap, Clinical formats)
  - Demo dictionaries (pre-configured examples)

- **LLM Processing**:
  - **Azure OpenAI (GPT-4o-mini)** generates Python code
  - Code generation approach (not direct parsing)
  - 600-second execution timeout
  - Subprocess isolation for safety

- **Caching**:
  - File-based cache (`~/.cache/data_analyzer/parsers/`)
  - 1-year cache lifetime
  - Cache key based on dictionary structure

### 3. MCP Server Components
- **DataLoader**: Handles multiple data formats
- **QualityChecker**: Performs validation checks
- **DataDictionaryParser**: Parses dictionaries using LLM

### 4. Validation Pipeline
- **Type Validation**: Ensures data matches expected types
- **Range Validation**: Checks numeric values against min/max
- **Allowed Values**: Validates against predefined lists
- **Statistics**: Calculates summary statistics

### 5. Web Interface
- **Schema Editor**: Configure expected data types
- **Rules Editor**: Set validation constraints
- **Results Dashboard**: Display validation results
- **Parser Download**: Export generated Python code

## Architecture Flow

```
Data Sources → Upload Interface → MCP Server
                                       ↓
Dictionary → LLM (Azure OpenAI) → Python Code → Parser
     ↓                                  ↓
  Cache ← ← ← ← ← ← ← ← ← ← ← Success ←
     ↓
Schema & Rules → Validation Pipeline → Results Dashboard
```

## Key Components

### LLM Integration
- **Purpose**: Generate deterministic parser code
- **Model**: Azure OpenAI GPT-4o-mini
- **Approach**: Code generation vs direct parsing
- **Benefits**:
  - Deterministic results
  - Verifiable code
  - Cacheable output
  - ~95% confidence

### Caching System
- **Location**: `~/.cache/data_analyzer/parsers/`
- **Duration**: 365 days (1 year)
- **Key Generation**: SHA256 hash of dictionary structure
- **Hit Rate**: ~80% for similar dictionaries

### Security Measures
- **Current**:
  - Subprocess isolation
  - 600-second timeout
  - No network access in parsers

- **Planned**:
  - Docker containerization
  - Resource limits (memory, CPU)
  - Import whitelist
  - AST validation

## Performance Metrics
- **First Parse**: 2-5 seconds (includes LLM call)
- **Cached Parse**: <100ms
- **Max Timeout**: 600 seconds (10 minutes)
- **Cache Size**: Unlimited (planned: LRU eviction)

## Supported Formats

### Data Formats
- ✅ CSV (fully supported)
- 🔄 JSON (planned)
- 🔄 Excel (planned)
- 🔄 Parquet (planned)

### Dictionary Formats
- ✅ CSV dictionaries
- ✅ JSON dictionaries
- ✅ REDCap format
- ✅ Clinical trial specifications
- ✅ Text-based descriptions

## Error Handling
- Automatic retry with feedback (3 attempts)
- Detailed error messages
- Partial results when possible
- Debug mode for troubleshooting