# SCAN - Secure Clinical Analyzer Notifier (Data Analyzer, Azure MCP Server & Web Application)

A comprehensive data quality analysis tool built with Azure MCP (Model Context Protocol) server and Streamlit web interface. Supports multiple data formats including CSV, JSON, Excel (XLSX/XLS), and Parquet with comprehensive data quality analysis capabilities. Based on your [csvChecker](https://github.com/sagearbor/csvChecker) repository with enhanced cloud capabilities.

## Calling this service (REST / MCP / A2A)

This service exposes the same analysis core through three interfaces:

- **REST API** — `python api_server.py` (default port 8003); endpoints `/analyze`, `/data-info`, `/health`
- **MCP** — `python mcp_server.py` (stdio); tools `analyze_data` and `get_data_info` for Claude Desktop / Cursor
- **A2A** — `POST /a2a/data-analyzer` with an encoded A2A Message envelope; AgentCard at `GET /.well-known/agent.json`

See **[docs/INTEGRATION.md](docs/INTEGRATION.md)** for full curl examples, payload schemas, and A2A envelope details.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │────│  Streamlit App  │────│   MCP Server    │
│   (Browser)     │    │  (Azure Apps)   │    │   (Analysis)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Features

### Data Quality Checks
- **Row Count Validation**: Ensure minimum number of rows
- **Data Type Validation**: Validate against expected schema (int, float, str, bool, datetime)
- **Value Range Checks**: Set min/max bounds for numeric columns
- **Allowed Values**: Define categorical value constraints
- **Missing Value Analysis**: Detect and quantify missing data
- **Duplicate Detection**: Identify duplicate rows

### Supported Data Formats
- **CSV**: Full support with encoding detection ✅
- **JSON**: Nested structure flattening and array handling ✅
- **Excel**: Multi-sheet support (XLSX/XLS) ✅
- **Parquet**: Binary columnar format support ✅

### Web Interface
- **Interactive File Upload**: Support for data files with encoding detection
- **Schema Editor**: Visual interface to define column types
- **Rules Configuration**: Set up validation rules with intuitive UI
- **Real-time Analysis**: Instant feedback on data quality
- **Comprehensive Reports**: Visual dashboards with detailed metrics
- **Example Data**: Built-in sample data for testing

### MCP Server
- **Standards Compliant**: Follows MCP protocol specifications
- **Async Processing**: Non-blocking analysis operations
- **Flexible Input**: Supports data content, base64, file paths
- **Extensible Architecture**: Easy to add new data formats
- **Detailed Output**: Structured JSON results with comprehensive metadata

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)

### 1. Clone and Setup
```bash
git clone <your-repo>
cd data-analyzer
```

### 2. Run Locally
See "Local Development" section below for running the app locally.

**Note:** For experimental deployment options, see `./scripts/experimental/azure/` (not used in production).

## 📁 Project Structure

```
data-analyzer/
├── mcp_server.py           # MCP server implementation
├── web_app.py              # Streamlit web application
├── requirements.txt        # Web app dependencies
├── mcp_requirements.txt    # MCP server dependencies
├── Dockerfile              # Container configuration
├── scripts/                # Utility scripts
│   └── experimental/       # Experimental features (not production)
└── README.md               # This file
```

## 🔧 Local Development

### Setup Virtual Environment (Recommended)
```bash
python -m venv venv                    # Create virtual environment
source venv/bin/activate               # Activate (Linux/Mac)
# venv\Scripts\activate                # Activate (Windows)
which python                           # Verify you're in venv
deactivate                             # When done
```

### Run MCP Server Locally
```bash
pip install -r mcp_requirements.txt   # Install dependencies
python mcp_server.py                   # Run MCP server
```

### Run Web App Locally
```bash
pip install -r requirements.txt       # Install dependencies
streamlit run web_app.py               # Run Streamlit app
```

### Docker Development
```bash
# Build image
docker build -t data-analyzer .

# Run container (foreground - for testing)
docker run -p 3002:8002 -e APP_ENV=dev data-analyzer

# Run container (detached - background mode)
docker run -d -p 3002:8002 -e APP_ENV=dev data-analyzer

# Run with auto-restart on reboot
docker run -d --restart unless-stopped -p 3002:8002 -e APP_ENV=dev data-analyzer

# Production deployment (auto-restarts on reboot via docker-compose.yml)
docker compose up -d
```

## 📋 Usage Examples

### Basic Usage
1. Upload a data file (CSV supported, more formats coming) or load example data
2. Configure schema (optional):
   ```json
   {
     "id": "int",
     "name": "str", 
     "age": "int",
     "country": "str"
   }
   ```
3. Set validation rules (optional):
   ```json
   {
     "age": {"min": 0, "max": 120},
     "country": {"allowed": ["USA", "CAN", "MEX"]}
   }
   ```
4. Run analysis and review results

### MCP Server API
```python
# Analyze data via MCP
{
  "tool": "analyze_data",
  "arguments": {
    "data_content": "id,name,age\n1,John,25\n2,Jane,30",
    "file_format": "csv",
    "schema": {"id": "int", "name": "str", "age": "int"},
    "rules": {"age": {"min": 0, "max": 120}},
    "min_rows": 1
  }
}
```

### Response Format
```json
{
  "timestamp": "2025-01-15T10:30:00",
  "file_format": "csv",
  "summary_stats": {
    "shape": {"rows": 5, "columns": 4},
    "columns": ["id", "name", "age", "country"],
    "missing_values": {"id": 0, "name": 1, "age": 0, "country": 0},
    "duplicate_rows": 0
  },
  "checks": {
    "row_count": {"passed": true, "message": "Found 5 rows"},
    "data_types": {"passed": false, "issues": [...]},
    "value_ranges": {"passed": true, "issues": []}
  },
  "overall_passed": false,
  "total_issues": 2
}
```

## ⚙️ Configuration

### Environment Variables
- `APP_ENV`: Environment indicator (dev/staging/prod) - controls warning banner display
- `BASE_URL_PATH`: Base URL path for reverse proxy deployments (optional)
  - **Empty** (default): App serves at root path `/` - use for dedicated domains
  - **`/path`**: App serves at `/path/` - use when NGINX strips path prefix
  - Example: `BASE_URL_PATH=/sageapp02` for https://domain.com/sageapp02/
- Additional configuration via `.env` file (see `.env.example`)

## 🔍 Data Quality Checks Details

### Row Count Check
- Validates minimum number of rows
- Configurable threshold
- Reports actual vs expected count

### Data Type Validation
- Supports: `int`, `float`, `str`, `bool`, `datetime`
- Attempts type conversion validation
- Reports type mismatches with sample data

### Value Range Validation
- **Numeric Ranges**: Set min/max bounds
- **Categorical Values**: Define allowed value sets
- **Custom Rules**: Flexible rule configuration
- Reports violating rows with details

### Missing Value Analysis
- Per-column missing value counts
- Percentage calculations
- Visual charts for missing data patterns

## 📊 Output Reports

### Summary Dashboard
| Metric | Value |
|--------|-------|
| Rows | 1,000 |
| Columns | 15 |
| Duplicates | 5 |
| Memory (MB) | 2.3 |

### Validation Results
- ✅ **Row Count**: PASSED (1,000 rows ≥ 100 minimum)
- ❌ **Data Types**: FAILED (2 type mismatches)
- ✅ **Value Ranges**: PASSED (all values in range)

### Issue Details
- **Type Mismatch**: Column 'age' expected int, found string
- **Range Violation**: Column 'salary' has 3 values > max(100000)
- **Invalid Values**: Column 'country' has invalid values: ['INVALID']

## 🐛 Troubleshooting

### Common Issues

**CSV upload errors**
- Check file encoding (try UTF-8, latin1, cp1252)
- Verify CSV format and delimiters
- Ensure file size < 200MB

**Unsupported format errors**
- Currently only CSV is supported
- JSON, Excel, and Parquet support coming soon
- Convert your data to CSV format for now

**MCP server connection issues**
- Verify dependencies are installed: `pip install -r mcp_requirements.txt`
- Check port configuration (default: 3002)
- Review application logs in terminal

**Environment banner not showing**
- Ensure `APP_ENV` is set in your `.env` file
- Try `APP_ENV=dev` for development
- Restart the application after changing `.env`

### Debug Commands
```bash
# Test container locally
docker run -p 3002:8002 -e APP_ENV=dev data-analyzer

# Check if virtual environment is activated
which python  # Should show path to venv

# Verify dependencies
pip list | grep streamlit
pip list | grep env-banner
```

## 🔒 Security Considerations

- **Data Privacy**: Files processed in-memory, not persisted
- **No External Storage**: Data never leaves local environment
- **Container Security**: Regular base image updates recommended
- **Dependency Updates**: Keep Python packages up to date

## 📈 Performance

### Performance Optimization
- Files processed in-memory for speed
- Pandas vectorized operations
- Efficient missing value detection
- Minimal memory footprint
- Recommended: 2GB+ RAM for large datasets

## 🧪 Testing

### Unit Tests
```bash
# Run MCP server tests
python -m pytest tests/test_mcp_server.py

# Run web app tests  
python -m pytest tests/test_web_app.py
```

### Integration Tests
```bash
# Test full pipeline
python tests/test_integration.py
```

### Load Testing
```bash
# Test with large CSV files
python tests/test_performance.py
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-check`
3. Add tests for new functionality
4. Commit changes: `git commit -am 'Add new validation check'`
5. Push branch: `git push origin feature/new-check`
6. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Documentation**: See inline code documentation
- **Azure Support**: Check Azure Container Apps documentation

## 🔄 Changelog

### v1.0.0
- Initial release with MCP server
- Streamlit web interface
- Azure Container Apps deployment
- Basic data quality checks
- Schema and rules validation

