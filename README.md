# Data Analyzer - Azure MCP Server & Web Application

A comprehensive data quality analysis tool built with Azure MCP (Model Context Protocol) server and Streamlit web interface. Currently supports CSV analysis with an extensible architecture for future data formats (JSON, Excel, Parquet, etc.). Based on your [csvChecker](https://github.com/sagearbor/csvChecker) repository with enhanced cloud capabilities.

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
- **JSON**: Coming soon 🔄
- **Excel**: Planned 📋
- **Parquet**: Planned 📋

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
- Azure CLI installed and configured
- Docker installed
- Python 3.11+
- Azure subscription with Container Apps enabled

### 1. Clone and Setup
```bash
git clone <your-repo>
cd data-analyzer
chmod +x deploy.sh
```

### 2. Deploy to Azure
```bash
# Deploy with default settings
./deploy.sh deploy

# Deploy with custom app name
APP_NAME=mydata ./deploy.sh deploy

# Deploy to different region
LOCATION=westus2 ./deploy.sh deploy
```

### 3. Access Application
After deployment, the script will provide your application URL:
```
Application URL: https://your-app.region.azurecontainerapps.io
```

## 📁 Project Structure

```
data-analyzer/
├── mcp_server.py           # MCP server implementation
├── web_app.py              # Streamlit web application
├── requirements.txt        # Web app dependencies
├── mcp_requirements.txt    # MCP server dependencies
├── Dockerfile              # Container configuration
├── deploy.sh               # Azure deployment script
├── azure_config.yaml       # Azure resource templates
└── README.md               # This file
```

## 🔧 Local Development

### Run MCP Server Locally
```bash
# Install dependencies
pip install -r mcp_requirements.txt

# Run MCP server
python mcp_server.py
```

### Run Web App Locally
```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run web_app.py
```

### Docker Development
```bash
# Build image
docker build -t data-analyzer .

# Run container
docker run -p 8501:8501 data-analyzer
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
| Variable | Description | Default |
|----------|-------------|---------|
| `APP_NAME` | Application name | `data-analyzer` |
| `RESOURCE_GROUP` | Azure resource group | `rg-{APP_NAME}` |
| `LOCATION` | Azure region | `eastus` |
| `CONTAINER_REGISTRY` | ACR name | `{APP_NAME}registry` |

### Azure Resources Created
- **Resource Group**: Contains all resources
- **Container Registry**: Stores Docker images
- **Log Analytics Workspace**: Application logging
- **Container Apps Environment**: Runtime environment
- **Container App**: The web application

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

**Deployment fails with ACR permissions**
```bash
# Enable admin user on ACR
az acr update --name $CONTAINER_REGISTRY --admin-enabled true
```

**Container app won't start**
```bash
# Check logs
az containerapp logs show --name csv-analyzer-web --resource-group rg-csv-analyzer
```

**CSV upload errors**
- Check file encoding (try UTF-8, latin1, cp1252)
- Verify CSV format and delimiters
- Ensure file size < 200MB

**Unsupported format errors**
- Currently only CSV is supported
- JSON, Excel, and Parquet support coming soon
- Convert your data to CSV format for now

**MCP server connection issues**
- Verify container has required dependencies
- Check port configuration (8501)
- Review application logs

### Debug Commands
```bash
# View deployment info
./deploy.sh info

# Check container status
az containerapp show --name data-analyzer-web --resource-group rg-data-analyzer

# View application logs
az containerapp logs show --name data-analyzer-web --resource-group rg-data-analyzer --follow

# Test container locally
docker run -p 8501:8501 data-analyzer
```

## 🔒 Security Considerations

- **Data Privacy**: Files processed in-memory, not persisted
- **Access Control**: Configure Azure AD authentication if needed
- **Network Security**: HTTPS enabled by default
- **Container Security**: Regular base image updates recommended

## 📈 Scaling & Performance

### Auto-scaling Configuration
- **Min Replicas**: 1
- **Max Replicas**: 10
- **Scale Trigger**: HTTP request volume
- **CPU/Memory**: 1 CPU, 2GB RAM per instance

### Performance Optimization
- Files processed in-memory for speed
- Pandas vectorized operations
- Efficient missing value detection
- Minimal memory footprint

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

---

## Directory Structure
data-analyzer/
├── README.md
├── Dockerfile
├── deploy.sh
├── requirements.txt
├── mcp_requirements.txt
├── azure_config.yaml
├── mcp_server.py
├── web_app.py
├── .gitignore
├── .dockerignore
└── docs/
    └── API.md

