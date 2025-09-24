# Multi-Format Support Update

## ✅ Changes Made

### 1. **Restored All Demo Data Formats**
- ✅ CSV - Western (CSV format)
- ✅ CSV - Asian (CSV format)
- ✅ JSON - Mixed (JSON format with nested data)
- ✅ CSV - Clinical Trial (CSV format)

The JSON demo data is properly handled as JSON, not incorrectly parsed as CSV.

### 2. **Multi-Format Data File Support**
The main file uploader now accepts:
- **CSV** - Comma-separated values
- **JSON** - JavaScript Object Notation
- **Excel** (.xlsx, .xls) - Microsoft Excel files

Each format is properly detected and parsed using the appropriate pandas function:
- CSV: `pd.read_csv()`
- JSON: `pd.read_json()`
- Excel: `pd.read_excel()`

### 3. **PDF Dictionary Support**
- ✅ Added PDF to accepted file types for dictionary upload
- ✅ PDF text extraction using PyPDF2
- ✅ Clear messaging about Azure OpenAI requirement

**How PDF Processing Works:**
1. PDF uploaded → Text extracted using PyPDF2
2. Extracted text sent to Azure OpenAI for parsing
3. LLM generates Python code to parse the dictionary
4. Code executed to extract schema and rules

### 4. **Azure OpenAI Integration**
- **Simulation Mode**: Only handles simple CSV/text dictionaries
- **Real MCP Mode**: Uses Azure OpenAI for complex formats

**To Use Azure OpenAI:**
1. Enable "🚀 Use Real MCP Server (Azure OpenAI)" in sidebar
2. Your Azure credentials from .env file will be used
3. Supports PDF, JSON, Excel, and complex text formats

### 5. **Clear User Guidance**
When users try to parse PDFs in simulation mode, they get:
```
❌ Format 'pdf' requires Azure OpenAI for parsing.
Please enable '🚀 Use Real MCP Server' in the sidebar
to use your Azure OpenAI configuration.
```

## 📋 Supported Formats Summary

### Data Files (for analysis)
| Format | Extension | Support Level |
|--------|-----------|--------------|
| CSV | .csv | ✅ Full |
| JSON | .json | ✅ Full |
| Excel | .xlsx, .xls | ✅ Full |

### Dictionary Files (for schema/rules)
| Format | Extension | Simulation | Azure OpenAI |
|--------|-----------|------------|--------------|
| CSV | .csv | ✅ Basic | ✅ Advanced |
| Text | .txt | ✅ Basic | ✅ Advanced |
| JSON | .json | ❌ | ✅ Full |
| Excel | .xlsx | ❌ | ✅ Full |
| PDF | .pdf | ❌ | ✅ Full |

## 🚀 How to Use

### For CSV/Text Dictionaries
1. Upload dictionary file
2. Works in simulation mode
3. Basic parsing without LLM

### For PDF/JSON/Excel Dictionaries
1. Enable "🚀 Use Real MCP Server (Azure OpenAI)" in sidebar
2. Upload dictionary file
3. Azure OpenAI will parse and generate code
4. Full extraction of schema and rules

## 🔧 Dependencies

### Required for PDF Support
```bash
pip install PyPDF2
```

### Required for Azure OpenAI
- Configure in `.env` file:
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_CHAT_DEPLOYMENT_NAME`

## 📝 Example Workflow

### With PDF Dictionary
1. Check ✅ "Use Real MCP Server (Azure OpenAI)"
2. Upload PDF dictionary
3. Select "Load, Parse & Apply"
4. System will:
   - Extract text from PDF
   - Send to Azure OpenAI
   - Generate parser code
   - Extract schema/rules
   - Apply to validation

### With JSON Data
1. Upload JSON file for analysis
2. System automatically detects format
3. Parses with `pd.read_json()`
4. Displays in data preview
5. Run analysis as normal

## 💡 Tips

- **PDF Quality**: OCR'd PDFs work best with clear text
- **Large Files**: Excel files are converted to CSV internally
- **Performance**: Enable caching for repeated dictionary formats
- **Debugging**: Enable Debug Mode to see parser output

## 🎯 Benefits

1. **Flexibility**: Support for all common data formats
2. **Intelligence**: Azure OpenAI handles complex dictionaries
3. **Efficiency**: Automatic format detection
4. **User-Friendly**: Clear guidance on when to use Azure OpenAI
5. **Professional**: Handles real-world data dictionaries including PDFs