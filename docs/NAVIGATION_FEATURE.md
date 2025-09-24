# Navigation Feature Documentation

## Overview

The Data Analyzer now includes a navigation bar with Home and About pages to improve user experience and provide better documentation of the tool's capabilities.

## Navigation Structure

### 🏠 Home Page
The main data analysis interface containing:
- CSV file upload
- Schema configuration
- Rules editor
- Data dictionary parser
- Analysis results dashboard

### ℹ️ About Page
Comprehensive documentation including:
- Tool description and capabilities
- System architecture overview
- Data flow visualization
- Key features and specifications

## Implementation Details

### Navigation Bar
- Located at the top of every page
- Two buttons: Home and About
- Active page highlighted with primary button style
- Session state manages current page

### Page Routing
```python
# Navigation state management
if "page" not in st.session_state:
    st.session_state.page = "home"

# Page rendering
if st.session_state.page == "about":
    render_about_page()
else:
    render_home_page()
```

## About Page Features

### 1. Tool Description
- What Data Analyzer does
- Key capabilities
- AI integration explanation

### 2. Architecture Visualization
Three-column layout showing:
- **Input Stage**: Data sources and dictionary formats
- **AI Processing**: LLM code generation and caching
- **Output Stage**: Validation and results

### 3. Detailed Data Flow
Step-by-step flow from upload to results:
1. User uploads data file
2. User uploads/selects dictionary
3. Dictionary sent to Azure OpenAI
4. LLM generates Python parser
5. Code executed in sandbox
6. Schema & rules extracted
7. Parser cached for reuse
8. Rules applied to data
9. Results displayed

### 4. Mermaid Diagram
- Complete system architecture in Mermaid format
- Stored in `assets/data_flow_diagram.mmd`
- Shows all components and connections
- Includes LLM integration points

### 5. Key Features
Documentation of:
- LLM Integration details
- Caching system specs
- Security measures
- Demo data availability

## Files Modified

1. **web_app.py**
   - Added `render_about_page()` function
   - Added `render_home_page()` function
   - Updated `main()` with navigation logic
   - Added navigation buttons

2. **assets/data_flow_diagram.mmd**
   - Complete Mermaid diagram
   - Shows data sources, LLM processing, validation pipeline
   - Color-coded components

3. **assets/architecture.md**
   - Detailed architecture documentation
   - Performance metrics
   - Security measures
   - Supported formats

## Usage

1. Run the application:
   ```bash
   streamlit run web_app.py
   ```

2. Navigate between pages:
   - Click "🏠 Home" for data analysis
   - Click "ℹ️ About" for documentation

3. On the About page:
   - Review tool capabilities
   - Understand the architecture
   - View data flow diagram
   - Copy Mermaid code for external viewing

## Benefits

1. **Better User Onboarding**
   - New users can understand the tool quickly
   - Clear explanation of AI integration

2. **Architecture Transparency**
   - Shows where LLM is used
   - Explains caching and performance

3. **Documentation Access**
   - In-app documentation
   - No need for external docs

4. **Professional Presentation**
   - Clean navigation interface
   - Organized information structure

## Future Enhancements

- Add more pages (Settings, Help, Examples)
- Interactive Mermaid diagram rendering
- Video tutorials
- API documentation page
- Changelog/version history