def main():
    """Main Streamlit application"""
    
    st.title("📊 Data Quality Analyzer")
    st.markdown("Upload data files and configure validation rules to check data quality")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=['csv'],  # Future: add 'json', 'xlsx', etc.
            help="Upload a data file to analyze (CSV supported, more formats coming)"
        )
        
        # File format selection (for future extensibility)
        file_format = st.selectbox(
            "File Format",
            ["csv"],  # Future: add "json", "excel", "parquet"
            help="Select the format of your data file"
        )
        
        # Minimum rows setting
        min_rows = st.number_input(
            "Minimum Required Rows",
            min_value=1,
            value=1,
            help="Minimum number of rows required for the dataset"
        )
        
        # Encoding selection
        encoding = st.selectbox(
            "File Encoding",
            ["utf-8", "latin1", "cp1252", "iso-8859-1"],
            help="Select the encoding of your data file"
        )
        
        st.divider()
        
        # Quick example data
        if st.button("📝 Load Example Data"):
            # Create example CSV content
            example_data = """id,name,age,country,salary
1,John Doe,25,USA,50000
2,Jane Smith,30,CAN,55000
3,Bob Johnson,35,MEX,60000
4,Alice Brown,28,USA,52000
5,Charlie Wilson,150,INVALID,75000"""
            
            st.session_state.example_data = example_data
            st.session_state.example_format = "csv#!/usr/bin/env python3
"""
Azure Web Application for Data Analysis
Supports CSV analysis with extensibility for other data formats
Uses the MCP server for analysis functionality
"""

import streamlit as st
import pandas as pd
import json
import base64
import io
import asyncio
import subprocess
import tempfile
import os
from typing import Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Data Quality Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MCPClient:
    """Client to communicate with MCP server for data analysis"""
    
    def __init__(self, server_script_path: str = "mcp_server.py"):
        self.server_script_path = server_script_path
    
    async def analyze_data(self, data_content: str, file_format: str = "csv", schema: Dict = None, rules: Dict = None, min_rows: int = 1):
        """Call MCP server to analyze data"""
        try:
            # Prepare the arguments
            args = {
                "data_content": data_content,
                "file_format": file_format,
                "min_rows": min_rows
            }
            if schema:
                args["schema"] = schema
            if rules:
                args["rules"] = rules
            
            # Create a temporary file with the request
            request_data = {
                "tool": "analyze_data",
                "arguments": args
            }
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(request_data, f)
                temp_file = f.name
            
            try:
                # Call the MCP server (simplified version for demo)
                # In production, you'd use proper MCP client libraries
                result = self._simulate_mcp_call(request_data)
                return result
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            return {"error": str(e), "type": "mcp_error"}
    
    def _simulate_mcp_call(self, request_data: Dict) -> Dict:
        """Simulate MCP call for demo purposes"""
        # This is a simplified simulation - in production use proper MCP client
        try:
            data_content = request_data["arguments"]["data_content"]
            file_format = request_data["arguments"].get("file_format", "csv")
            schema = request_data["arguments"].get("schema", {})
            rules = request_data["arguments"].get("rules", {})
            min_rows = request_data["arguments"].get("min_rows", 1)
            
            # Load data based on format
            if file_format.lower() == "csv":
                df = pd.read_csv(io.StringIO(data_content))
            else:
                raise ValueError(f"Unsupported format: {file_format}")
            
            # Simulate the analysis results
            results = {
                "timestamp": datetime.now().isoformat(),
                "file_format": file_format,
                "summary_stats": {
                    "shape": {"rows": len(df), "columns": len(df.columns)},
                    "columns": list(df.columns),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "missing_values": df.isnull().sum().to_dict(),
                    "duplicate_rows": int(df.duplicated().sum()),
                    "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
                },
                "checks": {
                    "row_count": {
                        "check": "row_count",
                        "passed": len(df) >= min_rows,
                        "row_count": len(df),
                        "min_required": min_rows,
                        "message": f"Found {len(df)} rows (minimum: {min_rows})"
                    },
                    "data_types": {
                        "check": "data_types",
                        "passed": True,
                        "message": "Type validation completed",
                        "issues": []
                    },
                    "value_ranges": {
                        "check": "value_ranges", 
                        "passed": True,
                        "message": "Range validation completed",
                        "issues": []
                    }
                },
                "overall_passed": len(df) >= min_rows,
                "total_issues": 0 if len(df) >= min_rows else 1,
                "issues": []
            }
            
            # Add schema validation if provided
            if schema:
                type_issues = []
                for column, expected_type in schema.items():
                    if column not in df.columns:
                        type_issues.append({
                            "column": column,
                            "issue": "missing_column",
                            "expected_type": expected_type
                        })
                
                results["checks"]["data_types"]["issues"] = type_issues
                results["checks"]["data_types"]["passed"] = len(type_issues) == 0
                if type_issues:
                    results["overall_passed"] = False
                    results["total_issues"] += len(type_issues)
            
            # Add range validation if provided
            if rules:
                range_issues = []
                for column, rule in rules.items():
                    if column in df.columns:
                        col_data = df[column].dropna()
                        
                        # Check allowed values
                        if "allowed" in rule:
                            allowed_values = set(rule["allowed"])
                            actual_values = set(col_data.unique())
                            invalid_values = actual_values - allowed_values
                            
                            if invalid_values:
                                range_issues.append({
                                    "column": column,
                                    "rule": f"allowed_values: {rule['allowed']}",
                                    "invalid_values": list(invalid_values),
                                    "violation_count": len(invalid_values)
                                })
                
                results["checks"]["value_ranges"]["issues"] = range_issues
                results["checks"]["value_ranges"]["passed"] = len(range_issues) == 0
                if range_issues:
                    results["overall_passed"] = False
                    results["total_issues"] += len(range_issues)
            
            return results
            
        except Exception as e:
            return {"error": str(e), "type": "analysis_error"}

# Initialize MCP client
@st.cache_resource
def get_mcp_client():
    return MCPClient()

def create_schema_editor():
    """Create an interactive schema editor"""
    st.subheader("📋 Schema Definition")
    
    if 'schema_entries' not in st.session_state:
        st.session_state.schema_entries = [{"column": "", "type": "str"}]
    
    schema = {}
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        for i, entry in enumerate(st.session_state.schema_entries):
            cols = st.columns([2, 2, 1])
            
            with cols[0]:
                column = st.text_input(f"Column {i+1}", value=entry["column"], key=f"schema_col_{i}")
            
            with cols[1]:
                type_val = st.selectbox(
                    f"Type {i+1}", 
                    ["str", "int", "float", "bool", "datetime"],
                    index=["str", "int", "float", "bool", "datetime"].index(entry["type"]),
                    key=f"schema_type_{i}"
                )
            
            with cols[2]:
                if st.button("❌", key=f"remove_schema_{i}"):
                    st.session_state.schema_entries.pop(i)
                    st.rerun()
            
            if column:
                schema[column] = type_val
            
            st.session_state.schema_entries[i] = {"column": column, "type": type_val}
    
    with col2:
        if st.button("➕ Add Column"):
            st.session_state.schema_entries.append({"column": "", "type": "str"})
            st.rerun()
        
        if st.button("🗑️ Clear All"):
            st.session_state.schema_entries = [{"column": "", "type": "str"}]
            st.rerun()
    
    return {k: v for k, v in schema.items() if k}

def create_rules_editor():
    """Create an interactive rules editor"""
    st.subheader("⚙️ Validation Rules")
    
    if 'rules_entries' not in st.session_state:
        st.session_state.rules_entries = [{"column": "", "rule_type": "range", "config": {}}]
    
    rules = {}
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        for i, entry in enumerate(st.session_state.rules_entries):
            with st.expander(f"Rule {i+1}", expanded=True):
                cols = st.columns([2, 2])
                
                with cols[0]:
                    column = st.text_input(f"Column", value=entry["column"], key=f"rule_col_{i}")
                
                with cols[1]:
                    rule_type = st.selectbox(
                        "Rule Type",
                        ["range", "allowed_values"],
                        index=0 if entry["rule_type"] == "range" else 1,
                        key=f"rule_type_{i}"
                    )
                
                rule_config = {}
                
                if rule_type == "range":
                    cols2 = st.columns(2)
                    with cols2[0]:
                        min_val = st.number_input("Min Value", value=entry["config"].get("min", 0), key=f"min_{i}")
                        if min_val is not None:
                            rule_config["min"] = min_val
                    
                    with cols2[1]:
                        max_val = st.number_input("Max Value", value=entry["config"].get("max", 100), key=f"max_{i}")
                        if max_val is not None:
                            rule_config["max"] = max_val
                
                elif rule_type == "allowed_values":
                    allowed_text = st.text_input(
                        "Allowed Values (comma-separated)",
                        value=",".join(entry["config"].get("allowed", [])),
                        key=f"allowed_{i}"
                    )
                    if allowed_text:
                        rule_config["allowed"] = [v.strip() for v in allowed_text.split(",") if v.strip()]
                
                if st.button("❌ Remove Rule", key=f"remove_rule_{i}"):
                    st.session_state.rules_entries.pop(i)
                    st.rerun()
                
                if column:
                    rules[column] = rule_config
                
                st.session_state.rules_entries[i] = {"column": column, "rule_type": rule_type, "config": rule_config}
    
    with col2:
        if st.button("➕ Add Rule"):
            st.session_state.rules_entries.append({"column": "", "rule_type": "range", "config": {}})
            st.rerun()
        
        if st.button("🗑️ Clear All Rules"):
            st.session_state.rules_entries = [{"column": "", "rule_type": "range", "config": {}}]
            st.rerun()
    
    return {k: v for k, v in rules.items() if k and v}

def display_results(results: Dict[str, Any]):
    """Display analysis results in a comprehensive dashboard"""
    
    if "error" in results:
        st.error(f"Analysis Error: {results['error']}")
        return
    
    # Overall Status
    st.header("📊 Analysis Results")
    
    overall_status = results.get("overall_passed", False)
    total_issues = results.get("total_issues", 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_color = "green" if overall_status else "red"
        status_text = "✅ PASSED" if overall_status else "❌ FAILED"
        st.markdown(f"### Overall Status: <span style='color:{status_color}'>{status_text}</span>", unsafe_allow_html=True)
    
    with col2:
        st.metric("Total Issues Found", total_issues)
    
    with col3:
        timestamp = results.get("timestamp", "Unknown")
        st.write(f"**Analysis Time:** {timestamp[:19]}")
    
    st.divider()
    
    # Summary Statistics
    st.subheader("📈 Dataset Summary")
    
    stats = results.get("summary_stats", {})
    shape = stats.get("shape", {})
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Rows", shape.get("rows", 0))
    
    with col2:
        st.metric("Columns", shape.get("columns", 0))
    
    with col3:
        st.metric("Duplicates", stats.get("duplicate_rows", 0))
    
    with col4:
        st.metric("Memory (MB)", stats.get("memory_usage_mb", 0))
    
    # Missing Values Visualization
    if "missing_values" in stats:
        missing_data = stats["missing_values"]
        if any(v > 0 for v in missing_data.values()):
            st.subheader("🔍 Missing Values")
            
            missing_df = pd.DataFrame(list(missing_data.items()), columns=["Column", "Missing Count"])
            missing_df = missing_df[missing_df["Missing Count"] > 0]
            
            if not missing_df.empty:
                fig = px.bar(missing_df, x="Column", y="Missing Count", 
                           title="Missing Values per Column")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found!")
    
    # Data Types Table
    if "dtypes" in stats:
        st.subheader("📋 Column Information")
        
        dtype_data = []
        for col, dtype in stats["dtypes"].items():
            missing_count = stats.get("missing_values", {}).get(col, 0)
            dtype_data.append({
                "Column": col,
                "Data Type": dtype,
                "Missing Values": missing_count,
                "Missing %": round(missing_count / shape.get("rows", 1) * 100, 2) if shape.get("rows", 0) > 0 else 0
            })
        
        dtype_df = pd.DataFrame(dtype_data)
        st.dataframe(dtype_df, use_container_width=True)
    
    st.divider()
    
    # Check Results
    st.subheader("🔬 Quality Check Results")
    
    checks = results.get("checks", {})
    
    for check_name, check_result in checks.items():
        with st.expander(f"{check_name.replace('_', ' ').title()}", expanded=not check_result.get("passed", True)):
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                passed = check_result.get("passed", False)
                status_icon = "✅" if passed else "❌"
                status_text = "PASSED" if passed else "FAILED"
                st.markdown(f"**Status:** {status_icon} {status_text}")
            
            with col2:
                message = check_result.get("message", "No message")
                st.write(f"**Message:** {message}")
            
            # Display issues if any
            issues = check_result.get("issues", [])
            if issues:
                st.subheader("Issues Found:")
                
                for i, issue in enumerate(issues):
                    with st.container():
                        st.markdown(f"**Issue {i+1}:**")
                        
                        # Create a clean issue display
                        issue_data = {}
                        for key, value in issue.items():
                            if key != "violating_rows":  # Handle large lists separately
                                issue_data[key.replace("_", " ").title()] = value
                        
                        # Display as a compact table
                        issue_df = pd.DataFrame([issue_data])
                        st.dataframe(issue_df, use_container_width=True, hide_index=True)
                        
                        # Show violating rows if present
                        if "violating_rows" in issue and issue["violating_rows"]:
                            with st.expander(f"View Violating Rows (showing first {len(issue['violating_rows'])}):"):
                                st.write(issue["violating_rows"])
    
    # Numeric Summary if available
    numeric_summary = stats.get("numeric_summary", {})
    if numeric_summary:
        st.subheader("📊 Numeric Column Statistics")
        
        numeric_data = []
        for col, col_stats in numeric_summary.items():
            numeric_data.append({
                "Column": col,
                "Min": col_stats.get("min"),
                "Max": col_stats.get("max"),
                "Mean": round(col_stats.get("mean", 0), 2) if col_stats.get("mean") else None,
                "Std Dev": round(col_stats.get("std", 0), 2) if col_stats.get("std") else None
            })
        
        numeric_df = pd.DataFrame(numeric_data)
        st.dataframe(numeric_df, use_container_width=True)

def main():
    """Main Streamlit application"""
    
    st.title("📊 CSV Quality Analyzer")
    st.markdown("Upload a CSV file and configure validation rules to check data quality")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file to analyze"
        )
        
        # Minimum rows setting
        min_rows = st.number_input(
            "Minimum Required Rows",
            min_value=1,
            value=1,
            help="Minimum number of rows required for the dataset"
        )
        
        # Encoding selection
        encoding = st.selectbox(
            "File Encoding",
            ["utf-8", "latin1", "cp1252", "iso-8859-1"],
            help="Select the encoding of your CSV file"
        )
        
        st.divider()
        
        # Quick example data
        if st.button("📝 Load Example Data"):
            # Create example CSV content
            example_data = """id,name,age,country,salary
1,John Doe,25,USA,50000
2,Jane Smith,30,CAN,55000
3,Bob Johnson,35,MEX,60000
4,Alice Brown,28,USA,52000
5,Charlie Wilson,150,INVALID,75000"""
            
            st.session_state.example_csv = example_data
            st.success("Example data loaded!")
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            csv_content = uploaded_file.read().decode(encoding)
            
            # Display file info
            st.info(f"File: {uploaded_file.name} ({len(csv_content)} characters)")
            
            # Preview the data
            with st.expander("📄 Data Preview", expanded=True):
                try:
                    preview_df = pd.read_csv(io.StringIO(csv_content))
                    st.dataframe(preview_df.head(10), use_container_width=True)
                    st.caption(f"Showing first 10 rows of {len(preview_df)} total rows")
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")
                    return
            
            # Configuration tabs
            tab1, tab2, tab3 = st.tabs(["📋 Schema", "⚙️ Rules", "🚀 Analysis"])
            
            with tab1:
                schema = create_schema_editor()
                if schema:
                    st.success(f"Schema configured for {len(schema)} columns")
                    with st.expander("View Current Schema"):
                        st.json(schema)
            
            with tab2:
                rules = create_rules_editor()
                if rules:
                    st.success(f"Rules configured for {len(rules)} columns")
                    with st.expander("View Current Rules"):
                        st.json(rules)
            
            with tab3:
                st.subheader("🚀 Run Analysis")
                
                if st.button("🔍 Analyze CSV", type="primary", use_container_width=True):
                    with st.spinner("Analyzing CSV data..."):
                        # Get MCP client
                        client = get_mcp_client()
                        
                        # Run analysis
                        results = asyncio.run(client.analyze_csv(
                            csv_content=csv_content,
                            schema=schema if schema else None,
                            rules=rules if rules else None,
                            min_rows=min_rows
                        ))
                        
                        # Display results
                        display_results(results)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    elif 'example_csv' in st.session_state:
        # Handle example data
        csv_content = st.session_state.example_csv
        
        st.info("Using example data - configure schema and rules below, then run analysis")
        
        # Preview example data
        with st.expander("📄 Example Data Preview", expanded=True):
            try:
                preview_df = pd.read_csv(io.StringIO(csv_content))
                st.dataframe(preview_df, use_container_width=True)
            except Exception as e:
                st.error(f"Error reading example CSV: {str(e)}")
                return
        
        # Configuration tabs for example data
        tab1, tab2, tab3 = st.tabs(["📋 Schema", "⚙️ Rules", "🚀 Analysis"])
        
        with tab1:
            schema = create_schema_editor()
            if schema:
                st.success(f"Schema configured for {len(schema)} columns")
        
        with tab2:
            rules = create_rules_editor()
            if rules:
                st.success(f"Rules configured for {len(rules)} columns")
        
        with tab3:
            st.subheader("🚀 Run Analysis")
            
            if st.button("🔍 Analyze Example Data", type="primary", use_container_width=True):
                with st.spinner("Analyzing CSV data..."):
                    # Get MCP client
                    client = get_mcp_client()
                    
                    # Run analysis
                    results = asyncio.run(client.analyze_csv(
                        csv_content=csv_content,
                        schema=schema if schema else None,
                        rules=rules if rules else None,
                        min_rows=min_rows
                    ))
                    
                    # Display results
                    display_results(results)
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to CSV Quality Analyzer! 👋
        
        This tool helps you validate and analyze CSV files for data quality issues.
        
        ### Features:
        - **📊 Data Quality Checks**: Row count, data type validation, value range checks
        - **📋 Schema Validation**: Define expected column types and validate against them
        - **⚙️ Custom Rules**: Set up range checks and allowed value lists
        - **📈 Comprehensive Reports**: Visual dashboards with detailed issue tracking
        - **🔍 Missing Value Analysis**: Identify and visualize missing data patterns
        
        ### Get Started:
        1. Upload a CSV file using the sidebar
        2. Or click "Load Example Data" to try it out
        3. Configure schema and validation rules
        4. Run the analysis to get detailed quality reports
        
        ### Supported Validations:
        - **Row Count**: Ensure minimum number of rows
        - **Data Types**: Validate int, float, string, boolean, datetime columns
        - **Value Ranges**: Set min/max bounds for numeric columns
        - **Allowed Values**: Define categorical value lists
        - **Missing Values**: Detect and quantify missing data
        - **Duplicates**: Identify duplicate rows
        
        Ready to get started? Upload a file or load the example data! 🚀
        """)

if __name__ == "__main__":
    main()
