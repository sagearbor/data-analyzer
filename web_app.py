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
    
    async def analyze_data(self, data_content: str, file_format: str = "csv", schema: Dict = None, rules: Dict = None, min_rows: int = 1, debug: bool = False):
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
                result = self._simulate_mcp_call(request_data, debug=debug)
                return result
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            return {"error": str(e), "type": "mcp_error"}
    
    def _auto_detect_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Auto-detect column types including dates"""
        detected_types = {}
        
        for col in df.columns:
            col_data = df[col].dropna()
            if len(col_data) == 0:
                detected_types[col] = "unknown"
                continue
                
            # Check if it's numeric
            try:
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                if not numeric_data.isna().all():
                    # Check if all values are integers
                    if (numeric_data % 1 == 0).all():
                        detected_types[col] = "int"
                    else:
                        detected_types[col] = "float"
                    continue
            except:
                pass
            
            # Check if it's datetime
            try:
                # Try to parse as datetime, but only if it has valid-looking patterns
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    datetime_parsed = pd.to_datetime(col_data, errors='coerce')
                
                # Check if more than 50% of non-null values were successfully parsed
                # and at least one value looks like a date
                valid_dates = datetime_parsed.notna().sum()
                if valid_dates > len(col_data) * 0.5 and valid_dates > 0:
                    # Double-check with strict parsing to catch invalid dates
                    sample_values = col_data.head(3).tolist()
                    if any(any(char in str(val) for char in ['-', '/', ':', ' ']) for val in sample_values):
                        detected_types[col] = "datetime"
                        continue
            except:
                pass
            
            # Check if it's boolean
            unique_values = set(str(v).lower() for v in col_data.unique())
            if unique_values.issubset({'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}):
                detected_types[col] = "bool"
                continue
            
            # Default to string
            detected_types[col] = "str"
        
        return detected_types
    
    def _debug_print(self, message: str, debug: bool):
        """Helper function to print debug messages only when debug mode is on"""
        if debug:
            print(message)
    
    def _simulate_mcp_call(self, request_data: Dict, debug: bool = False) -> Dict:
        """Simulate MCP call for demo purposes"""
        # This is a simplified simulation - in production use proper MCP client
        try:
            data_content = request_data["arguments"]["data_content"]
            file_format = request_data["arguments"].get("file_format", "csv")
            schema = request_data["arguments"].get("schema", {})
            rules = request_data["arguments"].get("rules", {})
            min_rows = request_data["arguments"].get("min_rows", 1)
            
            # DEBUG OUTPUT
            self._debug_print("🔧 DEBUG: Starting MCP simulation", debug)
            self._debug_print(f"🔧 DEBUG: Schema received: {schema}", debug)
            self._debug_print(f"🔧 DEBUG: Rules received: {rules}", debug)
            self._debug_print(f"🔧 DEBUG: Data preview: {data_content[:100]}...", debug)
            
            # Load data based on format
            if file_format.lower() == "csv":
                df = pd.read_csv(io.StringIO(data_content))
                self._debug_print(f"🔧 DEBUG: Loaded DataFrame shape: {df.shape}", debug)
                self._debug_print(f"🔧 DEBUG: Columns: {list(df.columns)}", debug)
                self._debug_print(f"🔧 DEBUG: DataFrame dtypes:\n{df.dtypes}", debug)
                if 'hire_date' in df.columns:
                    self._debug_print(f"🔧 DEBUG: hire_date values: {df['hire_date'].tolist()}", debug)
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
                "auto_detected_types": self._auto_detect_types(df),
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
            
            # Add schema validation - use provided schema OR auto-detected types
            auto_detected_types = self._auto_detect_types(df)
            self._debug_print(f"🔧 DEBUG: Auto-detected types: {auto_detected_types}", debug)
            
            # Combine manual schema with auto-detected types for validation
            validation_schema = auto_detected_types.copy()
            if schema:
                validation_schema.update(schema)  # Manual schema overrides auto-detection
                self._debug_print(f"🔧 DEBUG: Manual schema provided, merged with auto-detection", debug)
            
            self._debug_print(f"🔧 DEBUG: Final validation schema: {validation_schema}", debug)
            
            if validation_schema:
                self._debug_print(f"🔧 DEBUG: Processing schema validation for {len(validation_schema)} columns", debug)
                type_issues = []
                for column, expected_type in validation_schema.items():
                    self._debug_print(f"🔧 DEBUG: Validating column '{column}' as type '{expected_type}'", debug)
                    
                    if column not in df.columns:
                        self._debug_print(f"🔧 DEBUG: Column '{column}' missing from DataFrame", debug)
                        type_issues.append({
                            "column": column,
                            "issue": "missing_column",
                            "expected_type": expected_type
                        })
                        continue
                    
                    # Type validation
                    col_data = df[column].dropna()
                    self._debug_print(f"🔧 DEBUG: Column '{column}' has {len(col_data)} non-null values", debug)
                    self._debug_print(f"🔧 DEBUG: Sample values: {col_data.head(3).tolist()}", debug)
                    
                    if len(col_data) > 0:
                        try:
                            if expected_type == "int":
                                pd.to_numeric(col_data, errors='raise', downcast='integer')
                                self._debug_print(f"🔧 DEBUG: Column '{column}' passed int validation", debug)
                            elif expected_type == "float":
                                pd.to_numeric(col_data, errors='raise')
                                self._debug_print(f"🔧 DEBUG: Column '{column}' passed float validation", debug)
                            elif expected_type == "datetime":
                                self._debug_print(f"🔧 DEBUG: Testing datetime validation for '{column}'", debug)
                                parsed_dates = pd.to_datetime(col_data, errors='coerce')
                                self._debug_print(f"🔧 DEBUG: Parsed dates: {parsed_dates.tolist()}", debug)
                                self._debug_print(f"🔧 DEBUG: Any NA values: {parsed_dates.isna().any()}", debug)
                                self._debug_print(f"🔧 DEBUG: NA positions: {parsed_dates.isna().tolist()}", debug)
                                if parsed_dates.isna().any():
                                    self._debug_print(f"🔧 DEBUG: DATETIME VALIDATION FAILED - will create issue", debug)
                                    raise ValueError(f"Invalid datetime values found")
                                else:
                                    self._debug_print(f"🔧 DEBUG: Column '{column}' passed datetime validation", debug)
                            elif expected_type == "bool":
                                # Check if values can be converted to bool
                                unique_vals = set(str(v).lower() for v in col_data.unique())
                                if not unique_vals.issubset({'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}):
                                    raise ValueError("Invalid boolean values")
                                self._debug_print(f"🔧 DEBUG: Column '{column}' passed bool validation", debug)
                            # str type always passes
                        except Exception as e:
                            self._debug_print(f"🔧 DEBUG: VALIDATION EXCEPTION for '{column}': {e}", debug)
                            # For datetime issues, provide more specific information
                            if expected_type == "datetime":
                                invalid_dates = col_data[pd.to_datetime(col_data, errors='coerce').isna()]
                                self._debug_print(f"🔧 DEBUG: Invalid dates found: {invalid_dates.tolist()}", debug)
                                issue = {
                                    "column": column,
                                    "issue": "datetime_validation_failed",
                                    "expected_type": expected_type,
                                    "actual_type": str(df[column].dtype),
                                    "invalid_values": invalid_dates.tolist(),
                                    "valid_sample": col_data[pd.to_datetime(col_data, errors='coerce').notna()].head(2).tolist(),
                                    "description": f"Found {len(invalid_dates)} invalid date values in column '{column}'"
                                }
                                type_issues.append(issue)
                                self._debug_print(f"🔧 DEBUG: Added datetime issue: {issue}", debug)
                            else:
                                issue = {
                                    "column": column,
                                    "issue": "type_mismatch",
                                    "expected_type": expected_type,
                                    "actual_type": str(df[column].dtype),
                                    "sample_values": col_data.head(3).tolist()
                                }
                                type_issues.append(issue)
                                self._debug_print(f"🔧 DEBUG: Added type mismatch issue: {issue}", debug)
                
                results["checks"]["data_types"]["issues"] = type_issues
                results["checks"]["data_types"]["passed"] = len(type_issues) == 0
                results["checks"]["data_types"]["message"] = f"Type validation: {len(type_issues)} issues found"
                self._debug_print(f"🔧 DEBUG: Type validation complete - {len(type_issues)} issues found", debug)
                self._debug_print(f"🔧 DEBUG: Type issues: {type_issues}", debug)
                if type_issues:
                    results["overall_passed"] = False
                    results["total_issues"] += len(type_issues)
                    self._debug_print(f"🔧 DEBUG: Updated total issues to: {results['total_issues']}", debug)
            else:
                self._debug_print("🔧 DEBUG: No schema provided, skipping type validation", debug)
            
            # Add range validation if provided
            if rules:
                range_issues = []
                for column, rule in rules.items():
                    if column in df.columns:
                        col_data = df[column].dropna()
                        
                        # Check numeric ranges (min/max)
                        if "min" in rule or "max" in rule:
                            try:
                                numeric_data = pd.to_numeric(col_data, errors='coerce')
                                
                                if "min" in rule:
                                    violations = numeric_data < rule["min"]
                                    if violations.any():
                                        violating_rows = df[violations].index.tolist()
                                        range_issues.append({
                                            "column": column,
                                            "rule": f"min >= {rule['min']}",
                                            "violation_count": violations.sum(),
                                            "violating_rows": violating_rows[:10]
                                        })
                                
                                if "max" in rule:
                                    violations = numeric_data > rule["max"]
                                    if violations.any():
                                        violating_rows = df[violations].index.tolist()
                                        range_issues.append({
                                            "column": column,
                                            "rule": f"max <= {rule['max']}",
                                            "violation_count": violations.sum(),
                                            "violating_rows": violating_rows[:10]
                                        })
                            except:
                                range_issues.append({
                                    "column": column,
                                    "rule": "numeric_range",
                                    "issue": "column_not_numeric"
                                })
                        
                        # Check allowed values
                        if "allowed" in rule:
                            allowed_values = set(rule["allowed"])
                            actual_values = set(col_data.unique())
                            invalid_values = actual_values - allowed_values
                            
                            if invalid_values:
                                mask = col_data.isin(invalid_values)
                                violating_rows = df[mask].index.tolist()
                                range_issues.append({
                                    "column": column,
                                    "rule": f"allowed_values: {rule['allowed']}",
                                    "invalid_values": list(invalid_values),
                                    "violation_count": mask.sum(),
                                    "violating_rows": violating_rows[:10]
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
    
    # Data Types Table with Auto-Detection
    if "dtypes" in stats:
        st.subheader("📋 Column Information")
        
        dtype_data = []
        auto_detected = stats.get("auto_detected_types", {})
        
        for col, dtype in stats["dtypes"].items():
            missing_count = stats.get("missing_values", {}).get(col, 0)
            detected_type = auto_detected.get(col, "unknown")
            
            # Add emoji indicators
            type_emoji = {
                "int": "🔢", "float": "🔢", "str": "📝", 
                "datetime": "📅", "bool": "☑️", "unknown": "❓"
            }
            
            dtype_data.append({
                "Column": col,
                "Current Type": dtype,
                "Auto-Detected": f"{type_emoji.get(detected_type, '❓')} {detected_type}",
                "Missing Values": missing_count,
                "Missing %": round(missing_count / shape.get("rows", 1) * 100, 2) if shape.get("rows", 0) > 0 else 0
            })
        
        dtype_df = pd.DataFrame(dtype_data)
        st.dataframe(dtype_df, use_container_width=True)
        
        # Add helpful note about auto-detection
        st.info("💡 **Auto-Detection**: The 'Auto-Detected' column shows suggested data types based on content analysis. Use these suggestions when defining your schema!")
    
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
                        # Special handling for datetime issues
                        if issue.get("issue") == "datetime_validation_failed":
                            st.markdown(f"**🚨 Issue {i+1}: Invalid Date Values**")
                            st.error(issue.get("description", "Date validation failed"))
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Invalid Values:**")
                                invalid_vals = issue.get("invalid_values", [])
                                for val in invalid_vals:
                                    st.code(f"❌ '{val}'")
                            
                            with col2:
                                st.markdown("**Valid Examples:**")
                                valid_vals = issue.get("valid_sample", [])
                                for val in valid_vals:
                                    st.code(f"✅ '{val}'")
                        else:
                            st.markdown(f"**Issue {i+1}:**")
                            
                            # Create a clean issue display
                            issue_data = {}
                            for key, value in issue.items():
                                if key not in ["violating_rows", "invalid_values", "valid_sample", "description"]:  # Handle special keys separately
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
        
        # Debug mode toggle
        debug_mode = st.checkbox(
            "🔧 Debug Mode",
            value=False,
            help="Enable debug output in browser console (F12 → Console)"
        )
        
        st.divider()
        
        # Quick example data
        if st.button("📝 Load Example Data"):
            # Create example CSV content with date field
            example_data = """id,name,age,country,salary,hire_date
1,John Doe,25,USA,50000,2023-01-15
2,Jane Smith,30,CAN,55000,2022-03-20
3,Bob Johnson,35,MEX,60000,2021-07-10
4,Alice Brown,28,USA,52000,invalid_date
5,Charlie Wilson,150,INVALID,75000,2020-12-05"""
            
            st.session_state.example_data = example_data
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
                        results = asyncio.run(client.analyze_data(
                            data_content=csv_content,
                            file_format="csv",
                            schema=schema if schema else None,
                            rules=rules if rules else None,
                            min_rows=min_rows,
                            debug=debug_mode
                        ))
                        
                        # Display results
                        display_results(results)
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
    elif 'example_data' in st.session_state:
        # Handle example data
        csv_content = st.session_state.example_data
        
        st.info("Using example data - configure schema and rules below, then run analysis")
        
        # Editable example data
        with st.expander("📄 Example Data (Editable)", expanded=True):
            try:
                # Allow user to edit the example data
                edited_data = st.text_area(
                    "Edit your data here:",
                    value=csv_content,
                    height=200,
                    help="Edit the CSV data directly. Changes will be used for analysis."
                )
                
                # Update the data if it changed
                if edited_data != csv_content:
                    st.session_state.example_data = edited_data
                    csv_content = edited_data
                
                # Show preview of the edited data
                st.subheader("Preview:")
                preview_df = pd.read_csv(io.StringIO(csv_content))
                st.dataframe(preview_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error reading CSV data: {str(e)}")
                st.info("Please check your CSV format. Make sure it has proper headers and comma separation.")
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
                    results = asyncio.run(client.analyze_data(
                        data_content=csv_content,
                        file_format="csv",
                        schema=schema if schema else None,
                        rules=rules if rules else None,
                        min_rows=min_rows,
                        debug=debug_mode
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
