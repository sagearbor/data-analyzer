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
import streamlit.components.v1 as components
from demo_dictionaries import DEMO_DICTIONARIES, get_demo_dictionary
from mermaid_renderer import render_mermaid
try:
    from streamlit_option_menu import option_menu
except ImportError:
    option_menu = None

# Configure Streamlit page
st.set_page_config(
    page_title="Data Quality Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class MCPClient:
    """Client to communicate with MCP server for data analysis"""
    
    def __init__(self, server_script_path: str = "mcp_server.py", use_real_mcp: bool = False):
        self.server_script_path = server_script_path
        self.use_real_mcp = use_real_mcp
    
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
                if self.use_real_mcp:
                    # Call the real MCP server
                    result = await self._call_real_mcp_server(request_data, debug=debug)
                else:
                    # Use simulation for development
                    result = self._simulate_mcp_call(request_data, debug=debug)
                return result
            finally:
                os.unlink(temp_file)
                
        except Exception as e:
            return {"error": str(e), "type": "mcp_error"}
    
    async def parse_data_dictionary(self, dictionary_content: str,
                                   format_hint: str = "auto",
                                   use_cache: bool = True,
                                   debug: bool = False) -> Dict:
        """Parse data dictionary using MCP server or simulation"""
        request_data = {
            "tool": "parse_data_dictionary",
            "arguments": {
                "dictionary_content": dictionary_content,
                "format_hint": format_hint,
                "use_cache": use_cache,
                "debug": debug
            }
        }

        if self.use_real_mcp:
            return await self._call_real_mcp_server(request_data, debug)
        else:
            return self._simulate_dictionary_parsing(dictionary_content, format_hint, debug)

    def _simulate_dictionary_parsing(self, dictionary_content: str, format_hint: str, debug: bool = False) -> Dict:
        """Simulate dictionary parsing for demo purposes"""
        try:
            # Check if it's PDF or other complex format that needs real LLM
            if format_hint in ['pdf', 'excel', 'xlsx', 'json']:
                return {
                    "success": False,
                    "error": f"Format '{format_hint}' requires Azure OpenAI for parsing. Please enable '🚀 Use Real MCP Server' in the sidebar to use your Azure OpenAI configuration.",
                    "validation": {
                        "is_valid": False,
                        "errors": [f"Enable 'Use Real MCP Server' in sidebar to parse {format_hint.upper()} files with Azure OpenAI"],
                        "warnings": ["Your Azure OpenAI credentials are configured in .env file"],
                        "confidence_score": 0.0
                    }
                }

            # For simulation, parse simple CSV/text dictionaries
            if format_hint in ['csv', 'text', 'auto'] or 'Column' in dictionary_content[:500] or 'Field' in dictionary_content[:500]:
                lines = dictionary_content.strip().split('\n')
                if len(lines) < 2:
                    return {"success": False, "error": "Dictionary too short"}

                # Simple CSV parsing simulation
                headers = [h.strip() for h in lines[0].split(',')]
                schema = {}
                rules = {}

                for line in lines[1:]:
                    if not line.strip():
                        continue
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 2:
                        col_name = parts[0]
                        col_type = parts[1].lower()

                        # Map types
                        type_map = {
                            'integer': 'int', 'int': 'int',
                            'float': 'float', 'decimal': 'float', 'real': 'float',
                            'string': 'str', 'text': 'str', 'varchar': 'str',
                            'boolean': 'bool', 'bool': 'bool',
                            'date': 'datetime', 'datetime': 'datetime'
                        }
                        schema[col_name] = type_map.get(col_type, 'str')

                        # Extract min/max if present
                        if len(parts) > 3:
                            if parts[3] and parts[3] != '':
                                try:
                                    rules.setdefault(col_name, {})['min'] = float(parts[3])
                                except:
                                    pass
                        if len(parts) > 4:
                            if parts[4] and parts[4] != '':
                                try:
                                    rules.setdefault(col_name, {})['max'] = float(parts[4])
                                except:
                                    pass

                # Generate a simple parser code for simulation
                parser_code = f"""def parse_dictionary(content: str) -> dict:
    # Auto-generated parser for {format_hint} dictionary
    lines = content.strip().split('\\n')
    headers = lines[0].split(',')

    columns = {{}}
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) >= 2:
            columns[parts[0]] = {{
                'type': parts[1],
                'required': len(parts) > 2 and parts[2].lower() == 'yes'
            }}

    return {{
        'columns': columns,
        'metadata': {{'source_format': '{format_hint}', 'total_columns': len(columns)}}
    }}
"""

                return {
                    "success": True,
                    "schema": schema,
                    "rules": rules,
                    "metadata": {
                        "source_format": format_hint,
                        "total_columns": len(schema)
                    },
                    "validation": {
                        "is_valid": True,
                        "errors": [],
                        "warnings": [],
                        "confidence_score": 0.75  # Simulation confidence
                    },
                    "confidence_score": 0.75,
                    "parser_code": parser_code  # Include simulated parser code
                }
            else:
                return {
                    "success": False,
                    "error": f"Format {format_hint} not supported in simulation mode. Use real MCP server.",
                    "validation": {
                        "is_valid": False,
                        "errors": ["Simulation only supports CSV dictionaries"],
                        "warnings": [],
                        "confidence_score": 0.0
                    }
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "validation": {
                    "is_valid": False,
                    "errors": [str(e)],
                    "warnings": [],
                    "confidence_score": 0.0
                }
            }

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
    
    async def _call_real_mcp_server(self, request_data: Dict, debug: bool = False) -> Dict:
        """Call the real MCP server using subprocess with proper MCP protocol"""
        try:
            import subprocess
            import json
            
            if debug:
                print("🔧 DEBUG: Calling real MCP server")
                print(f"🔧 DEBUG: Request: {request_data}")
            
            # Start the MCP server process
            process = subprocess.Popen(
                ["python", self.server_script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Step 1: Initialize the server
            init_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "roots": {
                            "listChanged": True
                        },
                        "sampling": {}
                    },
                    "clientInfo": {
                        "name": "data-analyzer-client",
                        "version": "1.0.0"
                    }
                }
            }
            
            # Step 2: Send initialized notification
            initialized_notification = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {}
            }
            
            # Step 3: Make the actual tool call
            tool_request = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": request_data["tool"],
                    "arguments": request_data["arguments"]
                }
            }
            
            # Combine all messages
            messages = [
                json.dumps(init_request),
                json.dumps(initialized_notification), 
                json.dumps(tool_request)
            ]
            
            input_data = "\n".join(messages) + "\n"
            
            if debug:
                print(f"🔧 DEBUG: Sending MCP messages:")
                for i, msg in enumerate(messages, 1):
                    print(f"🔧 DEBUG: Message {i}: {msg}")
            
            stdout, stderr = process.communicate(input=input_data, timeout=30)
            
            if debug:
                print(f"🔧 DEBUG: MCP server stdout: {stdout}")
                if stderr:
                    print(f"🔧 DEBUG: MCP server stderr: {stderr}")
            
            # Parse the responses (we expect multiple JSON objects)
            if stdout.strip():
                responses = []
                for line in stdout.strip().split('\n'):
                    if line.strip():
                        try:
                            responses.append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
                
                if debug:
                    print(f"🔧 DEBUG: Parsed responses: {responses}")
                
                # Look for the tool call response (should be the last one with id=2)
                for response in responses:
                    if response.get("id") == 2:  # Our tool call response
                        if "result" in response:
                            # Parse the result content
                            result_content = response["result"]["content"][0]["text"]
                            return json.loads(result_content)
                        elif "error" in response:
                            return {"error": response["error"]["message"], "type": "mcp_server_error"}
            
            return {"error": "No valid response from MCP server", "type": "mcp_communication_error"}
            
        except subprocess.TimeoutExpired:
            if debug:
                print("🔧 DEBUG: MCP server timeout")
            return {"error": "MCP server timeout", "type": "mcp_timeout_error"}
        except Exception as e:
            if debug:
                print(f"🔧 DEBUG: MCP server error: {e}")
            return {"error": str(e), "type": "mcp_call_error"}
    
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
def get_mcp_client(use_real_mcp: bool = False):
    return MCPClient(use_real_mcp=use_real_mcp)

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

def render_about_page():
    """Render the About page with tool description and architecture"""
    st.title("🆘 About Data Analyzer")

    # Tool Description
    st.markdown("""
    ## What is Data Analyzer?

    Data Analyzer is an AI-powered data quality validation tool that helps you:

    🔍 **Validate Data Quality**
    - Check data types, ranges, and allowed values
    - Identify missing values and duplicates
    - Generate comprehensive statistics

    🤖 **Parse Data Dictionaries with AI**
    - Upload dictionaries in various formats (CSV, JSON, REDCap, Clinical)
    - AI generates deterministic Python code to parse dictionaries
    - Code is cached for reuse (1 year cache lifetime)

    📊 **Support Multiple Formats**
    - Currently supports CSV with plans for JSON, Excel, Parquet
    - Extensible architecture for adding new formats

    ⚙️ **Configure Validation Rules**
    - Set expected data types for columns
    - Define min/max ranges for numeric data
    - Specify allowed values for categorical data

    ## How It Works

    The tool uses **Azure OpenAI (GPT-4o-mini)** to generate Python code that parses data dictionaries.
    This approach provides:
    - ✅ **Deterministic results** - Same input always produces same output
    - 🔍 **Verifiable code** - Generated code can be inspected and tested
    - 📦 **Cacheable parsers** - Reuse parsers for similar dictionaries
    - 📊 **High confidence** - ~95% accuracy through validation
    """)

    # Architecture Diagram
    st.markdown("## 🏭 System Architecture")
    st.markdown("""The diagram below shows how data flows through the system and where AI comes into play:""")

    # Create columns for the flow diagram
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        ### 📁 Input Stage
        - **Data Files**
          - CSV (supported)
          - JSON (planned)
          - Excel (planned)

        - **Dictionaries**
          - CSV format
          - JSON format
          - REDCap format
          - Clinical specs
        """)

    with col2:
        st.markdown("""
        ### 🤖 AI Processing
        - **Azure OpenAI**
          - GPT-4o-mini model
          - Code generation
          - Python parsers

        - **Caching**
          - 1-year lifetime
          - File-based cache
          - Instant retrieval
        """)

    with col3:
        st.markdown("""
        ### 📊 Output Stage
        - **Validation**
          - Type checking
          - Range validation
          - Allowed values

        - **Results**
          - Dashboard view
          - Issue tracking
          - Statistics
        """)

    # Show the detailed flow
    with st.expander("🔄 View Detailed Data Flow"):
        st.markdown("""
        ```
        1. User uploads data file (CSV)
           ↓
        2. User uploads/selects data dictionary
           ↓
        3. Dictionary sent to Azure OpenAI
           ↓
        4. LLM generates Python parser code
           ↓
        5. Code executed in sandbox (600s timeout)
           ↓
        6. Schema & rules extracted
           ↓
        7. Cache parser for reuse (1 year)
           ↓
        8. Apply rules to data validation
           ↓
        9. Display results in dashboard
        ```
        """)

    # Display Mermaid diagram
    st.markdown("### 📊 Interactive Architecture Diagram")

    try:
        with open('assets/data_flow_diagram.mmd', 'r') as f:
            mermaid_code = f.read()

        # Render Mermaid diagram visually using our custom renderer
        render_mermaid(mermaid_code, height=700)

        # Option to view/copy the code
        with st.expander("🔍 View Diagram Source Code"):
            st.code(mermaid_code, language='mermaid')

            # Copy button with actual functionality
            if st.button("📋 Copy to Clipboard", key="copy_mermaid"):
                st.code(mermaid_code)
                st.info("📎 Select and copy the code above")

    except FileNotFoundError:
        st.error("Architecture diagram file not found. Please ensure assets/data_flow_diagram.mmd exists.")

    # Key Features
    st.markdown("""
    ## 🌟 Key Features

    ### LLM Integration
    - Uses Azure OpenAI to generate parser code
    - Code generation instead of direct parsing
    - 600-second timeout for complex operations

    ### Caching System
    - File-based cache with 1-year lifetime
    - Cache key based on dictionary structure
    - Instant retrieval for similar dictionaries

    ### Security
    - Subprocess isolation for code execution
    - No network access in parser code
    - Future: Docker containerization planned

    ### Demo Data
    - Culturally diverse datasets (Western, Asian, Clinical)
    - Matching demo dictionaries
    - Intentional errors for testing validation
    """)

    # Footer
    st.markdown("---")
    st.markdown("""
    **Version**: 1.0.0 | **License**: MIT | **GitHub**: [data-analyzer](https://github.com/yourusername/data-analyzer)
    """)

def render_home_page():
    """Render the main Home page with data analysis functionality"""
    st.title("📊 Data Quality Analyzer")
    st.markdown("Upload a CSV file and configure validation rules to check data quality")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a data file",
            type=['csv', 'json', 'xlsx', 'xls'],
            help="Upload a data file to analyze (CSV, JSON, or Excel)"
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
        
        # MCP mode toggle
        use_real_mcp = st.checkbox(
            "🚀 Use Real MCP Server (Azure OpenAI)",
            value=False,
            help="Enable to use your Azure OpenAI API for parsing PDFs and complex dictionaries. Credentials are in .env file."
        )
        if not use_real_mcp:
            st.caption("💡 Enable above to use Azure OpenAI for PDFs")
        
        st.divider()
        
        # Quick example data with format selection
        st.markdown("### 📝 Example Data")
        demo_format = st.selectbox(
            "Select demo data format:",
            ["CSV - Western", "CSV - Asian", "JSON - Mixed", "CSV - Clinical Trial"],
            key="demo_format_selector"
        )

        if st.button("Load Example Data"):
            example_data = ""

            if demo_format == "CSV - Western":
                # Western names with diverse data types and some errors
                example_data = """employee_id,first_name,last_name,age,salary,hire_date,last_login_datetime,bonus_percentage,department,is_active,skills,email,phone
1001,John,Smith,28,75000.50,2023-01-15,2024-01-20 09:30:00,15.5,Engineering,true,"Python;SQL;Docker",john.smith@techcorp.com,+1-555-0101
1002,Sarah,Johnson,32,82000.00,2022-06-10,2024-01-19 14:45:30,18.2,Marketing,true,"Analytics;SEO;Content",sarah.j@techcorp.com,+1-555-0102
1003,Michael,Williams,-5,68000.75,2021-11-22,2024-01-18 08:15:45,12.0,Sales,false,"CRM;Excel;Communication",m.williams@techcorp.com,INVALID_PHONE
1004,Emma,Brown,29,91000.00,invalid_date,2024-01-20 16:20:00,22.5,Engineering,true,"Java;AWS;Kubernetes",emma.brown@techcorp.com,+1-555-0104
1005,David,Martinez,45,105000.00,2020-03-08,2024-01-19 11:30:00,25.0,Management,true,"Leadership;Strategy;Finance",d.martinez@techcorp.com,+1-555-0105
1006,Lisa,Anderson,31,79000.50,2023-02-14,NOT_A_DATETIME,17.8,HR,true,"Recruiting;HRIS;Training",lisa.a@techcorp.com,+1-555-0106
1007,Robert,Taylor,256,72000.00,2022-08-30,2024-01-20 13:45:00,14.5,Finance,true,"Excel;SAP;Audit",r.taylor@techcorp.com,+1-555-0107"""

            elif demo_format == "CSV - Asian":
                # Asian names with different data patterns
                example_data = """staff_id,given_name,family_name,age,monthly_salary,join_date,last_activity,performance_score,dept_code,active_status,certifications,work_email,mobile
2001,Wei,Zhang,26,8500.00,2023-03-20,2024-01-20T10:45:00Z,4.2,DEV,1,"AWS-SA;CCNA;PMP",wei.zhang@company.cn,+86-138-0000-0001
2002,Yuki,Tanaka,30,9200.50,2022-09-15,2024-01-19T15:30:00Z,4.5,MKT,1,"Google-Ads;HubSpot",y.tanaka@company.jp,+81-90-1234-5678
2003,Priya,Sharma,28,7800.00,2021-12-01,2024-01-18T09:00:00Z,3.8,OPS,1,"Six-Sigma;Lean",priya.sharma@company.in,+91-98765-43210
2004,Min-jun,Kim,999,10500.00,2020-06-18,INVALID_TIME,4.7,DEV,1,"Java;Spring;React",minjun.kim@company.kr,+82-10-9876-5432
2005,Mei,Chen,32,8900.75,2023-01-10,2024-01-20T14:20:00Z,-1.5,FIN,0,"CPA;Excel-Expert",mei.chen@company.tw,MISSING_PHONE
2006,Raj,Patel,27,8200.00,bad_date,2024-01-19T11:15:00Z,4.1,HR,1,"PHR;Workday",raj.patel@company.in,+91-99999-88888
2007,Hiroshi,Yamamoto,35,11000.00,2019-04-22,2024-01-20T16:00:00Z,4.9,MGT,1,"MBA;PgMP;Agile",h.yamamoto@company.jp,+81-80-3333-4444"""

            elif demo_format == "JSON - Mixed":
                # JSON format with nested data and mixed cultures
                example_data = '''[
  {"id": 3001, "name": {"first": "Maria", "last": "Garcia"}, "age": 29, "salary": 72000, "hired": "2023-02-28", "active": true, "scores": [85, 90, 88], "department": "Research"},
  {"id": 3002, "name": {"first": "Ahmed", "last": "Hassan"}, "age": 31, "salary": 78000, "hired": "2022-11-15", "active": true, "scores": [92, 88, 91], "department": "Engineering"},
  {"id": 3003, "name": {"first": "Olga", "last": "Ivanova"}, "age": -10, "salary": 69000, "hired": "2021-07-20", "active": false, "scores": [78, 82], "department": "Quality"},
  {"id": 3004, "name": {"first": "João", "last": "Silva"}, "age": 34, "salary": "NOT_A_NUMBER", "hired": "invalid", "active": true, "scores": [88, 85, 87, 90], "department": "Sales"},
  {"id": 3005, "name": {"first": "Fatima", "last": "Al-Rashid"}, "age": 27, "salary": 71500, "hired": "2023-05-10", "active": "maybe", "scores": null, "department": "Marketing"}
]'''

            elif demo_format == "CSV - Clinical Trial":
                # Clinical trial data with medical/pharmaceutical fields
                example_data = """subject_id,site_id,enrollment_date,visit_date,age,gender,bmi,treatment_arm,adverse_event,lab_value,compliance_pct,completed_study,protocol_version
S001,SITE01,2023-01-15,2023-02-15,45,M,24.5,Treatment,None,120.5,95.5,Y,2.1
S002,SITE01,2023-01-20,2023-02-20,52,F,28.3,Placebo,Mild Headache,135.2,88.0,Y,2.1
S003,SITE02,2023-02-01,invalid_date,38,M,22.1,Treatment,None,118.0,92.5,Y,2.1
S004,SITE02,2023-02-05,2023-03-05,-5,F,31.2,Placebo,Nausea,142.8,76.5,N,2.1
S005,SITE03,2023-02-10,2023-03-10,61,X,26.7,Treatment,None,999.9,101.5,Y,2.0
S006,SITE03,bad_enrollment,2023-03-15,49,M,29.8,Treatment,Dizziness,128.3,82.0,Y,2.1
S007,SITE04,2023-02-20,2023-03-20,55,F,INVALID_BMI,Placebo,None,131.7,90.0,Y,2.1"""

            st.session_state.example_data = example_data
            # Store the format type for proper parsing
            if "JSON" in demo_format:
                st.session_state.example_format = "json"
            else:
                st.session_state.example_format = "csv"
            st.success(f"{demo_format} data loaded!")

        st.divider()

        # Data Dictionary Upload Section
        st.markdown("### 📚 Data Dictionary")

        # Option to load demo dictionary
        dict_demo_format = st.selectbox(
            "Load demo dictionary:",
            ["None"] + list(DEMO_DICTIONARIES.keys()),
            key="dict_demo_selector",
            help="Load a demo dictionary that matches the demo datasets"
        )

        if dict_demo_format != "None":
            action = st.radio(
                "Action:",
                ["Load Only", "Load & Parse", "Load, Parse & Apply"],
                key="demo_dict_action",
                horizontal=True,
                index=2  # Default to full workflow
            )
            if st.button("📁 Execute", key="load_demo_dict"):
                st.session_state['dict_content'] = get_demo_dictionary(dict_demo_format)
                st.session_state['dict_format'] = dict_demo_format
                st.session_state['dict_source'] = 'demo'
                st.session_state['dict_action'] = action
                st.success(f"Loaded {dict_demo_format} dictionary!")

                # Auto-trigger parsing if requested
                if action in ["Load & Parse", "Load, Parse & Apply"]:
                    st.session_state['auto_parse'] = True
                if action == "Load, Parse & Apply":
                    st.session_state['auto_apply'] = True

        # File upload option
        dict_file = st.file_uploader(
            "Or upload your own:",
            type=['csv', 'xlsx', 'json', 'txt', 'pdf'],
            key="dict_uploader",
            help="Upload a data dictionary to auto-populate schema and rules. Supports CSV, Excel, JSON, text, PDF, and REDCap formats."
        )

        if dict_file:
            upload_action = st.radio(
                "Action on upload:",
                ["Load Only", "Load & Parse", "Load, Parse & Apply"],
                key="upload_dict_action",
                horizontal=True,
                index=2  # Default to full workflow
            )

            # Read and store uploaded file content
            file_extension = dict_file.name.split('.')[-1].lower()

            if file_extension == 'pdf':
                # Extract text from PDF
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(dict_file)
                    dict_content = ""
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        dict_content += page.extract_text()
                    st.success(f"Extracted {len(dict_content)} characters from PDF")
                except ImportError:
                    st.error("PyPDF2 not installed. Please install it to process PDF files.")
                    st.code("pip install PyPDF2")
                    dict_content = ""
                except Exception as e:
                    st.error(f"Error reading PDF: {str(e)}")
                    dict_content = ""
            else:
                # Read text-based files
                dict_content = dict_file.read()
                try:
                    dict_content = dict_content.decode('utf-8')
                except UnicodeDecodeError:
                    dict_content = dict_content.decode('latin-1')

            st.session_state['dict_content'] = dict_content
            st.session_state['dict_format'] = file_extension
            st.session_state['dict_source'] = 'upload'
            st.session_state['dict_filename'] = dict_file.name
            st.session_state['dict_action'] = upload_action

            # Auto-trigger parsing if requested
            if upload_action in ["Load & Parse", "Load, Parse & Apply"]:
                st.session_state['auto_parse'] = True
            if upload_action == "Load, Parse & Apply":
                st.session_state['auto_apply'] = True

        # Show parse button if we have dictionary content
        if st.session_state.get('dict_content'):
            # Check if we should auto-parse
            should_parse = st.session_state.get('auto_parse', False)

            if not should_parse:
                parse_button = st.button(
                    "🤖 Parse Dictionary",
                    key="parse_dict_btn",
                    use_container_width=True,
                    type="primary"
                )
                should_parse = parse_button

            if should_parse:
                # Clear auto-parse flag
                if 'auto_parse' in st.session_state:
                    del st.session_state['auto_parse']
                with st.spinner("Parsing data dictionary with AI..."):
                    # Get dictionary content from session state
                    dict_content = st.session_state.get('dict_content', '')

                    # Determine format hint
                    if st.session_state.get('dict_source') == 'demo':
                        # For demo dictionaries, use CSV format
                        format_hint = 'csv'
                    else:
                        # For uploaded files, use file extension
                        format_hint = st.session_state.get('dict_format', 'auto')
                        if format_hint == 'txt':
                            format_hint = 'text'
                        elif format_hint == 'xlsx':
                            format_hint = 'excel'

                    # Get MCP client
                    client = get_mcp_client(use_real_mcp=use_real_mcp)

                    # Parse dictionary
                    result = asyncio.run(client.parse_data_dictionary(
                        dictionary_content=dict_content,
                        format_hint=format_hint,
                        use_cache=True,
                        debug=debug_mode
                    ))

                    if result.get('success'):
                        # Store parsed results in session state
                        st.session_state['parsed_schema'] = result.get('schema', {})
                        st.session_state['parsed_rules'] = result.get('rules', {})
                        st.session_state['dict_metadata'] = result.get('metadata', {})
                        st.session_state['parser_validation'] = result.get('validation', {})
                        st.session_state['parser_code'] = result.get('parser_code', '')  # Store parser code for download
                        st.session_state['parse_complete'] = True  # Mark parse as complete

                        confidence = result.get('confidence_score', 0.0)
                        st.success(f"✅ Dictionary parsed successfully!")
                        st.metric("Confidence Score", f"{confidence:.1%}")

                        # Show validation info
                        validation = result.get('validation', {})
                        if validation.get('warnings'):
                            with st.expander(f"⚠️ {len(validation['warnings'])} Warnings"):
                                for warning in validation['warnings']:
                                    st.warning(warning)

                    else:
                        st.error(f"❌ Failed to parse dictionary: {result.get('error', 'Unknown error')}")
                        if debug_mode and result.get('validation'):
                            with st.expander("Debug Info"):
                                st.json(result['validation'])

            # Show parsed results if available
            if st.session_state.get('parse_complete') and st.session_state.get('parsed_schema'):
                # Preview extracted schema
                with st.expander("📋 Extracted Schema", expanded=True):
                    st.json(st.session_state['parsed_schema'])

                # Preview extracted rules
                if st.session_state.get('parsed_rules'):
                    with st.expander("⚙️ Extracted Rules", expanded=True):
                        st.json(st.session_state['parsed_rules'])

                # Check if we should auto-apply
                should_apply = st.session_state.get('auto_apply', False)

                # Apply button (or auto-apply)
                if not should_apply:
                    should_apply = st.button("✅ Apply to Validation", key="apply_dict", type="primary", use_container_width=True)

                if should_apply:
                    # Clear auto-apply flag
                    if 'auto_apply' in st.session_state:
                        del st.session_state['auto_apply']
                    # Convert to UI format and apply
                    if st.session_state.get('parsed_schema'):
                        schema_entries = []
                        for col_name, col_type in st.session_state['parsed_schema'].items():
                            schema_entries.append({
                                'column': col_name,
                                'type': col_type
                            })
                        st.session_state.schema_entries = schema_entries

                    if st.session_state.get('parsed_rules'):
                        rules_entries = []
                        for col_name, col_rules in st.session_state['parsed_rules'].items():
                            # Determine rule type based on available constraints
                            if 'allowed' in col_rules:
                                rule_type = 'allowed_values'
                                config = {'allowed': col_rules['allowed']}
                            elif 'min' in col_rules or 'max' in col_rules:
                                rule_type = 'range'
                                config = {}
                                if 'min' in col_rules:
                                    config['min'] = col_rules['min']
                                if 'max' in col_rules:
                                    config['max'] = col_rules['max']
                            else:
                                continue  # Skip if no recognized rule type

                            rules_entries.append({
                                'column': col_name,
                                'rule_type': rule_type,
                                'config': config
                            })
                        st.session_state.rules_entries = rules_entries

                    st.success("✅ Schema and rules applied to validation!")
                    st.rerun()

            # Download parser code button (show if code exists)
            if st.session_state.get('parser_code'):
                st.divider()
                st.markdown("#### 💾 Download Parser")
                # Determine filename for download
                if st.session_state.get('dict_source') == 'demo':
                    base_name = st.session_state.get('dict_format', 'dictionary').replace(' ', '_').replace('-', '_')
                else:
                    base_name = st.session_state.get('dict_filename', 'dictionary').split('.')[0]

                # Download the generated parser code
                st.download_button(
                    label="📥 Download Parser Code (Python)",
                    data=st.session_state['parser_code'],
                    file_name=f"parser_{base_name}.py",
                    mime="text/x-python",
                    key="download_parser_code",
                    use_container_width=True,
                    help="Download the AI-generated Python code that parses this dictionary format"
                )

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
                        client = get_mcp_client(use_real_mcp=use_real_mcp)
                        
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
        file_content = st.session_state.example_data
        file_format = st.session_state.get('example_format', 'csv')

        st.info(f"Using example {file_format.upper()} data - configure schema and rules below, then run analysis")
        
        # Editable example data
        with st.expander("📄 Example Data (Editable)", expanded=True):
            try:
                # Allow user to edit the example data
                edited_data = st.text_area(
                    "Edit your data here:",
                    value=file_content,
                    height=200,
                    help=f"Edit the {file_format.upper()} data directly. Changes will be used for analysis."
                )

                # Update the data if it changed
                if edited_data != file_content:
                    st.session_state.example_data = edited_data
                    file_content = edited_data

                # Show preview of the edited data
                st.subheader("Preview:")
                if file_format == 'json':
                    preview_df = pd.read_json(io.StringIO(file_content))
                else:
                    preview_df = pd.read_csv(io.StringIO(file_content))
                st.dataframe(preview_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error reading {file_format.upper()} data: {str(e)}")
                if file_format == 'json':
                    st.info("Please check your JSON format. Make sure it's valid JSON syntax.")
                else:
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
                    client = get_mcp_client(use_real_mcp=use_real_mcp)
                    
                    # Run analysis
                    results = asyncio.run(client.analyze_data(
                        data_content=file_content,
                        file_format=file_format,
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

def main():
    """Main Streamlit application with navigation"""

    # Custom CSS for better styling
    st.markdown("""
    <style>
    /* Navbar styling */
    section[data-testid="stSidebar"] {
        top: 0;
    }
    .navbar {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create navigation menu
    st.markdown("## 📊 Data Quality Analyzer")

    if option_menu:
        # Use streamlit-option-menu for a professional navbar
        selected = option_menu(
            menu_title=None,  # No title
            options=["🏠 Home", "ℹ️ About"],
            icons=None,  # Icons already in option text
            menu_icon=None,
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {
                    "padding": "5px",
                    "background-color": "#f0f2f6",
                    "border-radius": "10px",
                    "margin-bottom": "1rem"
                },
                "icon": {"display": "none"},  # Hide icons since we have them in text
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "center",
                    "margin": "0px 5px",
                    "padding": "10px 20px",
                    "border-radius": "5px",
                    "--hover-color": "#e0e2e6"
                },
                "nav-link-selected": {
                    "background-color": "#667eea",
                    "color": "white",
                    "font-weight": "bold"
                },
            }
        )
        # Remove emoji from selected for comparison
        selected = selected.replace("🏠 ", "").replace("ℹ️ ", "")
    else:
        # Fallback to radio buttons if option_menu not available
        col1, col2, _ = st.columns([1, 1, 4])
        with col1:
            if st.button("🏠 Home", use_container_width=True, type="primary"):
                selected = "Home"
        with col2:
            if st.button("ℹ️ About", use_container_width=True):
                selected = "About"
        selected = st.session_state.get("page", "Home")

    st.markdown("---")

    # Render the selected page
    if selected == "About":
        render_about_page()
    else:
        render_home_page()

if __name__ == "__main__":
    main()
