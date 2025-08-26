#!/usr/bin/env python3
"""
Azure MCP Server for Data Analysis
Supports CSV analysis with extensibility for other data formats
Based on csvChecker functionality with future format support
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import io
import base64
from datetime import datetime
import tempfile
import os

# MCP SDK imports
from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types

# Data quality check modules (adapted from csvChecker, extensible for other formats)
class DataLoader:
    """Load and validate data files (CSV, future: JSON, Excel, etc.)"""
    
    @staticmethod
    def load_csv(file_path_or_content: Union[str, bytes, io.StringIO], **kwargs) -> pd.DataFrame:
        """Load CSV from file path, bytes, or StringIO"""
        try:
            if isinstance(file_path_or_content, bytes):
                # Decode bytes to string and create StringIO
                content = file_path_or_content.decode('utf-8')
                return pd.read_csv(io.StringIO(content), **kwargs)
            elif isinstance(file_path_or_content, str):
                if os.path.exists(file_path_or_content):
                    return pd.read_csv(file_path_or_content, **kwargs)
                else:
                    # Treat as CSV content string
                    return pd.read_csv(io.StringIO(file_path_or_content), **kwargs)
            else:
                return pd.read_csv(file_path_or_content, **kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load CSV: {str(e)}")
    
    @staticmethod
    def load_data(file_path_or_content: Union[str, bytes, io.StringIO], file_format: str = "csv", **kwargs) -> pd.DataFrame:
        """Load data from various formats - extensible for future formats"""
        if file_format.lower() == "csv":
            return DataLoader.load_csv(file_path_or_content, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}. Currently supported: CSV")

class QualityChecker:
    """Perform data quality checks on structured data (CSV, future: other formats)"""
    
    def __init__(self, df: pd.DataFrame, schema: Optional[Dict] = None, rules: Optional[Dict] = None):
        self.df = df
        self.schema = schema or {}
        self.rules = rules or {}
        self.issues = []
        
    def check_row_count(self, min_rows: int = 1) -> Dict[str, Any]:
        """Check if DataFrame has minimum required rows"""
        row_count = len(self.df)
        passed = row_count >= min_rows
        
        result = {
            "check": "row_count",
            "passed": passed,
            "row_count": row_count,
            "min_required": min_rows,
            "message": f"Found {row_count} rows (minimum: {min_rows})"
        }
        
        if not passed:
            self.issues.append(result)
            
        return result
    
    def check_data_types(self) -> Dict[str, Any]:
        """Validate column data types against schema or auto-detected types"""
        # Use provided schema OR auto-detected types for validation
        auto_detected_types = self._auto_detect_types()
        
        # Combine manual schema with auto-detected types for validation
        validation_schema = auto_detected_types.copy()
        if self.schema:
            validation_schema.update(self.schema)  # Manual schema overrides auto-detection
        
        if not validation_schema:
            return {
                "check": "data_types",
                "passed": True,
                "message": "No schema or auto-detection available, skipping type validation"
            }
        
        type_issues = []
        total_checks = 0
        
        for column, expected_type in validation_schema.items():
            if column not in self.df.columns:
                type_issues.append({
                    "column": column,
                    "issue": "missing_column",
                    "expected_type": expected_type
                })
                continue
                
            total_checks += 1
            actual_dtype = str(self.df[column].dtype)
            
            # Type mapping
            type_map = {
                "int": ["int64", "int32", "int16", "int8"],
                "float": ["float64", "float32"],
                "str": ["object", "string"],
                "bool": ["bool"],
                "datetime": ["datetime64[ns]"]
            }
            
            if expected_type in type_map:
                if actual_dtype not in type_map[expected_type]:
                    # Try to convert and see if it fails
                    try:
                        if expected_type == "int":
                            pd.to_numeric(self.df[column], errors='raise')
                        elif expected_type == "float":
                            pd.to_numeric(self.df[column], errors='raise')
                        elif expected_type == "datetime":
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                parsed_dates = pd.to_datetime(self.df[column].dropna(), errors='coerce')
                            if parsed_dates.isna().any():
                                raise ValueError(f"Invalid datetime values found")
                    except:
                        # For datetime issues, provide more specific information
                        if expected_type == "datetime":
                            col_data = self.df[column].dropna()
                            import warnings
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                invalid_dates = col_data[pd.to_datetime(col_data, errors='coerce').isna()]
                            type_issues.append({
                                "column": column,
                                "issue": "datetime_validation_failed",
                                "expected_type": expected_type,
                                "actual_type": actual_dtype,
                                "invalid_values": invalid_dates.tolist(),
                                "valid_sample": col_data[pd.to_datetime(col_data, errors='coerce').notna()].head(2).tolist(),
                                "description": f"Found {len(invalid_dates)} invalid date values in column '{column}'"
                            })
                        else:
                            type_issues.append({
                                "column": column,
                                "issue": "type_mismatch",
                                "expected_type": expected_type,
                                "actual_type": actual_dtype,
                                "sample_values": self.df[column].head(3).tolist()
                            })
        
        passed = len(type_issues) == 0
        result = {
            "check": "data_types",
            "passed": passed,
            "total_columns_checked": total_checks,
            "issues_found": len(type_issues),
            "issues": type_issues,
            "message": f"Type validation: {len(type_issues)} issues found"
        }
        
        if not passed:
            self.issues.extend(type_issues)
            
        return result
    
    def check_value_ranges(self) -> Dict[str, Any]:
        """Check if values fall within specified ranges/sets"""
        if not self.rules:
            return {
                "check": "value_ranges",
                "passed": True,
                "message": "No rules provided, skipping range validation"
            }
        
        range_issues = []
        
        for column, rule in self.rules.items():
            if column not in self.df.columns:
                continue
                
            col_data = self.df[column].dropna()
            
            # Check numeric ranges
            if "min" in rule or "max" in rule:
                try:
                    numeric_data = pd.to_numeric(col_data, errors='coerce')
                    
                    if "min" in rule:
                        violations = numeric_data < rule["min"]
                        if violations.any():
                            violating_rows = self.df[violations].index.tolist()
                            range_issues.append({
                                "column": column,
                                "rule": f"min >= {rule['min']}",
                                "violation_count": violations.sum(),
                                "violating_rows": violating_rows[:10]  # First 10
                            })
                    
                    if "max" in rule:
                        violations = numeric_data > rule["max"]
                        if violations.any():
                            violating_rows = self.df[violations].index.tolist()
                            range_issues.append({
                                "column": column,
                                "rule": f"max <= {rule['max']}",
                                "violation_count": violations.sum(),
                                "violating_rows": violating_rows[:10]  # First 10
                            })
                except:
                    range_issues.append({
                        "column": column,
                        "rule": "numeric_range",
                        "issue": "column_not_numeric",
                        "sample_values": col_data.head(3).tolist()
                    })
            
            # Check allowed values
            if "allowed" in rule:
                allowed_values = set(rule["allowed"])
                actual_values = set(col_data.unique())
                invalid_values = actual_values - allowed_values
                
                if invalid_values:
                    # Find rows with invalid values
                    mask = col_data.isin(invalid_values)
                    violating_rows = self.df[mask].index.tolist()
                    
                    range_issues.append({
                        "column": column,
                        "rule": f"allowed_values: {rule['allowed']}",
                        "invalid_values": list(invalid_values),
                        "violation_count": mask.sum(),
                        "violating_rows": violating_rows[:10]  # First 10
                    })
        
        passed = len(range_issues) == 0
        result = {
            "check": "value_ranges",
            "passed": passed,
            "issues_found": len(range_issues),
            "issues": range_issues,
            "message": f"Range validation: {len(range_issues)} issues found"
        }
        
        if not passed:
            self.issues.extend(range_issues)
            
        return result
    
    def _auto_detect_types(self) -> Dict[str, str]:
        """Auto-detect column types including dates"""
        detected_types = {}
        
        for col in self.df.columns:
            col_data = self.df[col].dropna()
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
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Generate summary statistics for the dataset"""
        stats = {
            "shape": {
                "rows": len(self.df),
                "columns": len(self.df.columns)
            },
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "auto_detected_types": self._auto_detect_types(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicate_rows": self.df.duplicated().sum(),
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
        }
        
        # Add basic stats for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats["numeric_summary"] = {}
            for col in numeric_cols:
                stats["numeric_summary"][col] = {
                    "min": float(self.df[col].min()) if not pd.isna(self.df[col].min()) else None,
                    "max": float(self.df[col].max()) if not pd.isna(self.df[col].max()) else None,
                    "mean": float(self.df[col].mean()) if not pd.isna(self.df[col].mean()) else None,
                    "std": float(self.df[col].std()) if not pd.isna(self.df[col].std()) else None
                }
        
        return stats

class QualityPipeline:
    """Orchestrate all quality checks"""
    
    def __init__(self, df: pd.DataFrame, schema: Optional[Dict] = None, rules: Optional[Dict] = None):
        self.checker = QualityChecker(df, schema, rules)
        
    def run_all_checks(self, min_rows: int = 1) -> Dict[str, Any]:
        """Run all quality checks and return comprehensive results"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "summary_stats": self.checker.get_summary_stats(),
            "checks": {},
            "overall_passed": True,
            "total_issues": 0
        }
        
        # Run individual checks
        row_count_result = self.checker.check_row_count(min_rows)
        type_check_result = self.checker.check_data_types()
        range_check_result = self.checker.check_value_ranges()
        
        results["checks"]["row_count"] = row_count_result
        results["checks"]["data_types"] = type_check_result
        results["checks"]["value_ranges"] = range_check_result
        
        # Determine overall status
        all_checks_passed = all([
            row_count_result["passed"],
            type_check_result["passed"],
            range_check_result["passed"]
        ])
        
        results["overall_passed"] = all_checks_passed
        results["total_issues"] = len(self.checker.issues)
        results["issues"] = self.checker.issues
        
        return results

# MCP Server Implementation
app = Server("data-analyzer")

@app.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="analyze_data",
            description="Analyze structured data for quality issues, data types, and validation (CSV supported, extensible)",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_content": {
                        "type": "string",
                        "description": "Data content as string or base64 encoded data"
                    },
                    "file_format": {
                        "type": "string",
                        "description": "Data format (csv, future: json, excel)",
                        "default": "csv"
                    },
                    "schema": {
                        "type": "object",
                        "description": "Expected data types for columns (e.g., {'id': 'int', 'name': 'str'})",
                        "additionalProperties": {"type": "string"}
                    },
                    "rules": {
                        "type": "object",
                        "description": "Validation rules (e.g., {'age': {'min': 0, 'max': 120}, 'country': {'allowed': ['USA', 'CAN']}})",
                        "additionalProperties": {"type": "object"}
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
                },
                "required": ["data_content"]
            }
        ),
        types.Tool(
            name="get_data_info",
            description="Get basic information about structured data without validation (CSV supported, extensible)",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_content": {
                        "type": "string",
                        "description": "Data content as string or base64 encoded data"
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
                },
                "required": ["data_content"]
            }
        )
    ]

@app.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    
    if name == "analyze_data":
        try:
            data_content = arguments["data_content"]
            file_format = arguments.get("file_format", "csv")
            schema = arguments.get("schema", {})
            rules = arguments.get("rules", {})
            min_rows = arguments.get("min_rows", 1)
            encoding = arguments.get("encoding", "utf-8")
            
            # Try to detect if content is base64 encoded
            try:
                if data_content.startswith("data:"):
                    # Handle data URLs
                    header, data = data_content.split(",", 1)
                    data_content = base64.b64decode(data).decode(encoding)
                elif len(data_content) % 4 == 0 and all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=" for c in data_content):
                    # Might be base64
                    try:
                        decoded = base64.b64decode(data_content).decode(encoding)
                        if "," in decoded or "\t" in decoded:  # Basic delimiter check
                            data_content = decoded
                    except:
                        pass  # Not base64, use as-is
            except:
                pass  # Use content as-is
            
            # Load data using the extensible loader
            df = DataLoader.load_data(data_content, file_format)
            
            # Run quality pipeline
            pipeline = QualityPipeline(df, schema, rules)
            results = pipeline.run_all_checks(min_rows)
            
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(results, indent=2, default=str)
                )
            ]
            
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "type": "analysis_error"
                    }, indent=2)
                )
            ]
    
    elif name == "get_data_info":
        try:
            data_content = arguments["data_content"]
            file_format = arguments.get("file_format", "csv")
            sample_rows = arguments.get("sample_rows", 5)
            
            # Handle base64 encoding
            try:
                if data_content.startswith("data:"):
                    header, data = data_content.split(",", 1)
                    data_content = base64.b64decode(data).decode('utf-8')
                elif len(data_content) % 4 == 0:
                    try:
                        decoded = base64.b64decode(data_content).decode('utf-8')
                        if "," in decoded or "\t" in decoded:
                            data_content = decoded
                    except:
                        pass
            except:
                pass
            
            # Load data using the extensible loader
            df = DataLoader.load_data(data_content, file_format)
            
            # Get basic info
            info = {
                "format": file_format,
                "shape": {"rows": len(df), "columns": len(df.columns)},
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample_data": df.head(sample_rows).to_dict(orient='records'),
                "missing_values": df.isnull().sum().to_dict(),
                "duplicate_rows": int(df.duplicated().sum())
            }
            
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps(info, indent=2, default=str)
                )
            ]
            
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=json.dumps({
                        "error": str(e),
                        "type": "info_error"
                    }, indent=2)
                )
            ]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

async def main():
    """Main entry point for the MCP server"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("data-analyzer-mcp")
    
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="data-analyzer",
                server_version="1.0.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
