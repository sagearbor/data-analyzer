"""
Data Quality Analyzer - Clean UI with Proper Layout
Fixed navbar, three-column layout, no sidebar issues
"""

import streamlit as st
import pandas as pd
import json
import base64
import io
import asyncio
import csv
from typing import Dict, Any, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import PyPDF2
import re
import plotly.graph_objects as go
import plotly.express as px
import hashlib
import pickle
import os
import tempfile
from pathlib import Path

# Import custom modules
from demo_dictionaries import DEMO_DICTIONARIES, get_demo_dictionary
# Import real validation classes from mcp_server (no MCP server needed to run)
from mcp_server import QualityPipeline, QualityChecker
# Force use of custom renderer for better compatibility
MERMAID_AVAILABLE = False
from mermaid_renderer import render_mermaid

# Import LLM parser
try:
    from src.llm_client import LLMDictionaryParser
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    print("LLM client not available. Install openai package to enable.")

# Import environment banner
try:
    import envbanner
    ENVBANNER_AVAILABLE = True
except ImportError:
    ENVBANNER_AVAILABLE = False
    print("env-banner not available. Install with: pip install env-banner")

# Browser console logging helper
def log_to_browser_console(message: str, data: dict = None):
    """Inject JavaScript to log to browser console (visible in Chrome DevTools)"""
    import streamlit.components.v1 as components
    import json
    log_data = json.dumps(data) if data else "{}"
    html = f"""
    <script>
        console.log('[Data Analyzer] {message}', {log_data});
    </script>
    """
    components.html(html, height=0, width=0)

# Configure Streamlit page
st.set_page_config(
    page_title="Data Quality Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Keep sidebar collapsed by default
)

# Initialize environment banner (bottom position to avoid navbar conflict)
if ENVBANNER_AVAILABLE:
    # Get environment and create custom message with PHI warning
    import os
    app_env = os.getenv("APP_ENV", "dev").upper()
    banner_text = f"{app_env} - do NOT use real data or files with PHI."
    envbanner.streamlit(position="bottom", opacity=0.9, text=banner_text)

# Clean, modern CSS without overlapping issues
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Hide sidebar by default */
    section[data-testid="stSidebar"] {
        display: none !important;
    }

    /* Remove excess padding */
    .block-container {
        padding-top: 0.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }

    /* Clean modern styling */
    .stApp {
        background: #ffffff;
    }

    /* Style tabs to look like navbar and fix to top */
    .stTabs [data-baseweb="tab-list"] {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 1000;
        background-color: #1e293b;
        padding: 0.5rem 1rem;
        border-radius: 0;
        margin-bottom: 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Add title to navbar */
    .stTabs [data-baseweb="tab-list"]::before {
        content: "Data Quality Analyzer.";
        color: #e2e8f0;
        font-size: 1.2rem;
        font-weight: 600;
        margin-right: auto;
        padding-right: 2rem;
    }

    /* Add minimal spacing below fixed navbar */
    .stTabs [data-baseweb="tab-panel"] {
        margin-top: 48px;
        padding-top: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        color: #e2e8f0;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        background-color: transparent;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: #334155;
        border-radius: 4px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        border-radius: 4px;
    }

    /* Upload sections styling */
    .upload-section {
        background: #f8fafc;
        border: 2px dashed #cbd5e1;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
    }

    .upload-section:hover {
        border-color: #3b82f6;
        background: #f0f9ff;
    }

    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(59, 130, 246, 0.3);
    }

    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning {
        border-radius: 6px;
        padding: 1rem;
    }

    /* Metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border: 1px solid #bfdbfe;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    /* DataFrame styling */
    .dataframe {
        font-size: 0.9rem;
    }

    /* Additional spacing for content */
    .element-container {
        margin-top: 0.3rem;
    }

    /* Ensure headings don't wrap */
    h3 {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
</style>
""", unsafe_allow_html=True)

class DataQualityAnalyzer:
    """
    Wrapper for real QualityPipeline validation logic.
    Uses actual validation classes from mcp_server.py (no MCP server needed).
    """

    async def analyze_data_quality(self, data: pd.DataFrame, dictionary: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Run data quality analysis using real QualityPipeline.

        Args:
            data: DataFrame to analyze
            dictionary: Optional dictionary with validation rules and schema
                       Can have 'rules' and/or 'schema' keys, or direct field definitions

        Returns:
            Dict with summary, issues, and recommendations
        """
        # Parse dictionary to extract schema and rules for QualityPipeline
        schema = None
        rules = None

        if dictionary and isinstance(dictionary, dict):
            # Handle different dictionary formats
            if 'rules' in dictionary:
                dict_rules = dictionary['rules']
                # Convert web_app dictionary format to QualityChecker format
                schema = {}
                rules = {}
                for field_name, field_spec in dict_rules.items():
                    if isinstance(field_spec, dict):
                        # Extract type for schema
                        if 'type' in field_spec:
                            schema[field_name] = field_spec['type']

                        # Extract validation rules
                        field_rules = {}
                        if 'min' in field_spec and pd.notna(field_spec['min']):
                            try:
                                field_rules['min'] = float(field_spec['min'])
                            except (ValueError, TypeError):
                                pass
                        if 'max' in field_spec and pd.notna(field_spec['max']):
                            try:
                                field_rules['max'] = float(field_spec['max'])
                            except (ValueError, TypeError):
                                pass
                        if 'allowed_values' in field_spec and field_spec['allowed_values']:
                            field_rules['allowed'] = field_spec['allowed_values']

                        if field_rules:
                            rules[field_name] = field_rules

            elif 'schema' in dictionary:
                schema = dictionary.get('schema')
                rules = dictionary.get('validation_rules', {})

        # Run QualityPipeline analysis
        pipeline = QualityPipeline(data, schema=schema, rules=rules)
        results = pipeline.run_all_checks(min_rows=1)

        # Transform QualityPipeline results to web_app expected format
        issues = []

        # Transform QualityPipeline issues (which have 'column', 'rule', 'violating_rows')
        # into web UI format (which needs 'type', 'severity', 'message', 'row', 'value')
        for qp_issue in results.get('issues', []):
            column = qp_issue.get('column')
            rule = qp_issue.get('rule', '')
            violating_rows = qp_issue.get('violating_rows', [])

            # Determine issue type and severity based on the rule
            if 'min >=' in rule or 'max <=' in rule:
                issue_type = "range_violation"
                severity = "error"
            elif 'allowed_values' in rule:
                issue_type = "invalid_categorical_value"
                severity = "error"
            elif qp_issue.get('issue') == 'type_mismatch':
                issue_type = "type_mismatch"
                severity = "error"
            elif qp_issue.get('issue') == 'datetime_validation_failed':
                issue_type = "invalid_date"
                severity = "error"
            else:
                issue_type = qp_issue.get('issue', 'validation_error')
                severity = "error"

            # Create individual issue for each violating row
            for row_idx in violating_rows:
                value = data[column].iloc[row_idx] if column in data.columns and row_idx < len(data) else None

                issues.append({
                    "type": issue_type,
                    "severity": severity,
                    "column": column,
                    "row": int(row_idx),
                    "value": value,
                    "message": f"Value {value} in column '{column}' violates rule: {rule}"
                })

        # Add missing value issues
        for col in data.columns:
            missing = data[col].isnull().sum()
            if missing > 0:
                issues.append({
                    "type": "missing_values",
                    "severity": "warning",
                    "column": col,
                    "count": int(missing),
                    "percentage": round(missing / len(data) * 100, 2),
                    "message": f"Column '{col}' has {missing} missing values ({round(missing/len(data)*100, 2)}%)"
                })

        # Build summary
        summary = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "issues_found": len(issues),
            "critical_issues": sum(1 for i in issues if i.get('severity') == 'error'),
            "warnings": sum(1 for i in issues if i.get('severity') == 'warning'),
            "data_types": results.get('summary_stats', {}).get('dtypes', {col: str(data[col].dtype) for col in data.columns}),
            "completeness": round((1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100, 2)
        }

        return {
            "summary": summary,
            "issues": issues,
            "recommendations": self._generate_recommendations(issues),
            "quality_checks": results.get('checks', {}),
            "summary_stats": results.get('summary_stats', {})
        }

    def _generate_recommendations(self, issues):
        """Generate recommendations based on issues found"""
        recommendations = []

        issue_types = set(i.get('type', i.get('issue', 'unknown')) for i in issues)

        if 'missing_values' in issue_types:
            recommendations.append({
                "type": "data_cleaning",
                "priority": "high",
                "message": "Consider implementing data imputation strategies for columns with missing values"
            })

        if any(t in issue_types for t in ['type_mismatch', 'datetime_validation_failed', 'invalid_value']):
            recommendations.append({
                "type": "data_validation",
                "priority": "critical",
                "message": "Data type issues detected. Review data source and implement validation at ingestion"
            })

        if 'range_violation' in issue_types or any('violation' in str(t) for t in issue_types):
            recommendations.append({
                "type": "business_rules",
                "priority": "high",
                "message": "Values outside expected ranges detected. Review business rules and data constraints"
            })

        if any('allowed' in str(t) for t in issue_types):
            recommendations.append({
                "type": "categorical_validation",
                "priority": "high",
                "message": "Invalid categorical values found. Verify allowed values match business requirements"
            })

        return recommendations

def load_demo_data(dataset_name: str):
    """Load demo dataset matching the dictionary options"""
    demo_data = {
        'western': pd.DataFrame({
            'employee_id': [1001, 1002, 1003, 1004, 1005],
            'first_name': ['John', 'Jane', 'Mike', 'Bob', 'Alice'],
            'last_name': ['Smith', 'Doe', 'Johnson', 'Brown', 'Wilson'],
            'age': [35, 28, 67, 45, 32],  # ERROR: 67 is outside range (max 65)
            'salary': [75000, 85000, 45000, 95000, None],  # ERROR: 45000 below min (50000), WARNING: None is missing
            'hire_date': ['2022-03-15', '2023-01-10', '2023-99-99', '2021-07-22', '2022-11-30'],  # ERROR: invalid date 2023-99-99
            'last_login_datetime': ['2023-01-15 10:30:00', '2023-02-20 14:45:00', '2023-03-10 09:15:00', '2023-04-05 16:20:00', '2023-05-12 11:00:00'],
            'bonus_percentage': [5.5, 10.0, 15.5, 8.0, 12.5],
            'department': ['Engineering', 'Marketing', 'InvalidDept', 'Sales', 'Finance'],  # ERROR: InvalidDept not in allowed values
            'is_active': [True, True, False, None, True],  # WARNING: None is missing
            'skills': ['Python;SQL', 'Marketing;Analytics', 'Java;AWS', 'Sales;CRM', 'Python;Leadership'],
            'email': ['john@company.com', 'jane@company.com', 'mike@company.com', 'bob@company.com', 'alice@company.com'],
            'phone': ['+1-555-1234', '+1-555-5678', '+1-555-9012', None, '+1-555-3456']  # WARNING: None is missing
        }),
        'clinical': pd.DataFrame({
            'patient_id': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006', 'P007', 'P008'],
            'age': [45, 62, 73, 28, 155, 39, 67, 51],  # 155 is invalid age
            'gender': ['M', 'F', 'M', 'F', 'X', 'F', 'M', 'invalid'],  # X and invalid are issues
            'diagnosis_code': ['I21.0', 'J18.9', 'N18.9', 'K92.2', 'invalid', 'G20.9', 'E11.9', ''],
            'admission_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', 'invalid-date', '2024-01-20', '2024-01-21', '2024-01-22'],
            'discharge_date': ['2024-01-20', '2024-01-22', '2024-01-25', '2024-01-19', None, '2024-01-21', '', '2024-01-23'],
            'treatment_type': ['Emergency', 'Inpatient', 'Inpatient', 'Observation', 'invalid', 'Outpatient', 'Inpatient', 'Unknown'],
            'lab_result_wbc': [7.5, 12.3, 6.8, 8.2, 50.0, 7.8, 10.5, None],  # 50.0 is out of range
            'lab_result_hemoglobin': [14.2, 12.8, 10.5, 13.5, 'invalid', 14.5, None, 13.2],
            'blood_pressure_systolic': [130, 145, 155, 110, 250, 125, 165, 135],  # 250 is too high
            'blood_pressure_diastolic': [85, 92, 95, 70, 130, 80, 98, 82],  # 130 is too high
            'temperature': [37.2, 38.5, 36.8, 36.9, 45.0, 36.7, 37.3, None],  # 45.0 is impossible
            'heart_rate': [88, 96, 78, 72, 200, 75, 90, 68],  # 200 is too high
            'follow_up_required': ['Yes', 'Yes', 'Yes', 'No', 'Maybe', 'Yes', 'Yes', None],  # Maybe is invalid
            'outcome_status': ['Improved', 'Recovered', 'Stable', 'Recovered', 'invalid', 'Stable', 'Ongoing', 'Improved'],
            'length_of_stay': [5, 6, 8, 1, 500, 1, None, 1]  # 500 days is unrealistic
        }),
        'asian': pd.DataFrame({
            'staff_id': [2001, 2002, 2003, 2004, 2005],
            'given_name': ['Akiko', 'Wei', None, 'Raj', 'Mei'],
            'family_name': ['Tanaka', 'Zhang', 'Kumar', 'invalid', 'Chen'],
            'age': [30, 21, 45, 38, 62],  # 21 below min, 62 above max
            'monthly_salary': [8500, 9200, 6000, 10500, None],  # 6000 below min
            'join_date': ['2020-06-01', 'invalid-date', '2021-03-15', '2022-09-10', '2023-01-20'],
            'dept_code': ['DEV', 'MKT', 'invalid', 'OPS', 'FIN'],
            'active_status': [1, 1, 0, None, 1],
            'work_email': ['akiko@work.com', 'wei@work.com', 'invalid', 'raj@work.com', 'mei@work.com']
        }),
        'mixed': pd.DataFrame({
            'id': [3001, 3002, 3003, 3004, 3005],
            'name_first': ['Carlos', 'Emma', 'invalid', 'Liu', None],
            'name_last': ['Rodriguez', 'invalid', 'Brown', 'Wang', 'Lee'],
            'age': [40, 24, 35, 56, 45],  # 24 below min, 56 above max
            'salary': [70000, 80000, 60000, 90000, None],  # 60000 below min, 90000 above max
            'hired': ['2022-05-01', '2023-08-15', 'invalid-date', '2021-12-01', '2023-03-10'],
            'active': [True, False, None, True, True],
            'department': ['Research', 'invalid', 'Engineering', 'Quality', 'Sales']
        })
    }
    return demo_data.get(dataset_name, demo_data['western'])

def create_issue_heatmap(df: pd.DataFrame, issues: list):
    """Create an interactive heatmap with hover tooltips using Plotly"""
    try:
        rows, cols = len(df), len(df.columns)

        # Condense large datasets
        max_display_rows = 60
        max_display_cols = 100  # Increased for wide datasets

        row_factor = max(1, rows // max_display_rows)
        col_factor = max(1, cols // max_display_cols)

        display_rows = min(rows, max_display_rows)
        display_cols = min(cols, max_display_cols)

        # Initialize matrix and hover text
        issue_matrix = np.zeros((display_rows, display_cols))
        hover_text = [['' for _ in range(display_cols)] for _ in range(display_rows)]

        # Map issues to matrix
        for issue in issues:
            if 'row' in issue and 'column' in issue:
                try:
                    col_idx = df.columns.get_loc(issue['column'])
                    row_idx = issue['row']

                    # Map to display coordinates
                    display_row = min(row_idx // row_factor, display_rows - 1)
                    display_col = min(col_idx // col_factor, display_cols - 1)

                    # Set severity (2 for error, 1 for warning)
                    severity_value = 2 if issue['severity'] == 'error' else 1
                    issue_matrix[display_row, display_col] = max(issue_matrix[display_row, display_col], severity_value)

                    # Build hover text
                    issue_info = f"<b>{issue['type'].replace('_', ' ').title()}</b><br>"
                    issue_info += f"Row: {row_idx}<br>"
                    issue_info += f"Column: {issue['column']}<br>"
                    issue_info += f"Value: {issue.get('value', 'N/A')}<br>"
                    issue_info += f"Severity: {issue['severity']}"

                    if hover_text[display_row][display_col]:
                        hover_text[display_row][display_col] += "<br><br>" + issue_info
                    else:
                        hover_text[display_row][display_col] = issue_info
                except:
                    pass

        # Calculate aspect ratio for proper dimensions
        aspect_ratio = cols / rows
        if aspect_ratio > 1:
            # Wide dataset
            fig_width = 300
            fig_height = max(50, 300 / aspect_ratio)
        else:
            # Tall dataset
            fig_height = 300
            fig_width = max(50, 300 * aspect_ratio)

        # Create interactive Plotly heatmap
        fig = go.Figure(data=go.Heatmap(
            z=issue_matrix,
            text=hover_text,
            hovertemplate='%{text}<extra></extra>',
            colorscale=[
                [0, '#ffffff'],      # White for no issue
                [0.5, '#fbbf24'],    # Yellow for warning
                [1, '#ef4444']       # Red for error
            ],
            showscale=False,
            xgap=1,
            ygap=1
        ))

        # Update layout
        fig.update_layout(
            title={
                'text': f'{rows} rows × {cols} cols' + (f' (scale {row_factor}:{col_factor})' if rows > max_display_rows or cols > max_display_cols else ''),
                'font': {'size': 10}
            },
            width=fig_width,
            height=fig_height,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis={'showticklabels': False, 'showgrid': False},
            yaxis={'showticklabels': False, 'showgrid': False},
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        # Display with Streamlit
        st.plotly_chart(fig, use_container_width=False)

        # Show summary
        error_count = np.sum(issue_matrix == 2)
        warning_count = np.sum(issue_matrix == 1)
        if error_count > 0 or warning_count > 0:
            st.caption(f"🔴 {int(error_count)} cells with errors, 🟡 {int(warning_count)} cells with warnings")

    except Exception as e:
        st.caption(f"Issue map unavailable: {str(e)}")

def export_to_excel_with_highlighting(df: pd.DataFrame, issues: list) -> bytes:
    """Export data to Excel with error cells highlighted"""
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data', index=False)

        # Create issues summary sheet
        issues_df = pd.DataFrame(issues)
        if not issues_df.empty:
            issues_df.to_excel(writer, sheet_name='Issues', index=False)

        # Get workbook and worksheet
        workbook = writer.book
        worksheet = workbook['Data']

        # Apply highlighting to cells with issues
        from openpyxl.styles import PatternFill, Font

        error_fill = PatternFill(start_color="FFCCCB", end_color="FFCCCB", fill_type="solid")
        warning_fill = PatternFill(start_color="FFE5B4", end_color="FFE5B4", fill_type="solid")

        for issue in issues:
            if 'row' in issue and 'column' in issue:
                try:
                    col_idx = df.columns.get_loc(issue['column']) + 1
                    row_idx = issue['row'] + 2  # +2 for header and 0-index
                    cell = worksheet.cell(row=row_idx, column=col_idx)

                    if issue['severity'] == 'error':
                        cell.fill = error_fill
                    elif issue['severity'] == 'warning':
                        cell.fill = warning_fill

                    # Add comment with issue details
                    from openpyxl.comments import Comment
                    cell.comment = Comment(issue['message'], "Data Analyzer")
                except:
                    pass

        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    return output.getvalue()

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'dictionary' not in st.session_state:
    st.session_state.dictionary = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'mcp_client' not in st.session_state:
    st.session_state.mcp_client = DataQualityAnalyzer()
if 'dict_cache' not in st.session_state:
    st.session_state.dict_cache = {}  # Cache for parsed dictionaries
if 'last_dict_file' not in st.session_state:
    st.session_state.last_dict_file = None  # Track last uploaded dictionary
if 'cache_dir' not in st.session_state:
    # Create persistent cache directory (user-specific to avoid multi-user conflicts)
    import getpass
    try:
        username = getpass.getuser()
    except:
        # Fallback if getuser() fails
        import os
        username = os.environ.get('USER', os.environ.get('USERNAME', 'default'))

    cache_dir = Path.home() / f'.data_analyzer_cache_{username}'
    cache_dir.mkdir(exist_ok=True)
    st.session_state.cache_dir = cache_dir

    print(f"\n📦 Cache directory: {cache_dir}")
    print(f"   (User-specific to avoid multi-user conflicts)\n")

# Create simple navigation tabs - no right-click support but clean UI
tab1, tab2 = st.tabs(["📊 Analyze", "ℹ️ About"])

with tab1:
    # Subtitle only - title is now in navbar
    st.markdown("Upload your data, optionally add validation rules, and analyze")

    # Create three columns for the main components
    col1, col2, col3 = st.columns([2, 2, 1.5])

    with col1:
        st.markdown("### 📁 Upload Data")

        # File uploader first
        uploaded_file = st.file_uploader(
            " ",  # Empty label to avoid duplication
            type=['csv', 'json', 'txt'],
            key="data_uploader",
            label_visibility="collapsed"
        )

        if uploaded_file:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                try:
                    if uploaded_file.name.endswith('.csv'):
                        st.session_state.data = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith('.json'):
                        st.session_state.data = pd.read_json(uploaded_file)
                    else:
                        st.session_state.data = pd.read_csv(uploaded_file, sep='\t')
                    st.success(f"✅ Loaded {len(st.session_state.data)} rows × {len(st.session_state.data.columns)} columns")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")

        # Demo data selector below file uploader
        demo_option = st.selectbox(
            "Or load demo data:",
            ["None", "CSV - Western", "CSV - Asian", "CSV - Clinical", "JSON - Mixed"],
            key="demo_selector",
            help="Clinical data includes matching dictionary in demo_data/clinical_dict.json"
        )

        if demo_option != "None":
            dataset_map = {
                "CSV - Western": "western",
                "CSV - Asian": "asian",
                "CSV - Clinical": "clinical",
                "JSON - Mixed": "mixed"
            }
            if demo_option in dataset_map:
                st.session_state.data = load_demo_data(dataset_map[demo_option])
                st.success(f"✅ Loaded {demo_option} demo data")
                if demo_option == "CSV - Clinical":
                    st.info("📖 Matching dictionary available: Upload 'demo_data/clinical_dict.json' for validation rules")

    with col2:
        st.markdown("### 📋 Dictionary")

        # Dictionary file uploader first - aligned with data uploader
        dict_file = st.file_uploader(
            " ",  # Empty label to avoid duplication
            type=['json', 'pdf', 'csv', 'txt'],
            key="dict_uploader",
            label_visibility="collapsed",
            help="Optional - defines validation rules for data quality checks (JSON, PDF, CSV, or TXT)"
        )

        # Add LLM parsing option if available with auto-detection
        if LLM_AVAILABLE and dict_file:
            # DEBUG: Show filename
            st.caption(f"📎 Uploaded file: **{dict_file.name}** ({dict_file.type if hasattr(dict_file, 'type') else 'unknown type'})")

            llm_mode = st.selectbox(
                "Dictionary parsing method:",
                ["Auto-detect (recommended)", "Always use AI parsing", "Never use AI (manual only)"],
                index=0,  # Default to auto-detect
                help="Auto: PDF→AI, structured CSV→manual | Always: Force AI for all | Never: Manual parsing only"
            )

            # Determine if we should use LLM based on mode and file type
            if "Always" in llm_mode:
                use_llm = True
            elif "Never" in llm_mode:
                use_llm = False
            else:  # Auto-detect
                # Check file type and structure
                print(f"\n🔍 AUTO-DETECT: Checking file '{dict_file.name}'")
                print(f"   Extension check: .pdf={dict_file.name.endswith('.pdf')}, .csv={dict_file.name.endswith('.csv')}")

                if dict_file.name.endswith('.pdf'):
                    use_llm = True
                    st.info("🤖 Auto-detected: PDF requires AI parsing")
                    print(f"   ✅ Detected as PDF, will use LLM")
                elif dict_file.name.endswith('.csv'):
                    # Peek at CSV to check if it's structured
                    dict_file.seek(0)
                    sample = dict_file.read(1024).decode('utf-8', errors='ignore')
                    dict_file.seek(0)
                    # Check for standard column names
                    if any(col in sample for col in ['Column', 'Type', 'Min', 'Max', 'Allowed_Values', 'Field Name']):
                        use_llm = False
                        st.info("📊 Auto-detected: Structured CSV, using manual parsing")
                    else:
                        use_llm = True
                        st.info("🤖 Auto-detected: Unstructured CSV, using AI parsing")
                else:
                    use_llm = True
                    st.info(f"🤖 Auto-detected: {dict_file.name.split('.')[-1].upper()} file, using AI parsing")
        else:
            use_llm = False

        if dict_file:
            try:
                # Calculate file hash for caching (works for all file types)
                dict_file.seek(0)
                raw_content = dict_file.read()
                file_hash = hashlib.md5(raw_content).hexdigest()
                dict_file.seek(0)  # Reset for reading

                # Create cache key with LLM flag
                cache_key = f"{file_hash}_llm" if use_llm else file_hash
                cache_file = st.session_state.cache_dir / f"{cache_key}.pkl"

                # Check if already cached
                if cache_key in st.session_state.dict_cache:
                    # Use in-memory cache
                    st.session_state.dictionary = st.session_state.dict_cache[cache_key]

                    # CLEAR CACHE LOGGING
                    print("\n" + "="*80)
                    print("💾 LOADING FROM MEMORY CACHE (NO LLM CALL)")
                    print(f"   Cache key: {cache_key}")
                    print(f"   File: {dict_file.name}")
                    print("="*80 + "\n")

                    st.success(f"⚡ Using cached dictionary (instant load)")
                    st.warning("🔄 **CACHE HIT**: Using previously parsed dictionary. Clear cache below if data dictionary changed.")

                    if use_llm:
                        fields_count = len(st.session_state.dictionary.get('fields', []))
                        st.info(f"📊 Contains {fields_count} AI-extracted field definitions")
                    else:
                        st.info(f"📊 Contains {len(st.session_state.dictionary.get('rules', {}))} validation rules")
                elif cache_file.exists():
                    # Load from persistent cache file
                    with open(cache_file, 'rb') as f:
                        st.session_state.dictionary = pickle.load(f)
                        st.session_state.dict_cache[cache_key] = st.session_state.dictionary

                    # CLEAR CACHE LOGGING
                    print("\n" + "="*80)
                    print("💾 LOADING FROM DISK CACHE (NO LLM CALL)")
                    print(f"   Cache file: {cache_file.name}")
                    print(f"   File: {dict_file.name}")
                    print("="*80 + "\n")

                    st.success(f"⚡ Loaded dictionary from disk cache (no API calls)")
                    st.warning("🔄 **CACHE HIT**: Using previously parsed dictionary. Clear cache below if data dictionary changed.")

                    if use_llm:
                        fields_count = len(st.session_state.dictionary.get('fields', []))
                        st.info(f"📊 Contains {fields_count} AI-extracted field definitions")
                    else:
                        st.info(f"📊 Contains {len(st.session_state.dictionary.get('rules', {}))} validation rules")
                # Use LLM parsing if enabled and not cached
                elif use_llm and LLM_AVAILABLE:
                    # CLEAR LLM MARKER
                    st.warning("🤖 **LLM ACTIVE**: Sending data to Azure OpenAI GPT-4 for intelligent dictionary parsing...")
                    print("\n" + "="*80)
                    print("🤖 LLM DICTIONARY PARSER INVOKED")
                    print(f"   File: {dict_file.name}")
                    print(f"   Size: {len(raw_content)} bytes")
                    print("="*80 + "\n")

                    # More informative spinner with warning about processing time
                    with st.spinner("🤖 Using AI to extract field definitions... This may take 30-60 seconds for large PDFs."):
                        # Read file content
                        file_content = ""
                        if dict_file.name.endswith('.pdf'):
                            pdf_reader = PyPDF2.PdfReader(dict_file)
                            for page in pdf_reader.pages:
                                file_content += page.extract_text() + "\n"
                        elif dict_file.name.endswith('.csv'):
                            file_content = dict_file.read().decode('utf-8')
                        elif dict_file.name.endswith('.txt'):
                            file_content = dict_file.read().decode('utf-8')
                        elif dict_file.name.endswith('.json'):
                            # For JSON, convert to readable text
                            json_data = json.load(dict_file)
                            file_content = json.dumps(json_data, indent=2)
                        else:
                            file_content = dict_file.read().decode('utf-8')

                        # Initialize LLM parser
                        llm_parser = LLMDictionaryParser()

                        # Estimate tokens for browser console log
                        import time
                        estimated_tokens = llm_parser.count_tokens(file_content)
                        start_time = time.time()
                        start_timestamp = time.strftime('%H:%M:%S')

                        # Log to browser console
                        log_to_browser_console(
                            f"🤖 LLM parsing started at {start_timestamp}",
                            {
                                "model": llm_parser.deployment,
                                "tokens": estimated_tokens,
                                "file": dict_file.name,
                                "size_bytes": len(file_content)
                            }
                        )

                        # Parse with LLM
                        # Don't truncate - the LLM parser handles chunking internally
                        # Process more fields for comprehensive extraction
                        parsed_result = llm_parser.parse_dictionary(file_content, max_fields=500)

                        # Calculate elapsed time
                        elapsed_time = time.time() - start_time

                        # Log completion to browser console
                        log_to_browser_console(
                            f"✅ LLM parsing completed in {elapsed_time:.1f}s",
                            {
                                "fields_extracted": len(parsed_result.get('fields', [])),
                                "chunks_processed": parsed_result.get('metadata', {}).get('chunks_processed', 0),
                                "mode": parsed_result.get('metadata', {}).get('mode', 'unknown')
                            }
                        )

                        # Store the parsed dictionary
                        st.session_state.dictionary = {
                            "source": "LLM Parser",
                            "filename": dict_file.name,
                            "rules": parsed_result.get("schema", {}),
                            "fields": parsed_result.get("fields", []),
                            "metadata": parsed_result.get("metadata", {})
                        }

                        # Cache the result both in memory and to disk
                        st.session_state.dict_cache[cache_key] = st.session_state.dictionary
                        with open(cache_file, 'wb') as f:
                            pickle.dump(st.session_state.dictionary, f)
                        st.info(f"💾 Dictionary cached - future loads will be instant (no API calls)")

                        # Add processing time to success message
                        processing_time = parsed_result.get('metadata', {}).get('processing_time_seconds', 0)
                        chunks_processed = parsed_result.get('metadata', {}).get('chunks_processed', 0)
                        st.success(f"✅ AI extracted {len(parsed_result.get('fields', []))} field definitions from {chunks_processed} chunks in {processing_time:.1f} seconds")

                        # Show extracted fields
                        if parsed_result.get('fields'):
                            # Expand by default if we got results, especially for large dictionaries
                            expand_fields = len(parsed_result['fields']) <= 20
                            with st.expander(f"📋 Extracted Fields ({len(parsed_result['fields'])})", expanded=expand_fields):
                                for field in parsed_result['fields'][:10]:
                                    field_info = f"**{field['field_name']}** ({field['data_type']})"
                                    if field.get('required'):
                                        field_info += " *[Required]*"
                                    if field.get('description'):
                                        field_info += f"\n   {field['description']}"
                                    if field.get('min_value') or field.get('max_value'):
                                        field_info += f"\n   Range: {field.get('min_value', 'N/A')} - {field.get('max_value', 'N/A')}"
                                    if field.get('allowed_values'):
                                        field_info += f"\n   Values: {', '.join(field['allowed_values'][:5])}"
                                    st.markdown(field_info)
                                if len(parsed_result['fields']) > 10:
                                    st.info(f"📊 Showing first 10 of {len(parsed_result['fields'])} extracted fields. Use 'View All Fields' below to see more.")

                elif dict_file.name.endswith('.pdf'):
                    # PDF without LLM - already have hash from above
                    dict_file.seek(0)  # Reset for reading

                    # Check persistent file cache first
                    cache_file = st.session_state.cache_dir / f"{file_hash}.pkl"

                    if cache_file.exists():
                        # Load from persistent cache file
                        with open(cache_file, 'rb') as f:
                            st.session_state.dictionary = pickle.load(f)
                            st.session_state.dict_cache[file_hash] = st.session_state.dictionary
                        st.success(f"⚡ Loaded dictionary from cache (instant)")
                        st.info(f"📊 Contains {len(st.session_state.dictionary.get('rules', {}))} validation rules")
                    elif file_hash in st.session_state.dict_cache:
                        # Use in-memory cache
                        st.session_state.dictionary = st.session_state.dict_cache[file_hash]
                        st.success(f"⚡ Using cached dictionary '{dict_file.name}' (instant load)")
                        st.info(f"📊 Contains {len(st.session_state.dictionary.get('rules', {}))} validation rules")
                    else:
                        # Manual PDF parsing (NO LLM)
                        st.info("📄 **MANUAL PARSING**: Using basic regex patterns (limited extraction). Enable AI parsing for better results.")
                        print("\n" + "="*80)
                        print("📄 MANUAL PDF PARSER (NO LLM)")
                        print(f"   File: {dict_file.name}")
                        print("   ⚠️ WARNING: Basic regex patterns only - may miss complex field definitions")
                        print("="*80 + "\n")

                        # Parse PDF dictionary with container to prevent UI blocking
                        with st.container():
                            progress_bar = st.progress(0, text="Parsing PDF dictionary...")

                            # Read PDF content
                            pdf_reader = PyPDF2.PdfReader(dict_file)
                            num_pages = len(pdf_reader.pages)

                            extracted_text = ""
                            extracted_rules = {}

                            # Process pages with continuous progress updates
                            for i, page in enumerate(pdf_reader.pages):
                                # Update progress for every page
                                progress_bar.progress((i + 1) / num_pages, text=f"Processing page {i+1} of {num_pages}...")

                                page_text = page.extract_text()
                                extracted_text += page_text

                                # Look for validation rules in the PDF (example patterns)
                                # Look for date fields
                                date_fields = re.findall(r'([\w_]+).*?(?:date|Date|DATE)', page_text)
                                for field in date_fields:
                                    if field not in extracted_rules:
                                        extracted_rules[field] = {"type": "date"}

                                # Look for numeric ranges
                                range_patterns = re.findall(r'([\w_]+).*?(?:range|Range|between).*?(\d+).*?(?:to|and|-|–).*?(\d+)', page_text)
                                for field, min_val, max_val in range_patterns:
                                    if field not in extracted_rules:
                                        extracted_rules[field] = {"min": int(min_val), "max": int(max_val)}

                            # Clear progress bar
                            progress_bar.empty()

                        # Store extracted dictionary
                        st.session_state.dictionary = {
                            "source": "PDF",
                            "filename": dict_file.name,
                            "rules": extracted_rules,
                            "pages": num_pages,
                            "text_length": len(extracted_text),
                            "hash": file_hash
                        }

                        # Cache the parsed dictionary both in memory and to file
                        st.session_state.dict_cache[file_hash] = st.session_state.dictionary

                        # Save to persistent cache file
                        cache_file = st.session_state.cache_dir / f"{file_hash}.pkl"
                        with open(cache_file, 'wb') as f:
                            pickle.dump(st.session_state.dictionary, f)

                        st.success(f"✅ Parsed {num_pages} pages from PDF dictionary")
                        st.info(f"💾 Dictionary cached to disk for permanent reuse")
                        st.caption(f"📁 Cache location: {cache_file}")

                        if extracted_rules:
                            with st.expander(f"Found {len(extracted_rules)} validation rules", expanded=False):
                                for field, rule in list(extracted_rules.items())[:10]:  # Show first 10
                                    st.text(f"{field}: {rule}")
                                if len(extracted_rules) > 10:
                                    st.text(f"... and {len(extracted_rules) - 10} more")
                elif dict_file.name.endswith('.json'):
                    st.session_state.dictionary = json.load(dict_file)
                    st.success("✅ JSON dictionary loaded")
                elif dict_file.name.endswith('.csv') and not use_llm:
                    # Parse CSV dictionary (NO LLM) - only if LLM mode not active
                    st.info("📊 **CSV PARSING**: Reading structured CSV data dictionary...")
                    print(f"\n📊 CSV DICTIONARY PARSER: {dict_file.name}")

                    import pandas as pd
                    dict_file.seek(0)
                    df = pd.read_csv(dict_file)
                    rules = {}
                    for _, row in df.iterrows():
                        if 'Column' in row or 'column' in row or 'Field' in row or 'field' in row:
                            field_name = row.get('Column') or row.get('column') or row.get('Field') or row.get('field')
                            if field_name:
                                rule = {}
                                if 'Type' in row or 'type' in row:
                                    rule['type'] = str(row.get('Type') or row.get('type'))
                                if 'Min' in row or 'min' in row:
                                    rule['min'] = row.get('Min') or row.get('min')
                                if 'Max' in row or 'max' in row:
                                    rule['max'] = row.get('Max') or row.get('max')
                                if 'Required' in row or 'required' in row:
                                    rule['required'] = row.get('Required') or row.get('required')
                                if 'Allowed_Values' in row or 'allowed_values' in row:
                                    allowed = row.get('Allowed_Values') or row.get('allowed_values')
                                    if allowed and not pd.isna(allowed):
                                        rule['allowed_values'] = [v.strip() for v in str(allowed).split(',')]
                                rules[field_name] = rule
                    st.session_state.dictionary = {
                        "source": "CSV",
                        "filename": dict_file.name,
                        "rules": rules
                    }
                    st.success(f"✅ Parsed {len(rules)} field definitions from CSV")
                else:
                    st.error(f"⚠️ Unsupported dictionary format: **{dict_file.name}**")
                    st.info(f"Debug: use_llm={use_llm}, LLM_AVAILABLE={LLM_AVAILABLE}, file ends with .csv={dict_file.name.endswith('.csv')}")
                    print(f"\n⚠️ UNSUPPORTED FORMAT: {dict_file.name}")
                    print(f"   use_llm: {use_llm}")
                    print(f"   LLM_AVAILABLE: {LLM_AVAILABLE}")
                    print(f"   File extension checks: .pdf={dict_file.name.endswith('.pdf')}, .csv={dict_file.name.endswith('.csv')}, .json={dict_file.name.endswith('.json')}")
            except Exception as e:
                st.error(f"Error loading dictionary: {str(e)}")

        # View All Fields button - accessible location near dictionary upload
        if st.session_state.dictionary and st.session_state.dictionary.get('fields'):
            fields_list = st.session_state.dictionary['fields']
            num_fields = len(fields_list)

            if num_fields > 0:
                st.markdown("---")
                with st.expander(f"📋 View All {num_fields} Extracted Fields", expanded=False):
                    for field in fields_list:
                        field_info = f"**{field['field_name']}** ({field.get('data_type', 'unknown')})"
                        if field.get('required'):
                            field_info += " *[Required]*"
                        if field.get('description'):
                            # Truncate very long descriptions
                            desc = field['description'][:150] + "..." if len(field['description']) > 150 else field['description']
                            field_info += f"\n   📝 {desc}"
                        if field.get('min_value') is not None or field.get('max_value') is not None:
                            field_info += f"\n   📊 Range: {field.get('min_value', 'N/A')} - {field.get('max_value', 'N/A')}"
                        if field.get('allowed_values'):
                            vals = field['allowed_values'][:8]  # Show first 8
                            vals_str = ', '.join(vals)
                            if len(field['allowed_values']) > 8:
                                vals_str += f" ... +{len(field['allowed_values']) - 8} more"
                            field_info += f"\n   ✓ Allowed: {vals_str}"
                        st.markdown(field_info)

        # Demo dictionary selector below file uploader
        demo_dict = st.selectbox(
            "Or load demo dictionary:",
            ["None"] + list(DEMO_DICTIONARIES.keys()),
            key="demo_dict_selector"
        )

        if demo_dict != "None":
            # Get demo dictionary CSV string and parse it
            demo_csv_string = get_demo_dictionary(demo_dict)

            # Parse CSV string into rules dictionary (same logic as CSV upload)
            import io
            df = pd.read_csv(io.StringIO(demo_csv_string))
            rules = {}
            for _, row in df.iterrows():
                if 'Column' in row or 'column' in row or 'Field' in row or 'field' in row:
                    field_name = row.get('Column') or row.get('column') or row.get('Field') or row.get('field')
                    if field_name:
                        rule = {}
                        if 'Type' in row or 'type' in row:
                            rule['type'] = str(row.get('Type') or row.get('type'))
                        if 'Min' in row or 'min' in row:
                            rule['min'] = row.get('Min') or row.get('min')
                        if 'Max' in row or 'max' in row:
                            rule['max'] = row.get('Max') or row.get('max')
                        if 'Required' in row or 'required' in row:
                            rule['required'] = row.get('Required') or row.get('required')
                        if 'Allowed_Values' in row or 'allowed_values' in row:
                            allowed = row.get('Allowed_Values') or row.get('allowed_values')
                            if allowed and not pd.isna(allowed):
                                rule['allowed_values'] = [v.strip() for v in str(allowed).split(',')]
                        rules[field_name] = rule

            st.session_state.dictionary = {
                "source": "Demo Dictionary",
                "filename": demo_dict,
                "rules": rules
            }
            st.success(f"✅ Loaded {demo_dict} ({len(rules)} field definitions)")

        # Add cache management
        st.markdown("---")
        st.markdown("#### 🗑️ Cache Management")

        cache_files = list(st.session_state.cache_dir.glob("*.pkl"))
        num_cached = len(cache_files)

        if num_cached > 0:
            st.caption(f"📦 {num_cached} dictionaries cached")

            if st.button("🗑️ Clear All Cache", help="Delete all cached dictionaries to force re-parsing"):
                try:
                    for cache_file in cache_files:
                        cache_file.unlink()
                    st.session_state.dict_cache = {}
                    st.session_state.dictionary = None
                    st.success(f"✅ Cleared {num_cached} cached dictionaries")
                    print(f"\n🗑️ CLEARED {num_cached} CACHE FILES\n")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error clearing cache: {e}")
        else:
            st.caption("No cached dictionaries")

    with col3:
        # Export dropdown - moved from bottom
        if st.session_state.analysis_results:
            st.markdown("### 📥 Export")
            export_format = st.selectbox(
                "Choose format:",
                ["Select format to export", "Excel with highlighting", "JSON report"],
                key="export_format",
                on_change=lambda: None  # Trigger rerun on selection
            )

            # Show download button immediately when format is selected
            if export_format == "Excel with highlighting":
                excel_data = export_to_excel_with_highlighting(
                    st.session_state.data,
                    st.session_state.analysis_results['issues']
                )
                st.download_button(
                    label="📊 Download Excel",
                    data=excel_data,
                    file_name=f"data_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                    type="secondary"
                )
            elif export_format == "JSON report":
                json_str = json.dumps(st.session_state.analysis_results, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    type="secondary"
                )

    # Prominent Run Analysis button - full width between uploads and results
    st.markdown("---")
    st.markdown("## 🚀 Run Analysis")

    # Create centered column for button
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        # Enable button only when data is loaded
        if st.button(
            "🚀 Analyze Data Quality",
            disabled=(st.session_state.data is None),
            use_container_width=True,
            type="primary",
            help="Load data first, then click to analyze" if st.session_state.data is None else "Run comprehensive quality checks on your data",
            key="run_analysis_main"
        ):
            if st.session_state.data is not None:
                # Log dictionary usage
                if st.session_state.dictionary:
                    dict_source = st.session_state.dictionary.get('source', 'Unknown')
                    dict_filename = st.session_state.dictionary.get('filename', 'Unknown')
                    num_rules = len(st.session_state.dictionary.get('rules', {}))
                    num_fields = len(st.session_state.dictionary.get('fields', []))

                    print("\n" + "="*80)
                    print("🔍 RUNNING DATA QUALITY ANALYSIS")
                    print(f"   Data: {len(st.session_state.data)} rows × {len(st.session_state.data.columns)} columns")
                    print(f"   Dictionary: {dict_filename} (source: {dict_source})")
                    print(f"   Rules: {num_rules}, Fields: {num_fields}")
                    print("="*80 + "\n")

                    st.info(f"📖 Using dictionary: **{dict_filename}** ({dict_source}) - {num_rules} rules, {num_fields} fields")
                else:
                    print("\n⚠️ RUNNING ANALYSIS WITHOUT DICTIONARY (auto-detection only)\n")
                    st.info("⚠️ No dictionary loaded - using auto-detection only")

                with st.spinner("🔍 Analyzing data quality... Please wait."):
                    try:
                        # Run analysis
                        results = asyncio.run(
                            st.session_state.mcp_client.analyze_data_quality(
                                st.session_state.data,
                                st.session_state.dictionary
                            )
                        )
                        st.session_state.analysis_results = results
                        st.success("✅ Analysis complete!")
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())

    st.markdown("---")

    # Display results if available
    if st.session_state.analysis_results:

        # Summary metrics
        st.subheader("📊 Analysis Summary")
        summary = st.session_state.analysis_results['summary']

        # Create columns with space for heatmap
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1.2, 1, 1, 1.5])
        with col1:
            st.metric("Total Rows", f"{summary['total_rows']:,}")
        with col2:
            st.metric("Total Columns", summary['total_columns'])
        with col3:
            st.metric("Issues Found", summary['issues_found'],
                     delta=None if summary['issues_found'] == 0 else f"{summary['critical_issues']} critical")
        with col4:
            st.metric("Warnings", summary['warnings'])
        with col5:
            st.metric("Completeness", f"{summary['completeness']}%")
        with col6:
            # Create issue heatmap visualization
            st.markdown("#### Issue Map")
            create_issue_heatmap(st.session_state.data, st.session_state.analysis_results['issues'])

        # Issues details
        if st.session_state.analysis_results['issues']:
            st.subheader("🔍 Issues Found")

            # Group issues by type
            issues_by_type = {}
            for issue in st.session_state.analysis_results['issues']:
                issue_type = issue['type']
                if issue_type not in issues_by_type:
                    issues_by_type[issue_type] = []
                issues_by_type[issue_type].append(issue)

            # Display issues by type - collapsed by default for cleaner look
            for issue_type, issues in issues_by_type.items():
                # Collapse by default for Missing Values and Invalid Values
                expand_by_default = issue_type not in ['missing_values', 'invalid_value']
                with st.expander(f"{issue_type.replace('_', ' ').title()} ({len(issues)} issues)", expanded=expand_by_default):
                    for issue in issues[:10]:  # Show first 10
                        if issue['severity'] == 'error':
                            st.error(f"❌ {issue['message']}")
                        else:
                            st.warning(f"⚠️ {issue['message']}")
                    if len(issues) > 10:
                        st.info(f"... and {len(issues) - 10} more")

        # Recommendations
        if st.session_state.analysis_results['recommendations']:
            st.subheader("💡 Recommendations")
            for rec in st.session_state.analysis_results['recommendations']:
                if rec['priority'] == 'critical':
                    st.error(f"🔴 **{rec['priority'].upper()}**: {rec['message']}")
                elif rec['priority'] == 'high':
                    st.warning(f"🟡 **{rec['priority'].upper()}**: {rec['message']}")
                else:
                    st.info(f"🔵 **{rec['priority'].upper()}**: {rec['message']}")

with tab2:
    st.title("About Data Quality Analyzer")

    st.markdown("""
    ### 🎯 Purpose
    The Data Quality Analyzer is a powerful tool designed to help you identify and resolve data quality issues
    in your datasets. It performs comprehensive checks to ensure your data meets quality standards.

    ### ✨ Features
    - **Multiple Format Support**: CSV, JSON, and TXT files
    - **Automatic Issue Detection**: Missing values, invalid entries, range violations
    - **Custom Validation Rules**: Define your own business rules via data dictionaries (JSON or PDF)
    - **Visual Reporting**: Clear metrics and issue summaries with interactive heatmaps
    - **Excel Export**: Highlighted cells showing exact error locations with comments
    - **Demo Data**: Built-in datasets for testing various validation scenarios
    - **Dictionary Caching**: Fast PDF dictionary parsing with automatic caching

    ### 🔍 What We Check
    1. **Missing Values**: Identifies null or empty cells
    2. **Invalid Values**: Detects entries like "invalid", "error", "n/a"
    3. **Data Type Validation**: Ensures values match expected types
    4. **Range Validation**: Checks if numeric values fall within specified ranges
    5. **Suspicious Values**: Flags test data or anomalous entries
    6. **Completeness**: Overall data completeness percentage

    ### 📚 How to Use
    1. **Upload your data** using the file uploader or select demo data
    2. **Optionally add a dictionary** (JSON or PDF) to define custom validation rules
    3. **Click Analyze** to run the quality checks
    4. **Review the results** including issues, recommendations, and visual heatmap
    5. **Export findings** to Excel (with highlighting) or JSON for further analysis

    ### 🛠 Technical Details
    Built with Streamlit and powered by the Model Context Protocol (MCP) for
    advanced data analysis capabilities. Features include:
    - Interactive Plotly visualizations
    - PDF parsing with PyPDF2
    - Excel generation with cell highlighting and comments
    - Efficient dictionary caching system

    ### 🔄 Data Flow Architecture
    """)

    # Load and render the Mermaid diagrams with selector
    try:
        # Load both diagrams
        with open('assets/data_flow_simple.mmd', 'r') as f:
            simple_diagram = f.read()
        with open('assets/data_flow_diagram.mmd', 'r') as f:
            detailed_diagram = f.read()

        st.info("📊 Interactive flowcharts showing the data analysis pipeline:")

        # Add diagram selector within the content area
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            diagram_view = st.radio(
                "Select diagram complexity:",
                ["🎯 Simple View", "🔬 Detailed View"],
                horizontal=True,
                label_visibility="visible"
            )

        # Render the selected diagram
        if diagram_view == "🎯 Simple View":
            render_mermaid(simple_diagram, height=400)
        else:
            render_mermaid(detailed_diagram, height=700)

    except FileNotFoundError as e:
        st.info(f"Data flow diagram not found: {str(e)}")
    except Exception as e:
        st.error(f"Error rendering diagram: {str(e)}")

    st.markdown("""
    ---
    *Version 2.1 - Enhanced UI with PDF Dictionary Support*
    """)
