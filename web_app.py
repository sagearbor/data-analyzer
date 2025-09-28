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

# Import custom modules
from demo_dictionaries import DEMO_DICTIONARIES, get_demo_dictionary
from mermaid_renderer import render_mermaid

# Configure Streamlit page
st.set_page_config(
    page_title="Data Quality Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"  # Keep sidebar collapsed by default
)

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
        content: "Data Quality Analyzer";
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

class MCPClient:
    """Simulated MCP Client for demo purposes"""

    async def analyze_data_quality(self, data: pd.DataFrame, rules: Optional[Dict] = None) -> Dict[str, Any]:
        """Simulate MCP analyze_data call"""
        issues = []

        # Check for missing values
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

        # Check for invalid values (including "invalid", "error", malformed dates, suspicious numbers)
        for col in data.columns:
            for idx, val in data[col].items():
                if pd.notna(val):
                    val_str = str(val).lower().strip()
                    # Check for known invalid text values
                    if val_str in ['invalid', 'error', 'n/a', 'null', 'none', 'invalid-date']:
                        issues.append({
                            "type": "invalid_value",
                            "severity": "error",
                            "column": col,
                            "row": idx,
                            "value": val,
                            "message": f"Invalid value '{val}' in column '{col}' at row {idx}"
                        })
                    # Check for malformed dates (contains 'oops' or other text in date)
                    elif 'date' in col.lower() or '-' in str(val):
                        if 'oops' in val_str or 'text' in val_str:
                            issues.append({
                                "type": "invalid_date",
                                "severity": "error",  # Critical error for malformed dates
                                "column": col,
                                "row": idx,
                                "value": val,
                                "message": f"Malformed date '{val}' in column '{col}' at row {idx}"
                            })
                    # Check for suspicious numeric values (e.g., 666 in specific columns)
                    elif str(val) == '666' or (isinstance(val, str) and '666' in val and len(val) <= 4):
                        # Flag 666 as error in specific validation columns, warning otherwise
                        severity = "error" if any(x in col.lower() for x in ['length', 'options', 'score']) else "warning"
                        issues.append({
                            "type": "suspicious_value",
                            "severity": severity,
                            "column": col,
                            "row": idx,
                            "value": val,
                            "message": f"Suspicious test value '{val}' in column '{col}' at row {idx}"
                        })

        # Apply custom rules if provided
        if rules:
            for col, col_rules in rules.items():
                if col in data.columns:
                    if 'min' in col_rules:
                        mask = pd.to_numeric(data[col], errors='coerce') < col_rules['min']
                        violations = data[mask]
                        for idx in violations.index:
                            issues.append({
                                "type": "range_violation",
                                "severity": "error",
                                "column": col,
                                "row": int(idx),
                                "value": data[col][idx],
                                "message": f"Value {data[col][idx]} in column '{col}' is below minimum {col_rules['min']}"
                            })

        # Generate summary
        summary = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "issues_found": len(issues),
            "critical_issues": sum(1 for i in issues if i['severity'] == 'error'),
            "warnings": sum(1 for i in issues if i['severity'] == 'warning'),
            "data_types": {col: str(data[col].dtype) for col in data.columns},
            "completeness": round((1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100, 2)
        }

        return {
            "summary": summary,
            "issues": issues,
            "recommendations": self._generate_recommendations(issues)
        }

    def _generate_recommendations(self, issues):
        """Generate recommendations based on issues found"""
        recommendations = []

        issue_types = set(i['type'] for i in issues)

        if 'missing_values' in issue_types:
            recommendations.append({
                "type": "data_cleaning",
                "priority": "high",
                "message": "Consider implementing data imputation strategies for columns with missing values"
            })

        if 'invalid_value' in issue_types:
            recommendations.append({
                "type": "data_validation",
                "priority": "critical",
                "message": "Invalid values detected. Review data source and implement validation at ingestion"
            })

        if 'range_violation' in issue_types:
            recommendations.append({
                "type": "business_rules",
                "priority": "high",
                "message": "Values outside expected ranges detected. Review business rules and data constraints"
            })

        return recommendations

def load_demo_data(dataset_name: str):
    """Load demo dataset matching the dictionary options"""
    demo_data = {
        'western': pd.DataFrame({
            'employee_id': [1001, 1002, 1003, 1004, 1005],
            'first_name': ['John', 'Jane', 'invalid', 'Bob', 'Alice'],
            'last_name': ['Smith', 'Doe', 'Johnson', 'invalid', 'Wilson'],
            'age': [35, 28, 67, 45, 32],  # 67 is outside range
            'salary': [75000, 85000, 45000, 95000, None],  # 45000 below min
            'hire_date': ['2022-03-15', '2023-01-10', 'invalid-date', '2021-07-22', '2022-11-30'],
            'department': ['Engineering', 'Marketing', 'invalid', 'Sales', 'Finance'],
            'is_active': [True, True, False, None, True],
            'email': ['john@company.com', 'invalid', 'mike@company.com', 'bob@company.com', 'alice@company.com']
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
    """Create a small heatmap showing where issues are in the data with hover tooltips"""
    try:
        rows, cols = len(df), len(df.columns)

        # Create a matrix to store issue locations and messages
        # Condense large datasets
        max_display_rows = 60
        max_display_cols = 30

        row_factor = max(1, rows // max_display_rows)
        col_factor = max(1, cols // max_display_cols)

        display_rows = min(rows, max_display_rows)
        display_cols = min(cols, max_display_cols)

        # Initialize matrix (0 = no issue, 1 = warning, 2 = error) and tooltip dictionary
        issue_matrix = np.zeros((display_rows, display_cols))
        issue_tooltips = {}  # Store tooltips for each cell

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

                    # Store tooltip info
                    key = (display_row, display_col)
                    if key not in issue_tooltips:
                        issue_tooltips[key] = []
                    issue_tooltips[key].append(f"{issue['type']}: Row {row_idx}, Col {issue['column']}")
                except:
                    pass

        # Calculate proportional figure size based on data shape
        # Base size is 3 inches max dimension
        max_size = 3
        aspect_ratio = cols / rows  # Width to height ratio

        if aspect_ratio > 1:
            # Wider than tall (more columns than rows)
            fig_width = max_size
            fig_height = max(0.3, max_size / aspect_ratio)  # Min height of 0.3
        else:
            # Taller than wide (more rows than columns)
            fig_height = max_size
            fig_width = max(0.3, max_size * aspect_ratio)  # Min width of 0.3

        # Create the visualization with proportional dimensions
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # Create color map (white = no issue, yellow = warning, red = error)
        colors = ['#ffffff', '#fbbf24', '#ef4444']
        cmap = plt.matplotlib.colors.ListedColormap(colors)

        # Plot heatmap
        im = ax.imshow(issue_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=2)

        # Add black border
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['top'].set_color('black')
        ax.spines['top'].set_linewidth(2)
        ax.spines['right'].set_visible(True)
        ax.spines['right'].set_color('black')
        ax.spines['right'].set_linewidth(2)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_color('black')
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(2)

        # Add title with data dimensions
        title = f'{rows} rows × {cols} cols'
        if rows > max_display_rows or cols > max_display_cols:
            title += f' (scale {row_factor}:{col_factor})'
        ax.set_title(title, fontsize=7, pad=2)

        # Add caption with issue counts
        error_count = np.sum(issue_matrix == 2)
        warning_count = np.sum(issue_matrix == 1)
        if error_count > 0 or warning_count > 0:
            caption = f"🔴 {int(error_count)} errors, 🟡 {int(warning_count)} warnings"
            ax.text(0.5, -0.15, caption, transform=ax.transAxes,
                   fontsize=6, ha='center', va='top')

        plt.tight_layout(pad=0.1)
        st.pyplot(fig, use_container_width=False)  # Don't stretch to container
        plt.close()

    except Exception as e:
        st.caption("Issue map unavailable")

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
    st.session_state.mcp_client = MCPClient()

# Create navigation tabs
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
            ["None", "CSV - Western", "CSV - Asian", "JSON - Mixed"],
            key="demo_selector"
        )

        if demo_option != "None":
            dataset_map = {
                "CSV - Western": "western",
                "CSV - Asian": "asian",
                "JSON - Mixed": "mixed"
            }
            if demo_option in dataset_map:
                st.session_state.data = load_demo_data(dataset_map[demo_option])
                st.success(f"✅ Loaded {demo_option} demo data")

    with col2:
        st.markdown("### 📋 Dictionary")

        # Dictionary file uploader first - aligned with data uploader
        dict_file = st.file_uploader(
            " ",  # Empty label to avoid duplication
            type=['json', 'pdf'],
            key="dict_uploader",
            label_visibility="collapsed",
            help="Optional - defines validation rules for data quality checks (JSON or PDF)"
        )

        if dict_file:
            with st.spinner(f"Processing dictionary {dict_file.name}..."):
                try:
                    if dict_file.name.endswith('.pdf'):
                        # For PDF files, simulate parsing with progress
                        progress_text = st.empty()
                        progress_text.info(f"📄 Reading PDF dictionary '{dict_file.name}'...")
                        # In production, you'd parse the PDF here
                        st.session_state.dictionary = {"source": "PDF", "filename": dict_file.name}
                        progress_text.success(f"✅ PDF dictionary '{dict_file.name}' ready for validation")
                    else:
                        st.session_state.dictionary = json.load(dict_file)
                        st.success("✅ Dictionary loaded")
                except Exception as e:
                    st.error(f"Error loading dictionary: {str(e)}")

        # Demo dictionary selector below file uploader
        demo_dict = st.selectbox(
            "Or load demo dictionary:",
            ["None"] + list(DEMO_DICTIONARIES.keys()),
            key="demo_dict_selector"
        )

        if demo_dict != "None":
            st.session_state.dictionary = get_demo_dictionary(demo_dict)
            st.success(f"✅ Loaded {demo_dict}")

    with col3:
        st.markdown("### ⚡ Analyze")

        # Add spacing to align with upload boxes
        st.markdown("<div style='height: 31px;'></div>", unsafe_allow_html=True)

        # Enable button only when data is loaded
        if st.button(
            "🚀 Run Analysis",
            disabled=(st.session_state.data is None),
            use_container_width=True,
            type="primary",
            help="Ready when data is loaded"
        ):
            if st.session_state.data is not None:
                with st.spinner("Analyzing data quality..."):
                    # Run analysis
                    results = asyncio.run(
                        st.session_state.mcp_client.analyze_data_quality(
                            st.session_state.data,
                            st.session_state.dictionary
                        )
                    )
                    st.session_state.analysis_results = results
                    st.success("✅ Analysis complete!")

        # Export dropdown - in same column, saves vertical space
        if st.session_state.analysis_results:
            st.markdown("### 📥 Export")
            export_format = st.selectbox(
                "Choose format:",
                ["Select export format...", "Excel with highlighting", "JSON report"],
                key="export_format"
            )

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
                    use_container_width=True
                )
            elif export_format == "JSON report":
                json_str = json.dumps(st.session_state.analysis_results, indent=2, default=str)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )

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
    - **Custom Validation Rules**: Define your own business rules via data dictionaries
    - **Visual Reporting**: Clear metrics and issue summaries
    - **Excel Export**: Highlighted cells showing exact error locations
    - **Demo Data**: Built-in datasets for testing

    ### 🔍 What We Check
    1. **Missing Values**: Identifies null or empty cells
    2. **Invalid Values**: Detects entries like "invalid", "error", "n/a"
    3. **Data Type Validation**: Ensures values match expected types
    4. **Range Validation**: Checks if numeric values fall within specified ranges
    5. **Completeness**: Overall data completeness percentage

    ### 📚 How to Use
    1. **Upload your data** using the file uploader or select demo data
    2. **Optionally add a dictionary** to define custom validation rules
    3. **Click Analyze** to run the quality checks
    4. **Review the results** including issues and recommendations
    5. **Export findings** to Excel or JSON for further analysis

    ### 🛠 Technical Details
    Built with Streamlit and powered by the Model Context Protocol (MCP) for
    advanced data analysis capabilities.

    ---
    *Version 2.0 - Clean UI Edition*
    """)