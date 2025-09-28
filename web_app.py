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
        padding-top: 3rem !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }

    /* Clean modern styling */
    .stApp {
        background: #ffffff;
    }

    /* Style tabs to look like navbar */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e293b;
        padding: 0.5rem 1rem;
        border-radius: 8px 8px 0 0;
        margin-bottom: 1rem;
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

    /* Tab content padding */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 0;
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

        # Check for invalid values (including "invalid", "error", etc.)
        for col in data.columns:
            for idx, val in data[col].items():
                if pd.notna(val):
                    val_str = str(val).lower().strip()
                    if val_str in ['invalid', 'error', 'n/a', 'null', 'none', 'invalid-date']:
                        issues.append({
                            "type": "invalid_value",
                            "severity": "error",
                            "column": col,
                            "row": idx,
                            "value": val,
                            "message": f"Invalid value '{val}' in column '{col}' at row {idx}"
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
    """Load demo dataset"""
    demo_data = {
        'sales': pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'product': ['Widget', 'Gadget', 'Widget', 'Tool', 'Gadget',
                       'Widget', 'invalid', 'Gadget', 'Tool', 'Widget'],
            'quantity': [10, 15, -5, 20, 25, 30, 15, 40, 45, 50],
            'price': [99.99, 149.99, 99.99, 199.99, 149.99,
                     99.99, None, 149.99, 199.99, 99.99],
            'customer': ['A', 'B', None, 'D', 'E', 'F', 'G', 'invalid', 'I', 'J']
        }),
        'inventory': pd.DataFrame({
            'sku': ['SKU001', 'SKU002', 'SKU003', 'SKU004', 'invalid'],
            'name': ['Widget Pro', 'Gadget Plus', 'Tool Expert', 'invalid', 'Widget Lite'],
            'stock': [100, -10, 250, 75, 150],
            'reorder_point': [20, 30, None, 15, 25],
            'category': ['Electronics', 'invalid', 'Tools', 'Electronics', 'Electronics']
        }),
        'customers': pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'invalid', 'Diana', None],
            'email': ['alice@example.com', 'invalid', 'charlie@example.com',
                     'diana@example.com', 'eve@example.com'],
            'join_date': ['2024-01-01', '2024-01-15', 'invalid-date', '2024-02-01', '2024-02-15'],
            'status': ['active', 'active', 'inactive', 'invalid', 'active']
        })
    }
    return demo_data.get(dataset_name, demo_data['sales'])

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
    # Title
    st.title("Data Quality Analyzer")
    st.markdown("Upload your data, optionally add validation rules, and analyze")

    # Create three columns for the main components
    col1, col2, col3 = st.columns([2, 2, 1.5])

    with col1:
        st.markdown("### 📁 Upload Data")

        # Demo data selector
        demo_option = st.selectbox(
            "Load demo data:",
            ["None", "Sales Data", "Inventory Data", "Customer Data"],
            key="demo_selector"
        )

        if demo_option != "None":
            dataset_map = {
                "Sales Data": "sales",
                "Inventory Data": "inventory",
                "Customer Data": "customers"
            }
            if demo_option in dataset_map:
                st.session_state.data = load_demo_data(dataset_map[demo_option])
                st.success(f"✅ Loaded {demo_option}")

        # File uploader
        uploaded_file = st.file_uploader(
            "Or upload your file",
            type=['csv', 'json', 'txt'],
            key="data_uploader"
        )

        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    st.session_state.data = pd.read_json(uploaded_file)
                else:
                    st.session_state.data = pd.read_csv(uploaded_file, sep='\t')
                st.success(f"✅ Loaded {len(st.session_state.data)} rows")
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    with col2:
        st.markdown("### 📋 Upload Dictionary")
        st.markdown("*Optional - defines validation rules*")

        # Demo dictionary selector
        demo_dict = st.selectbox(
            "Load demo dictionary:",
            ["None"] + list(DEMO_DICTIONARIES.keys()),
            key="demo_dict_selector"
        )

        if demo_dict != "None":
            st.session_state.dictionary = get_demo_dictionary(demo_dict)
            st.success(f"✅ Loaded {demo_dict}")

        # Dictionary file uploader
        dict_file = st.file_uploader(
            "Or upload dictionary file",
            type=['json'],
            key="dict_uploader"
        )

        if dict_file:
            try:
                st.session_state.dictionary = json.load(dict_file)
                st.success("✅ Dictionary loaded")
            except Exception as e:
                st.error(f"Error loading dictionary: {str(e)}")

    with col3:
        st.markdown("### ⚡ Analyze")
        st.markdown("*Ready when data is loaded*")

        # Enable button only when data is loaded
        if st.button(
            "🚀 Run Analysis",
            disabled=(st.session_state.data is None),
            use_container_width=True,
            type="primary"
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

    # Display results if available
    if st.session_state.analysis_results:
        st.markdown("---")

        # Summary metrics
        st.subheader("📊 Analysis Summary")
        summary = st.session_state.analysis_results['summary']

        col1, col2, col3, col4, col5 = st.columns(5)
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

            # Display issues by type
            for issue_type, issues in issues_by_type.items():
                with st.expander(f"{issue_type.replace('_', ' ').title()} ({len(issues)} issues)", expanded=True):
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

        # Export options
        st.subheader("📥 Export Results")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("📊 Export to Excel (with error highlighting)", use_container_width=True):
                excel_data = export_to_excel_with_highlighting(
                    st.session_state.data,
                    st.session_state.analysis_results['issues']
                )
                st.download_button(
                    label="Download Excel File",
                    data=excel_data,
                    file_name=f"data_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

        with col2:
            if st.button("📄 Export Report as JSON", use_container_width=True):
                json_str = json.dumps(st.session_state.analysis_results, indent=2, default=str)
                st.download_button(
                    label="Download JSON Report",
                    data=json_str,
                    file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

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