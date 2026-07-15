"""
Tests for the document/data upload and analysis flow via DataQualityAnalyzer
(web_app.py), replacing the old test_upload.py.disabled script.

DataQualityAnalyzer wraps the real QualityPipeline/QualityChecker logic
(mcp_server.py) without needing a running MCP server or Streamlit runtime;
web_app.py is importable in "bare mode" (no ScriptRunContext) since its
module-level Streamlit calls (st.set_page_config, etc.) only log warnings
outside of `streamlit run`.
"""

import io

import pandas as pd
import pytest

from web_app import DataQualityAnalyzer


@pytest.fixture
def analyzer():
    return DataQualityAnalyzer()


@pytest.fixture
def sample_df(sample_csv_data):
    """sample_csv_data (from conftest.py) as a DataFrame, issues intact"""
    return pd.read_csv(io.StringIO(sample_csv_data))


class TestAnalyzeWithoutDictionary:
    """analyze_data_quality() with no dictionary should still surface basic issues"""

    @pytest.mark.asyncio
    async def test_returns_expected_top_level_keys(self, analyzer, sample_df):
        results = await analyzer.analyze_data_quality(sample_df, None)

        assert set(results.keys()) >= {"summary", "issues", "recommendations", "quality_checks", "summary_stats"}

    @pytest.mark.asyncio
    async def test_summary_row_and_column_counts(self, analyzer, sample_df):
        results = await analyzer.analyze_data_quality(sample_df, None)

        summary = results["summary"]
        assert summary["total_rows"] == len(sample_df)
        assert summary["total_columns"] == len(sample_df.columns)

    @pytest.mark.asyncio
    async def test_detects_missing_values(self, analyzer, sample_df):
        """sample_csv_data has a blank name (row 6) and blank salary (row 7)"""
        results = await analyzer.analyze_data_quality(sample_df, None)

        missing_issues = [i for i in results["issues"] if i["type"] == "missing_values"]
        columns_with_missing = {i["column"] for i in missing_issues}

        assert "name" in columns_with_missing
        assert "salary" in columns_with_missing
        assert results["summary"]["warnings"] >= len(missing_issues)

    @pytest.mark.asyncio
    async def test_recommendations_include_data_cleaning_for_missing_values(self, analyzer, sample_df):
        results = await analyzer.analyze_data_quality(sample_df, None)

        recommendation_types = {r["type"] for r in results["recommendations"]}
        assert "data_cleaning" in recommendation_types


class TestAnalyzeWithDictionary:
    """analyze_data_quality() with a schema + validation rules dictionary"""

    @pytest.mark.asyncio
    async def test_schema_and_rules_accepted(self, analyzer, sample_df, mock_schema, mock_rules):
        dictionary = {"schema": mock_schema, "validation_rules": mock_rules}

        results = await analyzer.analyze_data_quality(sample_df, dictionary)

        assert results["summary"]["total_rows"] == len(sample_df)
        # data_types should reflect the dtypes pandas actually inferred
        assert set(results["summary"]["data_types"].keys()) == set(sample_df.columns)

    @pytest.mark.asyncio
    async def test_out_of_range_value_flagged(self, analyzer, mock_schema):
        """A salary far outside mock_rules' [30000, 200000] range should be flagged"""
        df = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "age": [30, 31],
            "department": ["Engineering", "Engineering"],
            "salary": [50000.0, 999999.0],  # second row violates max
            "hire_date": ["2022-01-01", "2022-01-02"],
            "is_active": [True, True],
        })
        rules = {"salary": {"min": 30000, "max": 200000}}
        dictionary = {"schema": mock_schema, "validation_rules": rules}

        results = await analyzer.analyze_data_quality(df, dictionary)

        range_issues = [i for i in results["issues"] if i["type"] == "range_violation"]
        assert any(i["column"] == "salary" for i in range_issues)
        assert results["summary"]["critical_issues"] >= 1

    @pytest.mark.asyncio
    async def test_disallowed_categorical_value_flagged(self, analyzer, mock_schema):
        """A department value outside mock_rules' allowed list should be flagged"""
        df = pd.DataFrame({
            "id": [1, 2],
            "name": ["Alice", "Bob"],
            "age": [30, 31],
            "department": ["Engineering", "NotARealDept"],
            "salary": [50000.0, 60000.0],
            "hire_date": ["2022-01-01", "2022-01-02"],
            "is_active": [True, True],
        })
        rules = {"department": {"allowed": ["Engineering", "Marketing", "HR", "Sales"]}}
        dictionary = {"schema": mock_schema, "validation_rules": rules}

        results = await analyzer.analyze_data_quality(df, dictionary)

        categorical_issues = [i for i in results["issues"] if i["type"] == "invalid_categorical_value"]
        assert any(i["column"] == "department" for i in categorical_issues)


class TestAnalyzeCleanData:
    """A dataset with no missing values or violations should produce no issues"""

    @pytest.mark.asyncio
    async def test_clean_data_has_no_issues(self, analyzer):
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Carol"],
            "age": [30, 31, 32],
        })

        results = await analyzer.analyze_data_quality(df, None)

        assert results["issues"] == []
        assert results["summary"]["issues_found"] == 0
        assert results["summary"]["completeness"] == 100.0


class TestAnalyzeFromFile:
    """End-to-end: read an uploaded CSV file from disk, then analyze it"""

    @pytest.mark.asyncio
    async def test_analyze_csv_loaded_from_disk(self, analyzer, create_test_csv):
        df = pd.read_csv(create_test_csv)

        results = await analyzer.analyze_data_quality(df, None)

        assert results["summary"]["total_rows"] == len(df)
        assert isinstance(results["issues"], list)
