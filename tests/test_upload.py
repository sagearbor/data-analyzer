"""
Tests for DataQualityAnalyzer (web_app.py) - the HTTP client wrapper around
the centralized data-analyzer REST API (POST /api/v1/analyze).

Option A (centralize the runtime): DataQualityAnalyzer no longer runs
QualityPipeline in-process; it POSTs the DataFrame to the API and maps the
JSON response back into the exact shape the Streamlit dashboard renders. All
`requests.post` calls are mocked here so these tests are deterministic,
offline, and don't require a running API server.

For end-to-end coverage of the actual engine (QualityPipeline/QualityChecker)
behind the endpoint, see tests/test_api.py's TestAnalyzeEndpoint (uses
FastAPI TestClient against the real engine, no live server needed either).
"""

import io
import json
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from web_app import DataQualityAnalyzer


def _mock_response(status_code=200, json_body=None, text=""):
    """Build a MagicMock standing in for a `requests.Response`."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body if json_body is not None else {}
    resp.text = text
    return resp


def _api_report(total_rows=0, total_columns=0, issues=None, recommendations=None,
                 quality_checks=None, summary_stats=None, data_types=None,
                 completeness=100.0):
    """Build a canned POST /api/v1/analyze response body (the exact shape
    api_server.py's QualityAnalysisResponse returns)."""
    issues = issues or []
    summary = {
        "total_rows": total_rows,
        "total_columns": total_columns,
        "issues_found": len(issues),
        "critical_issues": sum(1 for i in issues if i.get("severity") == "error"),
        "warnings": sum(1 for i in issues if i.get("severity") == "warning"),
        "data_types": data_types or {},
        "completeness": completeness,
    }
    return {
        "summary": summary,
        "issues": issues,
        "recommendations": recommendations or [],
        "quality_checks": quality_checks or {},
        "summary_stats": summary_stats or {},
    }


@pytest.fixture
def analyzer():
    return DataQualityAnalyzer()


@pytest.fixture
def sample_df(sample_csv_data):
    """sample_csv_data (from conftest.py) as a DataFrame, issues intact"""
    return pd.read_csv(io.StringIO(sample_csv_data))


# ============================================================================
# HTTP call construction
# ============================================================================


class TestApiRequestConstruction:
    """analyze_data_quality() should call the API with the right URL, headers,
    and multipart payload."""

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_posts_to_configured_base_url(self, mock_post, sample_df, monkeypatch):
        monkeypatch.setenv("DATA_ANALYZER_API_URL", "http://data-analyzer-api.internal:8000")
        monkeypatch.setenv("DATA_ANALYZER_API_KEY", "secret-key")
        analyzer = DataQualityAnalyzer()  # re-read env after monkeypatch

        mock_post.return_value = _mock_response(200, _api_report(
            total_rows=len(sample_df), total_columns=len(sample_df.columns)
        ))

        await analyzer.analyze_data_quality(sample_df, None)

        assert mock_post.called
        args, kwargs = mock_post.call_args
        assert args[0] == "http://data-analyzer-api.internal:8000/api/v1/analyze"
        assert kwargs["headers"] == {"X-API-Key": "secret-key"}

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_defaults_to_localhost_when_env_unset(self, mock_post, sample_df, monkeypatch):
        monkeypatch.delenv("DATA_ANALYZER_API_URL", raising=False)
        monkeypatch.delenv("DATA_ANALYZER_API_KEY", raising=False)
        analyzer = DataQualityAnalyzer()

        mock_post.return_value = _mock_response(200, _api_report(
            total_rows=len(sample_df), total_columns=len(sample_df.columns)
        ))

        await analyzer.analyze_data_quality(sample_df, None)

        args, kwargs = mock_post.call_args
        assert args[0] == "http://localhost:8000/api/v1/analyze"
        # No API key configured -> no X-API-Key header sent
        assert kwargs["headers"] == {}

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_sends_dataframe_as_csv_upload(self, mock_post, analyzer, sample_df):
        mock_post.return_value = _mock_response(200, _api_report(
            total_rows=len(sample_df), total_columns=len(sample_df.columns)
        ))

        await analyzer.analyze_data_quality(sample_df, None)

        _, kwargs = mock_post.call_args
        filename, content, content_type = kwargs["files"]["data_file"]
        assert filename == "data.csv"
        assert content_type == "text/csv"
        # Round-trips back to an equivalent DataFrame
        roundtrip = pd.read_csv(io.BytesIO(content))
        assert list(roundtrip.columns) == list(sample_df.columns)
        assert len(roundtrip) == len(sample_df)

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_sends_schema_and_rules_as_json_form_fields(self, mock_post, analyzer, mock_schema, mock_rules):
        df = pd.DataFrame({"age": [30], "salary": [50000.0], "department": ["Engineering"]})
        dictionary = {"schema": mock_schema, "validation_rules": mock_rules}

        mock_post.return_value = _mock_response(200, _api_report(total_rows=1, total_columns=3))

        await analyzer.analyze_data_quality(df, dictionary)

        _, kwargs = mock_post.call_args
        sent_schema = json.loads(kwargs["data"]["schema"])
        sent_rules = json.loads(kwargs["data"]["rules"])
        assert sent_schema == mock_schema
        assert sent_rules == mock_rules

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_omits_schema_and_rules_when_no_dictionary(self, mock_post, analyzer, sample_df):
        mock_post.return_value = _mock_response(200, _api_report(
            total_rows=len(sample_df), total_columns=len(sample_df.columns)
        ))

        await analyzer.analyze_data_quality(sample_df, None)

        _, kwargs = mock_post.call_args
        assert "schema" not in kwargs["data"]
        assert "rules" not in kwargs["data"]

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_sends_xlsx_upload_with_excel_content_type(self, mock_post, analyzer, sample_df):
        """When source_format='xlsx' (set after an .xlsx upload in the UI),
        the DataFrame should be re-encoded as a real Excel workbook and
        uploaded with the openxmlformats content-type, not CSV - so
        api_server.py's extension-based dispatch routes it to
        DataLoader.load_excel instead of pandas.read_csv."""
        mock_post.return_value = _mock_response(200, _api_report(
            total_rows=len(sample_df), total_columns=len(sample_df.columns)
        ))

        await analyzer.analyze_data_quality(sample_df, None, source_format="xlsx")

        _, kwargs = mock_post.call_args
        filename, content, content_type = kwargs["files"]["data_file"]
        assert filename == "data.xlsx"
        assert content_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        # Round-trips back to an equivalent DataFrame via a real Excel parse
        roundtrip = pd.read_excel(io.BytesIO(content))
        assert list(roundtrip.columns) == list(sample_df.columns)
        assert len(roundtrip) == len(sample_df)

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_sends_xls_upload_with_legacy_excel_content_type(self, mock_post, analyzer, sample_df):
        """source_format='xls' should use the legacy application/vnd.ms-excel
        content-type (still encoded as an .xlsx workbook under the hood,
        since openpyxl can only write .xlsx - api_server.py dispatches by
        filename extension, not by parsing the binary format)."""
        mock_post.return_value = _mock_response(200, _api_report(
            total_rows=len(sample_df), total_columns=len(sample_df.columns)
        ))

        await analyzer.analyze_data_quality(sample_df, None, source_format="xls")

        _, kwargs = mock_post.call_args
        filename, content, content_type = kwargs["files"]["data_file"]
        assert filename == "data.xls"
        assert content_type == "application/vnd.ms-excel"

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_source_format_none_defaults_to_csv(self, mock_post, analyzer, sample_df):
        """No source_format (e.g. demo data, CSV/JSON/TSV uploads) should
        preserve the original CSV upload behavior."""
        mock_post.return_value = _mock_response(200, _api_report(
            total_rows=len(sample_df), total_columns=len(sample_df.columns)
        ))

        await analyzer.analyze_data_quality(sample_df, None, source_format=None)

        _, kwargs = mock_post.call_args
        filename, content, content_type = kwargs["files"]["data_file"]
        assert filename == "data.csv"
        assert content_type == "text/csv"


# ============================================================================
# Response mapping
# ============================================================================


class TestResponseMapping:
    """The JSON body from /api/v1/analyze should map 1:1 onto the dict shape
    the rest of web_app.py (dashboard rendering code) consumes."""

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_returns_expected_top_level_keys(self, mock_post, analyzer, sample_df):
        mock_post.return_value = _mock_response(200, _api_report(
            total_rows=len(sample_df), total_columns=len(sample_df.columns)
        ))

        results = await analyzer.analyze_data_quality(sample_df, None)

        assert set(results.keys()) == {
            "summary", "issues", "recommendations", "quality_checks", "summary_stats"
        }

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_summary_row_and_column_counts_pass_through(self, mock_post, analyzer, sample_df):
        mock_post.return_value = _mock_response(200, _api_report(
            total_rows=len(sample_df), total_columns=len(sample_df.columns)
        ))

        results = await analyzer.analyze_data_quality(sample_df, None)

        assert results["summary"]["total_rows"] == len(sample_df)
        assert results["summary"]["total_columns"] == len(sample_df.columns)

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_issues_pass_through_unchanged(self, mock_post, analyzer, sample_df):
        api_issues = [
            {
                "type": "range_violation", "severity": "error", "column": "salary",
                "row": 6, "value": 999999, "message": "Value 999999 in column 'salary' violates rule: max <= 200000"
            },
            {
                "type": "missing_values", "severity": "warning", "column": "name",
                "count": 1, "percentage": 10.0, "message": "Column 'name' has 1 missing values (10.0%)"
            },
        ]
        mock_post.return_value = _mock_response(200, _api_report(
            total_rows=len(sample_df), total_columns=len(sample_df.columns), issues=api_issues
        ))

        results = await analyzer.analyze_data_quality(sample_df, None)

        assert results["issues"] == api_issues
        assert results["summary"]["issues_found"] == 2
        assert results["summary"]["critical_issues"] == 1
        assert results["summary"]["warnings"] == 1

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_quality_checks_and_summary_stats_pass_through(self, mock_post, analyzer, sample_df):
        qc = {"row_count": {"check": "row_count", "passed": True}}
        stats = {"shape": {"rows": len(sample_df), "columns": len(sample_df.columns)}}
        mock_post.return_value = _mock_response(200, _api_report(
            total_rows=len(sample_df), total_columns=len(sample_df.columns),
            quality_checks=qc, summary_stats=stats
        ))

        results = await analyzer.analyze_data_quality(sample_df, None)

        assert results["quality_checks"] == qc
        assert results["summary_stats"] == stats

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_clean_data_has_no_issues(self, mock_post, analyzer):
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"], "age": [30, 31, 32]})
        mock_post.return_value = _mock_response(200, _api_report(total_rows=3, total_columns=3))

        results = await analyzer.analyze_data_quality(df, None)

        assert results["issues"] == []
        assert results["summary"]["issues_found"] == 0
        assert results["summary"]["completeness"] == 100.0


# ============================================================================
# Local conditional-logic validation (not part of the REST engine)
# ============================================================================


class TestLogicValidationMerge:
    """Conditional logic validation (RuleExtractor/LogicValidator) is not part
    of /api/v1/analyze - it stays local and its violations get merged into
    the API's issues/summary after the HTTP call returns."""

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_logic_violations_count_present_without_fields(self, mock_post, analyzer, sample_df):
        """No dictionary['fields'] -> logic validation is skipped, count is 0"""
        mock_post.return_value = _mock_response(200, _api_report(
            total_rows=len(sample_df), total_columns=len(sample_df.columns)
        ))

        results = await analyzer.analyze_data_quality(sample_df, {"rules": {}})

        assert results["summary"]["logic_violations_count"] == 0


# ============================================================================
# Error handling
# ============================================================================


class TestApiErrorHandling:
    """analyze_data_quality() should raise RuntimeError (not crash) on any
    API failure, so callers can render it via st.error(...)."""

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_non_200_response_raises_runtime_error(self, mock_post, analyzer, sample_df):
        mock_post.return_value = _mock_response(
            400, {"error": "Invalid input", "detail": "Invalid 'schema' JSON: bad"}
        )

        with pytest.raises(RuntimeError, match="400"):
            await analyzer.analyze_data_quality(sample_df, None)

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_connection_error_raises_runtime_error(self, mock_post, analyzer, sample_df):
        import requests as requests_module
        mock_post.side_effect = requests_module.exceptions.ConnectionError("refused")

        with pytest.raises(RuntimeError, match="Could not connect"):
            await analyzer.analyze_data_quality(sample_df, None)

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_timeout_raises_runtime_error(self, mock_post, analyzer, sample_df):
        import requests as requests_module
        mock_post.side_effect = requests_module.exceptions.Timeout("timed out")

        with pytest.raises(RuntimeError, match="timed out"):
            await analyzer.analyze_data_quality(sample_df, None)

    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_non_json_response_raises_runtime_error(self, mock_post, analyzer, sample_df):
        resp = MagicMock()
        resp.status_code = 200
        resp.json.side_effect = ValueError("not json")
        mock_post.return_value = resp

        with pytest.raises(RuntimeError, match="non-JSON"):
            await analyzer.analyze_data_quality(sample_df, None)


# ============================================================================
# End-to-end: file loaded from disk, then analyzed via the mocked API
# ============================================================================


class TestAnalyzeFromFile:
    @pytest.mark.asyncio
    @patch("web_app.requests.post")
    async def test_analyze_csv_loaded_from_disk(self, mock_post, analyzer, create_test_csv):
        df = pd.read_csv(create_test_csv)
        mock_post.return_value = _mock_response(200, _api_report(
            total_rows=len(df), total_columns=len(df.columns)
        ))

        results = await analyzer.analyze_data_quality(df, None)

        assert results["summary"]["total_rows"] == len(df)
        assert isinstance(results["issues"], list)
