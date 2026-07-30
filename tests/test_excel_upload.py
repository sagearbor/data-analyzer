"""
Tests for Excel (.xlsx) upload support.

Excel loading is implemented once in mcp_server.DataLoader.load_excel (the
shared loader that mcp_server, the MCP tools, and web_app.py all call), so
these tests exercise that shared entry point directly plus the same
in-memory-bytes calling convention web_app.py uses for uploaded files
(``uploaded_file.read()`` -> ``DataLoader.load_excel(bytes)``).
"""

import io

import openpyxl
import pandas as pd
import pytest

from mcp_server import DataLoader


def _workbook_bytes(rows, headers=None) -> bytes:
    """Build a minimal valid .xlsx workbook in memory and return its bytes.

    rows: list of row tuples/lists to write below the header row.
    headers: optional header row; if omitted and rows is empty, the sheet
    is written with no rows at all (fully empty worksheet).
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    if headers is not None:
        ws.append(headers)
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


@pytest.mark.unit
class TestExcelUploadValid:
    """A well-formed .xlsx workbook should load into a correct DataFrame."""

    def test_load_excel_from_bytes_returns_dataframe(self):
        data = _workbook_bytes(
            rows=[(1, "Alice", 25), (2, "Bob", 30), (3, "Charlie", 35)],
            headers=["id", "name", "age"],
        )

        df = DataLoader.load_excel(data)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["id", "name", "age"]
        assert df.shape == (3, 3)
        assert df.iloc[0]["name"] == "Alice"
        assert df.iloc[2]["age"] == 35

    def test_load_excel_matches_web_app_upload_pattern(self):
        """web_app.py calls DataLoader.load_excel(uploaded_file.read()); verify
        that exact bytes-in-memory calling convention works end to end."""
        data = _workbook_bytes(
            rows=[("P001", 45), ("P002", 62)],
            headers=["patient_id", "age"],
        )
        uploaded_file = io.BytesIO(data)  # stands in for Streamlit's UploadedFile

        df = DataLoader.load_excel(uploaded_file.read())

        assert df.shape == (2, 2)
        assert list(df.columns) == ["patient_id", "age"]

    def test_load_excel_via_load_data_wrapper(self):
        """DataLoader.load_data(..., file_format='xlsx') dispatches to load_excel."""
        data = _workbook_bytes(
            rows=[(1, "x"), (2, "y")],
            headers=["id", "value"],
        )

        df = DataLoader.load_data(data, file_format="xlsx")

        assert df.shape == (2, 2)

    def test_load_excel_preserves_numeric_and_string_types(self):
        data = _workbook_bytes(
            rows=[(1, 19.99, "Widget A"), (2, 29.99, "Widget B")],
            headers=["product_id", "price", "product_name"],
        )

        df = DataLoader.load_excel(data)

        assert df["product_id"].tolist() == [1, 2]
        assert df["price"].tolist() == [19.99, 29.99]
        assert df["product_name"].tolist() == ["Widget A", "Widget B"]


@pytest.mark.unit
class TestExcelUploadMalformed:
    """Malformed/empty Excel input should be handled the same way the CSV
    path handles equivalent failures: DataLoader raises ValueError for
    content that can't be parsed as a workbook at all, and returns an
    empty-but-valid DataFrame when the workbook is valid but has no data."""

    def test_zero_byte_file_raises_value_error(self):
        """Mirrors DataLoader.load_csv's behavior of surfacing parse failures
        as ValueError (load_csv wraps pandas errors; load_excel wraps them
        identically via its own try/except -> `raise ValueError(...)`)."""
        with pytest.raises(ValueError, match="Failed to load Excel"):
            DataLoader.load_excel(b"")

    def test_garbage_bytes_raises_value_error(self):
        with pytest.raises(ValueError, match="Failed to load Excel"):
            DataLoader.load_excel(b"this is not a real xlsx file")

    def test_valid_workbook_with_headers_but_no_rows_returns_empty_dataframe(self):
        """A structurally valid workbook with only a header row (no data rows)
        should behave like DataLoader.load_csv's 'headers only' case: a valid
        DataFrame with the right columns and zero rows, not an exception."""
        data = _workbook_bytes(rows=[], headers=["col1", "col2", "col3"])

        df = DataLoader.load_excel(data)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["col1", "col2", "col3"]
        assert len(df) == 0

    def test_completely_empty_worksheet_does_not_crash(self):
        """A workbook with no header and no rows at all should still return
        a DataFrame (possibly empty/default-columned) rather than raising an
        unhandled exception - consistent with load_csv's empty-content case
        returning an empty DataFrame instead of propagating an error."""
        data = _workbook_bytes(rows=[], headers=None)

        df = DataLoader.load_excel(data)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
