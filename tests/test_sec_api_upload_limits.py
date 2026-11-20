"""Security: every upload route is size-bounded, and datasets have row/column/
Excel-decompression limits and a capped issue list (ADP-73: A-02, A-03)."""

import io
import os
import sys
import zipfile
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY = "sec-test-key-limits"


@pytest.fixture
def api(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "API_KEY", KEY)
    return api_server


@pytest.fixture
def client(api):
    return TestClient(api.app)


HEADERS = {"X-API-Key": KEY}


def _xlsx_bytes(rows: int, cols: int = 2, sheets: int = 1) -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    for i in range(sheets):
        ws = wb.active if i == 0 else wb.create_sheet()
        ws.append([f"c{j}" for j in range(cols)])
        for r in range(rows):
            ws.append(["same-text-repeated" for _ in range(cols)])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# --- 1a: dictionary route is covered by the streamed size limit -------------

def test_dictionary_upload_over_its_own_limit_is_refused_before_any_llm_work(api, client, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_MAX_DICTIONARY_UPLOAD_BYTES", "1000")
    manager = MagicMock()
    monkeypatch.setattr(api, "program_manager", manager)

    resp = client.post(
        "/api/v1/dictionary/parse",
        files={"dictionary_file": ("d.csv", b"a,b\n" * 1000, "text/csv")},
        headers=HEADERS,
    )

    assert resp.status_code == 413
    assert "DATA_ANALYZER_MAX_DICTIONARY_UPLOAD_BYTES" in resp.json()["error"]
    manager.create_program_from_dictionary.assert_not_called()


def test_dictionary_upload_without_content_length_is_still_stopped_while_streaming(api, client, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_MAX_DICTIONARY_UPLOAD_BYTES", "1000")
    manager = MagicMock()
    monkeypatch.setattr(api, "program_manager", manager)

    def chunks():
        for _ in range(50):
            yield b"x" * 100

    resp = client.post(
        "/api/v1/dictionary/parse",
        content=chunks(),  # chunked transfer: no Content-Length header
        headers={**HEADERS, "Content-Type": "multipart/form-data; boundary=zz"},
    )

    assert resp.status_code == 413
    manager.create_program_from_dictionary.assert_not_called()


def test_dictionary_default_limit_is_much_smaller_than_the_analyze_limit(api, monkeypatch):
    monkeypatch.delenv("DATA_ANALYZER_MAX_DICTIONARY_UPLOAD_BYTES", raising=False)
    monkeypatch.delenv("DATA_ANALYZER_MAX_UPLOAD_BYTES", raising=False)

    assert api._max_dictionary_request_bytes() == 2 * 1024 * 1024
    assert api._max_dictionary_request_bytes() < api._max_request_bytes()


def test_small_dictionary_upload_is_not_blocked_by_the_limit(api, client, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_MAX_DICTIONARY_UPLOAD_BYTES", "100000")
    manager = MagicMock()
    program = MagicMock(program_id="pid", num_fields=1, num_basic_rules=0, num_logic_rules=0,
                        generated_code="", schema={}, dictionary_format="csv", model_used="m")
    program.name = "n"
    manager.create_program_from_dictionary.return_value = program
    monkeypatch.setattr(api, "program_manager", manager)

    resp = client.post(
        "/api/v1/dictionary/parse",
        files={"dictionary_file": ("d.csv", b"a,b\n1,2\n", "text/csv")},
        headers=HEADERS,
    )

    assert resp.status_code == 200


# --- 1c: dataset shape limits ------------------------------------------------

def test_csv_with_more_rows_than_the_limit_is_refused_with_413(api, client, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_MAX_ROWS", "50")
    csv = "a,b\n" + "1,2\n" * 100

    resp = client.post("/api/v1/analyze", files={"data_file": ("d.csv", csv, "text/csv")}, headers=HEADERS)

    assert resp.status_code == 413
    assert "DATA_ANALYZER_MAX_ROWS" in resp.json()["error"]


def test_csv_with_more_columns_than_the_limit_is_refused_with_413(api, client, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_MAX_COLUMNS", "5")
    csv = ",".join(f"c{i}" for i in range(20)) + "\n" + ",".join("1" for _ in range(20)) + "\n"

    resp = client.post("/api/v1/analyze", files={"data_file": ("d.csv", csv, "text/csv")}, headers=HEADERS)

    assert resp.status_code == 413
    assert "DATA_ANALYZER_MAX_COLUMNS" in resp.json()["error"]


def test_csv_within_limits_is_still_analysed(api, client, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_MAX_ROWS", "50")

    resp = client.post("/api/v1/analyze", files={"data_file": ("d.csv", "a,b\n1,2\n3,4\n", "text/csv")}, headers=HEADERS)

    assert resp.status_code == 200
    assert resp.json()["summary"]["total_rows"] == 2


def test_xlsx_that_expands_beyond_the_uncompressed_limit_is_refused_before_loading(api, client, monkeypatch):
    import pandas as pd

    data = _xlsx_bytes(rows=3000)
    monkeypatch.setenv("DATA_ANALYZER_MAX_EXCEL_UNCOMPRESSED_BYTES", "20000")

    def must_not_load(*a, **k):
        raise AssertionError("workbook must be refused before pandas/openpyxl loads it")

    monkeypatch.setattr(pd, "ExcelFile", must_not_load)

    resp = client.post("/api/v1/analyze", files={"data_file": ("d.xlsx", data, "application/octet-stream")}, headers=HEADERS)

    assert resp.status_code == 413
    assert "DATA_ANALYZER_MAX_EXCEL_UNCOMPRESSED_BYTES" in resp.json()["error"]


def test_xlsx_sheet_dimensions_over_the_row_limit_are_refused_before_loading(api, client, monkeypatch):
    import pandas as pd

    data = _xlsx_bytes(rows=300)
    monkeypatch.setenv("DATA_ANALYZER_MAX_ROWS", "100")
    monkeypatch.setattr(pd, "ExcelFile", lambda *a, **k: (_ for _ in ()).throw(AssertionError("loaded")))

    resp = client.post("/api/v1/analyze", files={"data_file": ("d.xlsx", data, "application/octet-stream")}, headers=HEADERS)

    assert resp.status_code == 413
    assert "DATA_ANALYZER_MAX_ROWS" in resp.json()["error"]


def test_xlsx_with_too_many_sheets_is_refused(api, client, monkeypatch):
    data = _xlsx_bytes(rows=2, sheets=6)
    monkeypatch.setenv("DATA_ANALYZER_MAX_EXCEL_SHEETS", "3")

    resp = client.post("/api/v1/analyze", files={"data_file": ("d.xlsx", data, "application/octet-stream")}, headers=HEADERS)

    assert resp.status_code == 413
    assert "DATA_ANALYZER_MAX_EXCEL_SHEETS" in resp.json()["error"]


def test_xlsx_within_limits_is_still_analysed(api, client):
    data = _xlsx_bytes(rows=5)

    resp = client.post("/api/v1/analyze", files={"data_file": ("d.xlsx", data, "application/octet-stream")}, headers=HEADERS)

    assert resp.status_code == 200
    assert resp.json()["summary"]["total_rows"] == 5


def test_zip_with_too_many_internal_parts_is_refused_as_an_excel_file(api, monkeypatch):
    import mcp_server

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for i in range(30):
            zf.writestr(f"p{i}.xml", "x")
    monkeypatch.setenv("DATA_ANALYZER_MAX_EXCEL_ARCHIVE_MEMBERS", "10")

    with pytest.raises(mcp_server.DataTooLargeError):
        mcp_server.DataLoader.load_excel(buf.getvalue())


def test_json_inline_route_enforces_the_row_limit_too(api, client, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_MAX_ROWS", "5")

    resp = client.post(
        "/api/v1/analyze/json",
        json={"data_content": "a\n" + "1\n" * 50, "format": "csv"},
        headers=HEADERS,
    )

    assert resp.status_code == 413


# --- 1d: capped issue list with an explicit truncation indicator --------------

def test_issue_list_is_cut_at_the_limit_and_says_so_with_the_true_total(api, monkeypatch):
    import pandas as pd

    monkeypatch.setenv("DATA_ANALYZER_MAX_ISSUES_RETURNED", "10")
    df = pd.DataFrame({"age": list(range(100))})
    results = {
        "issues": [{"column": "age", "rule": "min >= 1000", "violating_rows": list(range(100))}],
        "checks": {}, "summary_stats": {},
    }

    report = api._build_quality_report(df, results)

    assert len(report["issues"]) == 10
    assert report["issues_returned"] == 10
    assert report["issues_total"] == 100
    assert report["issues_truncated"] is True
    assert report["summary"]["issues_found"] == 100  # true count, not the cut list
    assert report["summary"]["critical_issues"] == 100


def test_issue_list_is_not_marked_truncated_when_under_the_limit(api, monkeypatch):
    import pandas as pd

    monkeypatch.setenv("DATA_ANALYZER_MAX_ISSUES_RETURNED", "10")
    df = pd.DataFrame({"age": [1, 2, 3]})
    results = {
        "issues": [{"column": "age", "rule": "min >= 1000", "violating_rows": [0, 1, 2]}],
        "checks": {}, "summary_stats": {},
    }

    report = api._build_quality_report(df, results)

    assert report["issues_truncated"] is False
    assert report["issues_total"] == report["issues_returned"] == 3


def test_analyze_response_carries_the_truncation_fields(api, client):
    resp = client.post("/api/v1/analyze", files={"data_file": ("d.csv", "a\n1\n2\n", "text/csv")}, headers=HEADERS)

    body = resp.json()
    assert resp.status_code == 200
    assert body["issues_truncated"] is False
    assert body["issues_total"] == len(body["issues"])
