"""Security: client-facing errors are stable and carry a correlation id; logs
hold no uploaded file names or data values; DEBUG cannot expose details in a
deployed environment (ADP-78: A-08, A-09, A-10)."""

import os
import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY = "sec-test-key-errors"
HEADERS = {"X-API-Key": KEY}
SENSITIVE_NAME = "subject-9876-jane.csv"
SENSITIVE_VALUE = "PHI-VALUE-4242"


@pytest.fixture
def api(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "API_KEY", KEY)
    return api_server


@pytest.fixture
def client(api):
    return TestClient(api.app, raise_server_exceptions=False)


def test_unparseable_excel_upload_returns_a_fixed_message_without_library_text(client):
    resp = client.post(
        "/api/v1/analyze",
        files={"data_file": ("x.xlsx", b"not really an excel file", "application/octet-stream")},
        headers=HEADERS,
    )

    assert resp.status_code == 400
    body = resp.json()
    assert body["error"] == "Failed to parse the uploaded .xlsx file. Check that it is a valid file of that type."
    assert "engine" not in resp.text and "openpyxl" not in resp.text.lower()


def test_unexpected_analysis_failure_returns_a_fixed_message_plus_a_request_id(api, client, monkeypatch):
    def boom(self, min_rows=1):
        raise RuntimeError(f"internal detail {SENSITIVE_VALUE}")

    monkeypatch.setattr(api.mcp_server.QualityPipeline, "run_all_checks", boom)

    resp = client.post("/api/v1/analyze", files={"data_file": ("d.csv", b"a\n1\n", "text/csv")}, headers=HEADERS)

    assert resp.status_code == 500
    assert SENSITIVE_VALUE not in resp.text
    body = resp.json()
    assert body["error"] == "Unexpected error while analysing the data."
    assert body["request_id"] and body["request_id"] == resp.headers["X-Request-ID"]


def test_request_id_is_generated_by_the_server_not_echoed_from_the_client(client):
    resp = client.post(
        "/api/v1/analyze",
        files={"data_file": ("d.csv", b"a\n1\n", "text/csv")},
        headers={"X-API-Key": "wrong", "X-Request-ID": "forged\nINFO fake log line"},
    )

    assert resp.status_code == 403
    assert resp.json()["request_id"] != "forged\nINFO fake log line"
    assert resp.headers["X-Request-ID"] == resp.json()["request_id"]


def test_logs_for_a_failed_parse_contain_no_file_name_or_cell_values(api, client, caplog):
    csv = f"a,b\n1,2\n{SENSITIVE_VALUE},x,extra,columns,here\n"
    with caplog.at_level("DEBUG"):
        client.post(
            "/api/v1/analyze",
            files={"data_file": (SENSITIVE_NAME, csv.encode(), "text/csv")},
            headers=HEADERS,
        )
        client.post(
            "/api/v1/analyze",
            files={"data_file": (SENSITIVE_NAME.replace(".csv", ".xlsx"), SENSITIVE_VALUE.encode(), "text/csv")},
            headers=HEADERS,
        )

    logged = "\n".join(r.getMessage() + (str(r.exc_text or "")) for r in caplog.records)
    assert SENSITIVE_NAME not in logged
    assert "subject-9876" not in logged
    assert SENSITIVE_VALUE not in logged
    assert "analysis_parse_failed" in logged  # the useful, safe detail is still there


def test_logs_for_a_successful_analysis_contain_no_file_name(client, caplog):
    with caplog.at_level("DEBUG"):
        resp = client.post(
            "/api/v1/analyze",
            files={"data_file": (SENSITIVE_NAME, b"a\n1\n", "text/csv")},
            headers=HEADERS,
        )

    assert resp.status_code == 200
    assert "subject-9876" not in "\n".join(r.getMessage() for r in caplog.records)


def test_dictionary_parse_logs_and_errors_omit_the_uploaded_file_name(api, client, monkeypatch, caplog):
    manager = MagicMock()
    manager.create_program_from_dictionary.side_effect = RuntimeError(f"LLM said {SENSITIVE_VALUE}")
    monkeypatch.setattr(api, "program_manager", manager)

    with caplog.at_level("DEBUG"):
        resp = client.post(
            "/api/v1/dictionary/parse",
            files={"dictionary_file": (SENSITIVE_NAME, b"a,b\n1,2\n", "text/csv")},
            headers=HEADERS,
        )

    assert resp.status_code == 500
    assert resp.json()["error"] == "Dictionary parsing failed."
    assert SENSITIVE_VALUE not in resp.text
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "subject-9876" not in logged
    assert SENSITIVE_VALUE not in logged


def test_dictionary_lookup_log_strips_newlines_from_the_caller_supplied_id(api, client, monkeypatch, caplog):
    manager = MagicMock()
    manager.db.load_program.return_value = None
    monkeypatch.setattr(api, "program_manager", manager)

    with caplog.at_level("DEBUG"):
        resp = client.get("/api/v1/dictionary/abc%0aINFO%20forged%20entry", headers=HEADERS)

    assert resp.status_code == 404
    logged = "\n".join(r.getMessage() for r in caplog.records if r.name == "api_server")
    assert "abc?INFO forged entry" in logged  # newline neutralised, text kept on one line
    assert "abc\nINFO" not in logged


def test_lookup_database_failure_returns_a_fixed_message(api, client, monkeypatch):
    manager = MagicMock()
    manager.db.load_program.side_effect = RuntimeError(f"disk path /secret/{SENSITIVE_VALUE}")
    monkeypatch.setattr(api, "program_manager", manager)

    resp = client.get("/api/v1/dictionary/abc", headers=HEADERS)

    assert resp.status_code == 500
    assert resp.json()["error"] == "Database error."
    assert SENSITIVE_VALUE not in resp.text


def test_bad_schema_form_field_gets_a_fixed_message(client):
    resp = client.post(
        "/api/v1/analyze",
        files={"data_file": ("d.csv", b"a\n1\n", "text/csv")},
        data={"schema": "{not json " + SENSITIVE_VALUE},
        headers=HEADERS,
    )

    assert resp.status_code == 400
    assert SENSITIVE_VALUE not in resp.text


def test_invalid_base64_gets_a_fixed_message(client):
    resp = client.post(
        "/api/v1/analyze/json",
        json={"data_content": "***not-base64***", "format": "csv", "encoding": "base64"},
        headers=HEADERS,
    )

    assert resp.status_code == 400
    assert resp.json()["error"] == "Invalid base64 in 'data_content'."


def test_stray_value_error_does_not_leak_its_text_to_the_client(api, client, monkeypatch):
    @api.app.get("/__sec_probe_value_error")
    async def probe():
        raise ValueError(f"contains {SENSITIVE_VALUE}")

    resp = client.get("/__sec_probe_value_error")

    assert resp.status_code == 400
    assert SENSITIVE_VALUE not in resp.text
    assert resp.json()["error"] == "Invalid input"


def test_debug_true_does_not_reveal_exception_text_in_a_deployed_environment(api, client, monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("APP_ENV", "prod")

    @api.app.get("/__sec_probe_unhandled_deployed")
    async def probe():
        raise KeyError(SENSITIVE_VALUE)

    resp = client.get("/__sec_probe_unhandled_deployed")

    assert resp.status_code == 500
    assert SENSITIVE_VALUE not in resp.text
    assert resp.json()["detail"] == "An unexpected error occurred"


def test_debug_true_still_gives_detail_on_a_local_developer_machine(api, client, monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    for name in ("APP_ENV", "CONTAINER_APP_NAME", "CONTAINER_APP_REVISION",
                 "CONTAINER_APP_HOSTNAME", "CONTAINER_APP_ENV_DNS_SUFFIX"):
        monkeypatch.delenv(name, raising=False)

    @api.app.get("/__sec_probe_unhandled_local")
    async def probe():
        raise KeyError("local-detail")

    resp = client.get("/__sec_probe_unhandled_local")

    assert resp.status_code == 500
    assert "local-detail" in resp.json()["detail"]


def test_the_413_response_also_carries_a_server_generated_request_id(api, client, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_MAX_UPLOAD_BYTES", "100")

    resp = client.post(
        "/api/v1/analyze",
        files={"data_file": ("d.csv", b"a\n" * 500, "text/csv")},
        headers={**HEADERS, "X-Request-ID": "client-chosen"},
    )

    assert resp.status_code == 413
    assert resp.json()["request_id"] != "client-chosen"
