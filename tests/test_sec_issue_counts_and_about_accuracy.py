"""Issue totals are the TRUE totals (the checker lists only the first 10 rows of a
finding), on both routes and on the page; the About page describes only checks the
engine performs and says what it does not check."""
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY = "sec-test-key-issue-counts"
H = {"X-API-Key": KEY}
APP = str(Path(__file__).resolve().parent.parent / "web_app.py")

CSV = "age,c\n" + "".join("500,bad\n" for _ in range(30))
RULES = {"age": {"max": 100}, "c": {"allowed": ["ok"]}}


@pytest.fixture
def client(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "API_KEY", KEY)
    api_server.limiter.reset()
    yield TestClient(api_server.app)
    api_server.limiter.reset()


def _multipart(client):
    return client.post("/api/v1/analyze", files={"data_file": ("d.csv", CSV.encode(), "text/csv")},
                       data={"rules": json.dumps(RULES)}, headers=H)


def _json(client):
    return client.post("/api/v1/analyze/json", json={"data_content": CSV, "rules": RULES}, headers=H)


@pytest.mark.parametrize("route", [_multipart, _json])
def test_range_and_allowed_value_violations_beyond_ten_rows_are_counted_and_flagged(client, route):
    body = route(client).json()
    assert body["summary"]["issues_found"] == body["issues_total"] == 60
    assert body["summary"]["critical_issues"] == 60
    assert body["summary"]["issue_type_totals"] == {"range_violation": 30, "invalid_categorical_value": 30}
    assert body["issues_returned"] == len(body["issues"]) == 20
    assert body["issues_truncated"] is True


def test_a_finding_with_no_more_than_ten_rows_is_not_flagged_as_truncated(client):
    body = client.post("/api/v1/analyze/json", json={"data_content": "age\n500\n5\n", "rules": {"age": {"max": 100}}},
                       headers=H).json()
    assert body["summary"]["issues_found"] == 1 and body["issues_truncated"] is False


class FakeParser:
    deployment = "fake-deployment"

    def count_tokens(self, text):
        return 1

    def parse_dictionary(self, text, max_fields=500, deployment=None):
        return {"schema": {"age": {"type": "int", "max": 100}},
                "fields": [{"field_name": "age", "data_type": "int", "description": "d", "max_value": 100}],
                "metadata": {"chunks_processed": 1, "processing_time_seconds": 0.1, "mode": "fake"}}


class _Resp:
    def __init__(self, r):
        self.status_code, self.text, self.json = r.status_code, r.text, r.json


def test_the_page_shows_the_true_totals_with_a_note_that_only_the_first_rows_are_listed(client, monkeypatch, tmp_path):
    monkeypatch.setenv("DICT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("DATA_ANALYZER_API_KEY", KEY)
    monkeypatch.delenv("CONTAINER_APP_NAME", raising=False)
    monkeypatch.setattr("src.llm_client.LLMDictionaryParser", FakeParser)
    monkeypatch.setattr("requests.post", lambda url, files=None, data=None, headers=None, **k:
                        _Resp(client.post("/api/v1/analyze", files=files, data=data, headers=headers)))
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    at.file_uploader(key="dict_uploader").upload("d.txt", b"anything", "application/octet-stream").run()
    csv = "age\n" + "500\n" * 30
    at.file_uploader(key="data_uploader").upload("d.csv", csv.encode(), "text/csv").run()
    at.button(key="run_analysis_main").click().run()
    assert not at.exception
    assert any(m.label == "Issues Found" and str(m.value) == "30" for m in at.metric)
    assert "Range Violation (30 issues)" in [e.label.replace("\\", "") for e in at.expander]
    assert any("Showing the first 10 of 30 issues" in i.value for i in at.info)


# ---- About page ---------------------------------------------------------------
def _about_text(monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_API_URL", "http://api:8000")
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state["nav_choice"] = "About"
    at.run()
    assert not at.exception
    return " ".join("\n".join(m.value for m in at.markdown).split())


def test_the_about_page_no_longer_claims_checks_the_engine_does_not_perform(monkeypatch):
    text = _about_text(monkeypatch)
    assert "Detects entries like" not in text
    assert "Suspicious Values" not in text.replace("suspicious value", "")  # as a listed check
    assert "Flags test data" not in text


def test_the_about_page_lists_what_the_tool_does_not_check(monkeypatch):
    text = _about_text(monkeypatch)
    assert "What It Does Not Check" in text
    for phrase in ("n/a", "Duplicate rows are only counted", "Required", "Text columns"):
        assert phrase in text


def test_the_about_page_does_not_imply_the_tool_is_validated_for_real_study_data(monkeypatch):
    text = _about_text(monkeypatch)
    assert "has not been validated or approved for use with real study data" in text
