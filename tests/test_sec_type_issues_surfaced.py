"""A column that fails its type check must appear in the top-level `issues` list
and in summary.issues_found (both analyze routes and the page); every kind of
finding the checker can produce reaches `issues`."""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY = "sec-test-key-type-issues"
H = {"X-API-Key": KEY}
APP = str(Path(__file__).resolve().parent.parent / "web_app.py")

REPRO = ("record_id,age,sex,sbp,visit_date\n1,34,F,120,2026-01-05\n2,150,M,118,2026-01-06\n"
         "3,-5,X,300,2026-01-07\n4,,F,95,not-a-date\n5,61,M,,2026-01-09\n")


@pytest.fixture
def client(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "API_KEY", KEY)
    api_server.limiter.reset()
    yield TestClient(api_server.app)
    api_server.limiter.reset()


def _multipart(client, csv, schema=None, rules=None, min_rows=1):
    data = {"min_rows": str(min_rows)}
    if schema is not None:
        data["schema"] = json.dumps(schema)
    if rules is not None:
        data["rules"] = json.dumps(rules)
    return client.post("/api/v1/analyze", files={"data_file": ("d.csv", csv.encode(), "text/csv")},
                       data=data, headers=H)


def _json(client, csv, schema=None, rules=None, min_rows=1):
    body = {"data_content": csv, "min_rows": min_rows}
    if schema is not None:
        body["schema"] = schema
    if rules is not None:
        body["rules"] = rules
    return client.post("/api/v1/analyze/json", json=body, headers=H)


ROUTES = [_multipart, _json]


def _of_type(body, t, column=None):
    return [i for i in body["issues"] if i["type"] == t and (column is None or i["column"] == column)]


def _consistent(body):
    assert body["summary"]["issues_found"] == body["issues_total"]
    assert body["issues_returned"] == len(body["issues"])
    assert body["issues_truncated"] == (body["issues_returned"] < body["issues_total"])


@pytest.mark.parametrize("route", ROUTES)
def test_a_bad_date_is_a_top_level_issue_and_counted_in_the_summary(client, route):
    resp = route(client, REPRO, schema={"visit_date": "date", "age": "int"})
    assert resp.status_code == 200
    body = resp.json()
    bad = _of_type(body, "invalid_date", "visit_date")
    assert len(bad) == 1 and bad[0]["row"] == 3 and bad[0]["value"] == "not-a-date"
    assert bad[0]["severity"] == "error"
    # 2 missing-value entries (age, sbp) + 1 invalid date
    assert body["summary"]["issues_found"] == 3 and body["summary"]["critical_issues"] == 1
    _consistent(body)


@pytest.mark.parametrize("route", ROUTES)
def test_an_int_column_containing_text_is_reported_per_row(client, route):
    body = route(client, "n\n1\nabc\n3\nxyz\n", schema={"n": "int"}).json()
    bad = _of_type(body, "type_mismatch", "n")
    assert sorted(i["row"] for i in bad) == [1, 3]
    assert body["summary"]["critical_issues"] == 2
    _consistent(body)


@pytest.mark.parametrize("route", ROUTES)
def test_a_bool_column_containing_other_text_is_reported(client, route):
    body = route(client, "flag\ntrue\nmaybe\nfalse\n", schema={"flag": "bool"}).json()
    bad = _of_type(body, "type_mismatch", "flag")
    assert [i["row"] for i in bad] == [1] and bad[0]["value"] == "maybe"


@pytest.mark.parametrize("route", ROUTES)
def test_a_clean_typed_file_has_no_type_issue(client, route):
    body = route(client, "n,d,f,s\n1,2026-01-01,yes,a\n2,2026-01-02,no,b\n",
                 schema={"n": "int", "d": "date", "f": "bool", "s": "str"}).json()
    assert body["issues"] == [] and body["summary"]["issues_found"] == 0


@pytest.mark.parametrize("route", ROUTES)
def test_a_schema_column_missing_from_the_data_is_an_issue(client, route):
    body = route(client, "a\n1\n", schema={"nope": "int"}).json()
    assert _of_type(body, "missing_column", "nope")
    assert body["summary"]["issues_found"] == 1


@pytest.mark.parametrize("route", ROUTES)
def test_too_few_rows_is_an_issue(client, route):
    body = route(client, "a\n1\n", min_rows=5).json()
    assert _of_type(body, "row_count_below_minimum")
    assert body["summary"]["issues_found"] == 1


@pytest.mark.parametrize("route", ROUTES)
def test_range_allowed_and_invalid_rule_findings_still_reach_issues_and_are_counted(client, route):
    body = route(client, "age,c\n150,x\n5,y\n",
                 rules={"age": {"max": 100}, "c": {"allowed": ["y"]}}).json()
    kinds = {i["type"] for i in body["issues"]}
    assert {"range_violation", "invalid_categorical_value"} <= kinds
    assert body["summary"]["issues_found"] == 2
    _consistent(body)


def test_a_rule_the_checker_cannot_apply_is_counted_in_the_summary():
    import api_server
    from mcp_server import QualityPipeline

    df = pd.DataFrame({"age": [1, 2]})
    report = api_server._build_quality_report(df, QualityPipeline(df, rules={"age": {"min": "zz"}}).run_all_checks())
    assert [i["type"] for i in report["issues"]] == ["rule_invalid"]
    assert report["summary"]["issues_found"] == report["issues_total"] == 1


def test_type_violations_beyond_the_listed_rows_are_counted_not_dropped(client):
    csv = "n\n" + "\n".join(["x"] * 30) + "\n"
    body = _multipart(client, csv, schema={"n": "int"}).json()
    assert body["issues_total"] == 30 and body["summary"]["issues_found"] == 30
    _consistent(body)


# ---- the page ---------------------------------------------------------------
class _Resp:
    def __init__(self, r):
        self.status_code, self.text = r.status_code, r.text
        self.json = r.json


class FakeParser:
    deployment = "fake-deployment"

    def count_tokens(self, text):
        return 1

    def parse_dictionary(self, text, max_fields=500, deployment=None):
        fields = [{"field_name": "visit_date", "data_type": "date", "description": "d"},
                  {"field_name": "age", "data_type": "int", "description": "d", "min_value": 0, "max_value": 120}]
        return {"schema": {"visit_date": {"type": "date"}, "age": {"type": "int", "min": 0, "max": 120}},
                "fields": fields, "metadata": {"chunks_processed": 1, "processing_time_seconds": 0.1, "mode": "fake"}}


def test_the_page_lists_the_invalid_date_under_its_own_heading(client, monkeypatch, tmp_path):
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
    at.file_uploader(key="data_uploader").upload("d.csv", REPRO.encode(), "text/csv").run()
    at.button(key="run_analysis_main").click().run()
    assert not at.exception
    headings = [e.label.replace("\\", "") for e in at.expander]
    assert "Invalid Date (1 issues)" in headings, headings
    assert any("not\\-a\\-date" in e.value for e in at.error)
