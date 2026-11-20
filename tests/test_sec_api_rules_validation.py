"""ADP-82 / A-13: malformed `rules` / `schema` are refused with 422 (never silently
ignored or a 500), on both analyze routes; well-formed rules behave as before;
the checker reports a rule it cannot apply instead of dropping it."""
import json
import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY = "sec-test-key-rules"
CSV = "age,colour\n10,red\n200,blue\n30,green\n"

BAD_RULES = {
    "min_not_a_number": ({"age": {"min": "zz"}}, "age", "min"),
    "min_null": ({"age": {"min": None}}, "age", "min"),
    "max_boolean": ({"age": {"max": True}}, "age", "max"),
    "allowed_not_a_list": ({"colour": {"allowed": 5}}, "colour", "allowed"),
    "allowed_item_is_an_object": ({"colour": {"allowed": [{"x": 1}]}}, "colour", "allowed"),
    "rule_is_a_string": ({"colour": "x"}, "colour", None),
    "unknown_key": ({"age": {"minimum": 3}}, "age", "minimum"),
}


@pytest.fixture
def client(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "API_KEY", KEY)
    api_server.limiter.reset()
    yield TestClient(api_server.app)
    api_server.limiter.reset()


H = {"X-API-Key": KEY}


def _multipart(client, rules=None, schema=None):
    data = {}
    if rules is not None:
        data["rules"] = json.dumps(rules)
    if schema is not None:
        data["schema"] = json.dumps(schema)
    return client.post("/api/v1/analyze", files={"data_file": ("d.csv", CSV.encode(), "text/csv")}, data=data, headers=H)


def _json(client, rules=None, schema=None):
    body = {"data_content": CSV}
    if rules is not None:
        body["rules"] = rules
    if schema is not None:
        body["schema"] = schema
    return client.post("/api/v1/analyze/json", json=body, headers=H)


@pytest.mark.parametrize("name", list(BAD_RULES))
@pytest.mark.parametrize("route", [_multipart, _json])
def test_a_malformed_rule_is_refused_with_422_naming_the_column_and_key(client, route, name):
    rules, column, key = BAD_RULES[name]
    resp = route(client, rules=rules)
    assert resp.status_code == 422
    text = resp.json()["detail"]
    assert f"'{column}'" in text
    if key:
        assert f"'{key}'" in text


def test_the_error_message_never_echoes_the_offending_value(client):
    resp = _multipart(client, rules={"age": {"min": "SECRET-VALUE-123"}})
    assert resp.status_code == 422
    assert "SECRET-VALUE-123" not in resp.text


@pytest.mark.parametrize("schema", [{"age": 5}, {"age": "integer"}, {"age": ["int"]}])
@pytest.mark.parametrize("route", [_multipart, _json])
def test_a_malformed_schema_is_refused_with_422(client, route, schema):
    resp = route(client, schema=schema)
    assert resp.status_code == 422
    assert "'age'" in resp.json()["detail"]


@pytest.mark.parametrize("route", [_multipart, _json])
def test_well_formed_rules_and_schema_still_work_and_report_violations(client, route):
    resp = route(client, rules={"age": {"min": 0, "max": 120}, "colour": {"allowed": ["red", "blue"]}},
                 schema={"age": "int", "colour": "str"})
    assert resp.status_code == 200
    kinds = sorted((i["type"], i["column"]) for i in resp.json()["issues"])
    assert ("range_violation", "age") in kinds
    assert ("invalid_categorical_value", "colour") in kinds


@pytest.mark.parametrize("route", [_multipart, _json])
def test_numeric_and_empty_rule_shapes_that_worked_before_are_accepted(client, route):
    resp = route(client, rules={"age": {"min": 1.5, "max": 99}, "colour": {}})
    assert resp.status_code == 200


def test_the_checker_reports_a_rule_it_cannot_apply_instead_of_dropping_it():
    from mcp_server import QualityChecker

    df = pd.DataFrame({"age": [1, 2], "c": ["a", "b"]})
    out = QualityChecker(df, rules={"age": {"min": "zz"}, "c": {"allowed": 5}}).check_value_ranges()
    assert out["passed"] is False
    found = {(i["column"], i["key"], i["reason"]) for i in out["issues"] if i["issue"] == "rule_invalid"}
    assert found == {("age", "min", "min_not_numeric"), ("c", "allowed", "allowed_not_list")}


def test_the_checker_still_applies_the_valid_part_of_a_partly_invalid_rule():
    from mcp_server import QualityChecker

    df = pd.DataFrame({"age": [1, 200]})
    out = QualityChecker(df, rules={"age": {"max": 100, "min": None}}).check_value_ranges()
    rules = [i["rule"] for i in out["issues"]]
    assert "max <= 100" in rules and "rule_invalid" in rules


def test_the_checker_flags_a_non_object_rule_and_a_mixed_type_allowed_list_does_not_crash():
    from mcp_server import QualityChecker

    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = QualityChecker(df, rules={"a": "x", "b": {"allowed": [1, "x"]}}).check_value_ranges()
    assert any(i.get("reason") == "rule_not_object" and i["column"] == "a" for i in out["issues"])
    assert any("allowed_values" in i["rule"] for i in out["issues"])


def test_the_report_surfaces_a_row_less_rule_invalid_issue():
    import api_server
    from mcp_server import QualityPipeline

    df = pd.DataFrame({"age": [1, 2]})
    results = QualityPipeline(df, rules={"age": {"min": "zz"}}).run_all_checks()
    issues = api_server._build_quality_report(df, results)["issues"]
    bad = [i for i in issues if i["type"] == "rule_invalid"]
    assert len(bad) == 1 and bad[0]["column"] == "age" and "NOT checked" in bad[0]["message"]
