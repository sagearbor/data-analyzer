"""Dictionary type names (date, integer, categorical, ...) are mapped to the
checker's five types by ONE shared function; unknown names are a 422 on the API
and a visible "type not checked" note on the page - never a failed analysis."""
import os
import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.schema_types import normalize_schema, normalize_schema_type

KEY = "sec-test-key-schema-types"
APP = str(Path(__file__).resolve().parent.parent / "web_app.py")
H = {"X-API-Key": KEY}

ALIASES = {
    "int": ["int", "INTEGER", " Int64 ", "bigint"],
    "float": ["float", "number", "Numeric", "decimal", "double", "float64"],
    "str": ["str", "string", "TEXT", "varchar", "categorical", "category", "dropdown", "radio"],
    "bool": ["bool", "boolean"],
    "datetime": ["datetime", "date", "Timestamp", "time", "datetime64", "date_ymd", "datetime_dmy"],
}


@pytest.fixture
def api(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "API_KEY", KEY)
    api_server.limiter.reset()
    yield api_server
    api_server.limiter.reset()


@pytest.fixture
def client(api):
    return TestClient(api.app)


@pytest.mark.parametrize("canon,names", ALIASES.items())
def test_every_documented_alias_maps_to_its_canonical_type(canon, names):
    assert [normalize_schema_type(n) for n in names] == [canon] * len(names)


@pytest.mark.parametrize("bad", ["array", "calc", "", None, 5, ["int"]])
def test_an_unknown_type_name_does_not_map(bad):
    assert normalize_schema_type(bad) is None


def test_normalize_schema_leaves_unmappable_columns_out_and_lists_them():
    assert normalize_schema({"a": "date", "b": "array", "c": "Integer"}) == ({"a": "datetime", "c": "int"}, ["b"])


def _multipart(client, csv, schema):
    import json

    return client.post("/api/v1/analyze", files={"data_file": ("d.csv", csv.encode(), "text/csv")},
                       data={"schema": json.dumps(schema)}, headers=H)


def _json(client, csv, schema):
    return client.post("/api/v1/analyze/json", json={"data_content": csv, "schema": schema}, headers=H)


@pytest.mark.parametrize("route", [_multipart, _json])
@pytest.mark.parametrize("alias", ["date", "Timestamp", "datetime64"])
def test_a_date_alias_is_applied_as_a_datetime_check_not_ignored(client, route, alias):
    resp = route(client, "visit\n2024-01-05\nnot-a-date\n", {"visit": alias})
    assert resp.status_code == 200
    assert any(i["column"] == "visit" for i in resp.json()["quality_checks"]["data_types"]["issues"])


@pytest.mark.parametrize("route", [_multipart, _json])
@pytest.mark.parametrize("alias", ["integer", "INT64", "number", "string", "categorical", "boolean"])
def test_every_alias_family_is_accepted_on_both_routes(client, route, alias):
    assert route(client, "x\n1\n2\n", {"x": alias}).status_code == 200


@pytest.mark.parametrize("route", [_multipart, _json])
def test_an_unknown_type_is_still_a_422_naming_the_column_but_not_the_value(client, route):
    resp = route(client, "x\n1\n", {"x": "SECRET-TYPE-NAME"})
    assert resp.status_code == 422
    assert "'x'" in resp.json()["detail"] and "SECRET-TYPE-NAME" not in resp.text


# ---- the page --------------------------------------------------------------
class _Resp:
    def __init__(self, r):
        self.status_code, self._r, self.text = r.status_code, r, r.text

    def json(self):
        return self._r.json()


@pytest.fixture
def routed(monkeypatch, api, tmp_path):
    """Send the page's analyze POST to the in-process API (no network)."""
    tc = TestClient(api.app)
    monkeypatch.setenv("DICT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("DATA_ANALYZER_API_KEY", KEY)
    monkeypatch.delenv("CONTAINER_APP_NAME", raising=False)

    def fake_post(url, files=None, data=None, headers=None, **kw):
        return _Resp(tc.post("/api/v1/analyze", files=files, data=data, headers=headers))

    monkeypatch.setattr("requests.post", fake_post)


FIELDS = [
    {"field_name": "visit_date", "data_type": "date", "description": "d"},
    {"field_name": "count", "data_type": "integer", "description": "d"},
    {"field_name": "arm", "data_type": "categorical", "description": "d"},
    {"field_name": "tags", "data_type": "array", "description": "d"},
]


class FakeParser:
    deployment = "fake-deployment"

    def count_tokens(self, text):
        return 1

    def parse_dictionary(self, text, max_fields=500, deployment=None):
        return {"schema": {f["field_name"]: {"type": f["data_type"]} for f in FIELDS}, "fields": FIELDS,
                "metadata": {"chunks_processed": 1, "processing_time_seconds": 0.1, "mode": "fake"}}


def test_the_page_maps_types_and_lists_the_unmappable_one_as_not_checked(routed, monkeypatch):
    monkeypatch.setattr("src.llm_client.LLMDictionaryParser", FakeParser)
    data = pd.DataFrame({"visit_date": ["2024-01-05", "2024-02-01"], "count": [1, 2],
                         "arm": ["a", "b"], "tags": ["x", "y"]})
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    at.file_uploader(key="dict_uploader").upload("d.txt", b"anything", "application/octet-stream").run()
    at.file_uploader(key="data_uploader").upload("d.csv", data.to_csv(index=False).encode(), "text/csv").run()
    at.button(key="run_analysis_main").click().run()
    assert not at.exception
    failed = [e.value for e in at.error if "Analysis failed" in e.value]
    assert not failed, failed
    results = at.session_state["analysis_results"]
    assert results["summary"]["column_types_not_checked"] == ["tags"]
    assert any("1 column type(s) could not be checked" in w.value.replace("\\", "") and "tags" in w.value
               for w in at.warning)


DEMO_PAIRS = [
    ("CSV - Western", "CSV - Western"),
    ("CSV - Asian", "CSV - Asian"),
    ("JSON - Mixed", "JSON - Mixed"),
    ("CSV - Clinical Trial", "CSV - Clinical"),
    ("REDCap - Clinical (synthetic, with branching logic)", "REDCap - Clinical (synthetic)"),
]


@pytest.mark.parametrize("dict_name,data_name", DEMO_PAIRS)
def test_every_builtin_demo_dictionary_with_its_demo_data_analyses_without_a_422(routed, dict_name, data_name):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.selectbox(key="demo_selector").select(data_name).run()
    at.selectbox(key="demo_dict_selector").select(dict_name).run()
    at.button(key="run_analysis_main").click().run()
    assert not at.exception
    failed = [e.value for e in at.error if "Analysis failed" in e.value]
    assert not failed, failed
    assert at.session_state["analysis_results"]["summary"]["total_rows"] > 0
