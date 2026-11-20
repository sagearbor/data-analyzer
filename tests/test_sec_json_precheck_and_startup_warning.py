"""JSON datasets are bounded BEFORE pandas builds a table, and the API warns at
start-up when callers cannot be told apart (shared client-id mode)."""
import io
import json
import os
import sys

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY = "sec-test-key-json-precheck"
H = {"X-API-Key": KEY}


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


def _multipart(client, text):
    return client.post("/api/v1/analyze", files={"data_file": ("d.json", text.encode(), "application/json")}, headers=H)


def _json_route(client, text):
    return client.post("/api/v1/analyze/json", json={"data_content": text, "format": "json"}, headers=H)


ROUTES = [_multipart, _json_route]


@pytest.fixture
def pandas_must_not_run(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("pandas was called before the JSON was bounded")

    monkeypatch.setattr(pd, "read_json", boom)
    monkeypatch.setattr(pd, "DataFrame", boom)
    monkeypatch.setattr(pd, "json_normalize", boom)


@pytest.mark.parametrize("route", ROUTES)
def test_a_json_list_with_too_many_rows_is_refused_with_413_before_pandas(client, route, monkeypatch, pandas_must_not_run):
    monkeypatch.setenv("DATA_ANALYZER_MAX_ROWS", "5")
    resp = route(client, json.dumps([{"a": i} for i in range(6)]))
    assert resp.status_code == 413 and "rows" in resp.json()["detail"]


@pytest.mark.parametrize("route", ROUTES)
def test_a_json_list_with_too_many_columns_is_refused_with_413_before_pandas(client, route, monkeypatch, pandas_must_not_run):
    monkeypatch.setenv("DATA_ANALYZER_MAX_COLUMNS", "3")
    resp = route(client, json.dumps([{"a": 1, "b": 2}, {"c": 3, "d": 4}]))
    assert resp.status_code == 413 and "columns" in resp.json()["detail"]


@pytest.mark.parametrize("route", ROUTES)
def test_a_json_object_of_columns_over_the_limits_is_refused_before_pandas(client, route, monkeypatch, pandas_must_not_run):
    monkeypatch.setenv("DATA_ANALYZER_MAX_ROWS", "3")
    assert route(client, json.dumps({"a": [1, 2, 3, 4]})).status_code == 413
    monkeypatch.setenv("DATA_ANALYZER_MAX_COLUMNS", "2")
    assert route(client, json.dumps({"a": [1], "b": [1], "c": [1]})).status_code == 413


@pytest.mark.parametrize("route", ROUTES)
def test_oversized_json_bytes_are_refused_with_413_before_pandas(client, route, monkeypatch, pandas_must_not_run):
    monkeypatch.setenv("DATA_ANALYZER_MAX_JSON_BYTES", "50")
    resp = route(client, json.dumps([{"a": "x" * 100}]))
    assert resp.status_code == 413 and "DATA_ANALYZER_MAX_JSON_BYTES" in resp.json()["detail"]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("depth", [25, 5000])
def test_deeply_nested_json_is_refused_cleanly_not_a_500(client, route, depth, pandas_must_not_run):
    resp = route(client, "[" * depth + "]" * depth)
    assert resp.status_code == 400 and "nested" in resp.json()["detail"]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("text", ["5", '"x"', "null"])
def test_json_that_is_not_a_list_or_object_is_a_400_with_a_stable_message(client, route, text, pandas_must_not_run):
    resp = route(client, text)
    assert resp.status_code == 400 and "list of records or an object of columns" in resp.json()["detail"]


@pytest.mark.parametrize("route", ROUTES)
def test_valid_list_of_records_and_dict_of_columns_still_analyse(client, route):
    for text in (json.dumps([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]), json.dumps({"a": [1, 2], "b": ["x", "y"]})):
        resp = route(client, text)
        assert resp.status_code == 200 and resp.json()["summary"]["total_rows"] == 2


def test_the_json_byte_limit_is_never_above_the_overall_limit(monkeypatch):
    from mcp_server import DataLoader

    monkeypatch.setenv("DATA_ANALYZER_MAX_JSON_BYTES", "1000000")
    assert DataLoader.max_json_bytes(overall_limit=500) == 500
    monkeypatch.delenv("DATA_ANALYZER_MAX_JSON_BYTES")
    assert DataLoader.max_json_bytes(overall_limit=10**9) == 26214400


def test_the_library_loader_uses_the_same_bounds(monkeypatch):
    from mcp_server import DataLoader, DataTooLargeError

    monkeypatch.setenv("DATA_ANALYZER_MAX_ROWS", "2")
    with pytest.raises(DataTooLargeError):
        DataLoader.load_json(json.dumps([{"a": 1}, {"a": 2}, {"a": 3}]))
    with pytest.raises(ValueError, match="nested"):
        DataLoader.load_json("[" * 5000 + "]" * 5000)
    assert len(DataLoader.load_json(json.dumps([{"a": 1}, {"a": 2}]))) == 2


# ---- start-up warning -------------------------------------------------------
def test_deployed_without_trusted_proxy_hops_logs_one_shared_mode_warning(api, monkeypatch, caplog):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.delenv("DATA_ANALYZER_TRUSTED_PROXY_HOPS", raising=False)
    with caplog.at_level("WARNING"):
        assert api.warn_if_client_id_shared() is True
    lines = [r.getMessage() for r in caplog.records if "client_id_shared_mode" in r.getMessage()]
    assert len(lines) == 1


@pytest.mark.parametrize("hops,env,expected", [("1", "prod", False), ("0", "prod", True), ("", "dev", False)])
def test_the_shared_mode_warning_depends_on_deployment_and_the_hops_setting(api, monkeypatch, caplog, hops, env, expected):
    monkeypatch.setenv("APP_ENV", env)
    monkeypatch.delenv("CONTAINER_APP_NAME", raising=False)
    monkeypatch.setenv("DATA_ANALYZER_TRUSTED_PROXY_HOPS", hops)
    with caplog.at_level("WARNING"):
        assert api.warn_if_client_id_shared() is expected


def test_the_startup_event_calls_the_check_and_does_not_refuse_to_start(api, monkeypatch, caplog):
    import asyncio

    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.delenv("DATA_ANALYZER_TRUSTED_PROXY_HOPS", raising=False)
    with caplog.at_level("WARNING"):
        asyncio.run(api.startup_event())
    assert any("client_id_shared_mode" in r.getMessage() for r in caplog.records)
