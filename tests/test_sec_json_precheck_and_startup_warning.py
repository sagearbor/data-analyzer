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
    assert resp.status_code == 400 and "list of records" in resp.json()["detail"] and "object of columns" in resp.json()["detail"]


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


# ---- exactly two accepted shapes; counting equals building -------------------------
REFUSED_SHAPES = {
    "list_of_lists": [[1, 2], [3, 4]],
    "list_with_a_non_object": [{"a": 1}, 5],
    "dict_of_dicts": {"a": {"0": 1}, "b": {"0": 2}},
    "split_orientation_with_unequal_lengths": {"columns": ["a", "b"], "data": [[1, 2]]},
    "ragged_columns": {"a": [1, 2, 3], "b": [1]},
    "column_that_is_a_scalar": {"a": [1, 2], "b": 5},
}


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("name", list(REFUSED_SHAPES))
def test_every_shape_other_than_the_two_accepted_ones_is_a_400_on_both_routes(client, route, name, pandas_must_not_run):
    resp = route(client, json.dumps(REFUSED_SHAPES[name]))
    assert resp.status_code == 400
    assert "list of records" in resp.json()["detail"] and "object of columns" in resp.json()["detail"]


@pytest.mark.parametrize("name", list(REFUSED_SHAPES))
def test_the_library_loader_refuses_the_same_shapes(name):
    from mcp_server import DataLoader

    with pytest.raises(ValueError, match="list of records"):
        DataLoader.load_json(json.dumps(REFUSED_SHAPES[name]))


def test_nested_records_whose_flattened_leaf_count_exceeds_the_column_limit_are_413_before_any_dataframe(monkeypatch, pandas_must_not_run):
    from mcp_server import DataLoader, DataTooLargeError

    monkeypatch.setenv("DATA_ANALYZER_MAX_COLUMNS", "5")
    records = [{"id": 1, "p": {"a": 1, "b": 2, "c": {"d": 3, "e": 4}}}]  # leaves: id, p.a, p.b, p.c.d, p.c.e = 5
    DataLoader.check_json_shape(records, 5)  # exactly at the limit: fine
    records[0]["p"]["c"]["f"] = 5             # 6 leaves
    with pytest.raises(DataTooLargeError, match="columns"):
        DataLoader.check_json_shape(records, 5)
    with pytest.raises(DataTooLargeError):
        DataLoader.load_json(json.dumps(records))


def test_the_leaf_path_count_matches_what_json_normalize_builds_and_never_undercounts():
    from mcp_server import DataLoader

    samples = [
        [{"a": 1}],
        [{"a": {"b": 1, "c": {"d": 2}}}, {"a": {"e": 3}, "z": [1, 2]}],
        [{"a": {}}, {"a": {"b": {"c": {"d": {"e": 1}}}}}],
        [{"x": None, "y": {"z": None}}],
    ]
    for level in (0, 1, 2, 5):
        for recs in samples:
            built = len(pd.json_normalize(recs, max_level=level).columns)
            counted = DataLoader._count_columns(recs, level, 10**6)
            assert counted >= built                      # never an undercount
            if not any(v == {} for r in recs for v in r.values()):
                assert counted == built                  # exact (an empty object is the one conservative case)


def test_the_cells_limit_is_enforced_for_json_and_csv(client, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_MAX_CELLS", "10")
    resp = _multipart(client, json.dumps([{"a": i, "b": i, "c": i} for i in range(4)]))  # 12 cells
    assert resp.status_code == 413 and "DATA_ANALYZER_MAX_CELLS" in resp.json()["detail"]
    csv = client.post("/api/v1/analyze", files={"data_file": ("d.csv", b"a,b,c\n" + b"1,2,3\n" * 4, "text/csv")}, headers=H)
    assert csv.status_code == 413 and "DATA_ANALYZER_MAX_CELLS" in csv.json()["detail"]
    ok = client.post("/api/v1/analyze", files={"data_file": ("d.csv", b"a,b,c\n1,2,3\n", "text/csv")}, headers=H)
    assert ok.status_code == 200


def test_pandas_read_json_is_not_called_at_all_on_the_valid_paths(client, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("read_json must not be used")

    monkeypatch.setattr(pd, "read_json", boom)
    for route in ROUTES:
        assert route(client, json.dumps([{"a": 1}, {"a": 2}])).status_code == 200
        assert route(client, json.dumps({"a": [1, 2]})).status_code == 200


@pytest.mark.parametrize("obj", [
    [{"i": 1, "f": 1.5, "s": "a", "n": None, "b": True, "num_str": "12"},
     {"i": 2, "f": 2.0, "s": "b", "n": None, "b": False, "num_str": "13"}],
    {"i": [1, 2], "f": [1.5, 2.0], "s": ["a", "b"], "n": [None, None], "num_str": ["12", "13"]},
    [{"a": 1}, {"a": None}],
    [{"visit_date": "2026-01-05"}, {"visit_date": "not-a-date"}],
    [{"date": "2026-01-05"}, {"date": "2026-01-06"}],
    [{"created_at": "2026-01-05T10:00:00"}, {"created_at": "2026-01-06T11:30:00"}],
])
def test_built_tables_have_the_same_dtypes_and_values_as_pd_read_json(obj):
    import warnings

    from mcp_server import DataLoader

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = pd.read_json(io.BytesIO(json.dumps(obj).encode()))
    built = DataLoader.build_json_dataframe(DataLoader.precheck_json(json.dumps(obj)))
    assert [str(t) for t in built.dtypes] == [str(t) for t in expected.dtypes]
    assert built.astype(str).values.tolist() == expected.astype(str).values.tolist()
