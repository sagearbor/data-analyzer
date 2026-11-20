"""Security: caller-supplied content is never treated as a server file path
(ADP-77: A-05, B-07)."""

import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import analysis_service
import mcp_server
from mcp_server import DataLoader

SECRET_LINE = "SYNTHETIC_SECRET=canary-value-123"


@pytest.fixture
def canary_file(tmp_path):
    path = tmp_path / "canary.txt"
    path.write_text(SECRET_LINE + "\nsecond line\n")
    return str(path)


def test_analysis_service_get_info_does_not_read_a_server_file_named_in_the_content(canary_file):
    result = analysis_service.get_info({"data_content": canary_file, "file_format": "csv"})

    assert SECRET_LINE not in json.dumps(result, default=str)
    assert "second line" not in json.dumps(result, default=str)


def test_analysis_service_analyze_does_not_read_a_server_file_named_in_the_content(canary_file):
    result = analysis_service.analyze({"data_content": canary_file, "file_format": "csv"})

    assert SECRET_LINE not in json.dumps(result, default=str)
    assert "second line" not in json.dumps(result, default=str)


def test_analysis_service_json_format_does_not_read_a_server_file(canary_file, tmp_path):
    path = tmp_path / "secret.json"
    path.write_text(json.dumps([{"leak": "canary-json-value"}]))

    result = analysis_service.get_info({"data_content": str(path), "file_format": "json"})

    assert "canary-json-value" not in json.dumps(result, default=str)


def test_mcp_tool_handler_does_not_read_a_server_file_named_in_the_arguments(canary_file):
    out = asyncio.run(
        mcp_server.handle_call_tool("get_data_info", {"data_content": canary_file, "file_format": "csv"})
    )

    assert SECRET_LINE not in out[0].text
    assert "second line" not in out[0].text


def test_a2a_agent_does_not_read_a_server_file_named_in_the_payload(canary_file):
    import a2a_agent

    result = asyncio.run(
        a2a_agent.DataAnalyzerAgent().run(
            {"operation": "get_info", "data_content": canary_file, "file_format": "csv"}
        )
    )

    assert SECRET_LINE not in json.dumps(result, default=str)


def test_content_that_looks_like_a_path_is_parsed_as_csv_text_by_default(canary_file):
    df = DataLoader.load_csv(canary_file)

    assert list(df.columns) == [canary_file]  # the string itself became the header


def test_loader_never_opens_a_path_for_json_or_excel_or_parquet_by_default(canary_file):
    with pytest.raises(ValueError):
        DataLoader.load_json(canary_file)
    with pytest.raises(ValueError):
        DataLoader.load_excel(canary_file)
    with pytest.raises(ValueError):
        DataLoader.load_parquet(canary_file)


def test_load_data_is_content_only_unless_paths_are_explicitly_allowed(canary_file):
    df = DataLoader.load_data(canary_file, "csv")

    assert SECRET_LINE not in list(df.columns)


def test_trusted_local_file_loader_still_reads_files_for_the_cli_and_app(tmp_path):
    path = tmp_path / "ok.csv"
    path.write_text("a,b\n1,2\n")

    df = DataLoader.load_local_file(str(path), "csv")

    assert len(df) == 1 and list(df.columns) == ["a", "b"]


def test_trusted_local_file_loader_rejects_a_missing_file_without_echoing_the_directory(tmp_path):
    with pytest.raises(ValueError) as exc:
        DataLoader.load_local_file(str(tmp_path / "nope.csv"), "csv")

    assert str(tmp_path) not in str(exc.value)


def test_content_only_callers_never_use_the_trusted_loader_or_allow_paths():
    """Static guard: the adapter modules must not call load_local_file or pass allow_paths=True."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for name in ("analysis_service.py", "a2a_agent.py", "api_server.py"):
        text = open(os.path.join(root, name), encoding="utf-8").read()
        assert "load_local_file(" not in text, name
        assert "allow_paths=True" not in text, name
    mcp_text = open(os.path.join(root, "mcp_server.py"), encoding="utf-8").read()
    handler = mcp_text[mcp_text.index("async def handle_call_tool"):]
    assert "load_local_file(" not in handler
    assert "allow_paths=True" not in handler
