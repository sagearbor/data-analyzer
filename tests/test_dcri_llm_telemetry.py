"""
Offline tests for the dcri_llm integration in src/llm_client.py.

One LLM call must emit exactly one schema-v1 `llm_call` event carrying token
fields and no prompt/completion text, use a bearer token (no api-key header),
and turn truncation / empty output / HTTP errors into LLMCallError instead of
silently returning zero fields.
"""
import json
from unittest.mock import MagicMock

import pytest
import requests

from src import dcri_llm
from src.llm_client import LLMCallError, LLMDictionaryParser, max_output_tokens_for

SECRET_PROMPT = "patient_mrn_column: Integer, Required - SECRET-PROMPT-MARKER"
SECRET_COMPLETION_FIELD = "secret_completion_marker_field"


def _responses_payload(text, status="completed", incomplete_reason=None):
    payload = {
        "model": "gpt-5-nano-2025-08-07",
        "status": status,
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "content": [{"type": "output_text", "text": text}]},
        ],
        "usage": {
            "input_tokens": 1234,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 567,
            "output_tokens_details": {"reasoning_tokens": 320},
        },
    }
    if incomplete_reason:
        payload["incomplete_details"] = {"reason": incomplete_reason}
    return payload


@pytest.fixture
def parser(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-endpoint.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
    monkeypatch.setenv("SOLUTION_NAME", "data-analyzer-api")
    monkeypatch.delenv("TELEMETRY_BLOB_ACCOUNT_URL", raising=False)
    return LLMDictionaryParser()


@pytest.fixture
def events():
    captured = []
    dcri_llm.add_sink(captured.append)
    yield captured
    dcri_llm._extra_sinks.remove(captured.append)


def _mock_post(monkeypatch, payload=None, exc=None):
    resp = MagicMock()
    if exc is not None:
        resp.raise_for_status.side_effect = exc
    else:
        resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    post = MagicMock(return_value=resp)
    monkeypatch.setattr("src.llm_client.requests.post", post)
    return post


def test_one_call_emits_one_schema_v1_event_without_text(parser, events, monkeypatch, capsys):
    completion = json.dumps({"fields": [{"field_name": SECRET_COMPLETION_FIELD, "data_type": "int"}]})
    post = _mock_post(monkeypatch, _responses_payload(completion))

    with dcri_llm.llm_action("api_parse_dictionary"):
        fields = parser.extract_fields_using_responses_api(SECRET_PROMPT)

    assert [f.field_name for f in fields] == [SECRET_COMPLETION_FIELD]
    assert len(events) == 1
    ev = events[0]
    assert ev["event"] == "llm_call" and ev["schema"] == 1
    assert ev["solution"] == "data-analyzer-api"
    assert ev["operation"] == "api_parse_dictionary"
    assert ev["api"] == "responses"
    assert ev["endpoint_host"] == "fake-endpoint.openai.azure.com"
    assert ev["model_requested"] == "gpt-5-nano"
    assert ev["input_tokens"] == 1234
    assert ev["output_tokens"] == 567
    assert ev["reasoning_tokens"] == 320
    assert ev["max_completion_tokens"] >= 32000
    assert ev["budget_below_floor"] is False
    assert ev["finish_reason"] == "stop" and ev["outcome"] == "ok"
    assert ev["empty_output"] is False and ev["finish_length"] is False
    assert "dropped_fields" not in ev

    # No prompt or completion text anywhere in the event or on stdout.
    blob = json.dumps(ev) + capsys.readouterr().out
    assert "SECRET-PROMPT-MARKER" not in blob
    assert "patient_mrn_column" not in blob
    assert SECRET_COMPLETION_FIELD not in blob

    # Keyless: bearer token, never an api-key header.
    headers = post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer fake-token"
    assert "api-key" not in {k.lower() for k in headers}


def test_finish_length_raises_and_is_recorded(parser, events, monkeypatch):
    _mock_post(monkeypatch, _responses_payload("{\"fields\": [", status="incomplete",
                                               incomplete_reason="max_output_tokens"))
    with pytest.raises(LLMCallError, match="truncated"):
        parser.extract_fields_using_responses_api("text")
    assert len(events) == 1
    assert events[0]["finish_length"] is True and events[0]["outcome"] == "error"


def test_empty_output_raises(parser, events, monkeypatch):
    _mock_post(monkeypatch, _responses_payload(""))
    with pytest.raises(LLMCallError, match="no text"):
        parser.parse_dictionary("short dictionary")
    assert events[0]["empty_output"] is True


def test_http_error_raises_and_is_recorded(parser, events, monkeypatch):
    err = requests.HTTPError("403 Forbidden")
    err.response = MagicMock(status_code=403)
    _mock_post(monkeypatch, exc=err)
    with pytest.raises(LLMCallError, match="403"):
        parser.extract_fields_using_responses_api("text")
    assert events[0]["http_status"] == 403 and events[0]["outcome"] == "error"


def test_invalid_json_raises_instead_of_zero_fields(parser, events, monkeypatch):
    _mock_post(monkeypatch, _responses_payload("this is not json {{{"))
    with pytest.raises(LLMCallError):
        parser.extract_fields_using_responses_api("text")


def test_reasoning_budget_never_below_floor():
    for model in ("gpt-5-nano", "gpt-5-mini", "gpt-5.4-nano", "o4-mini"):
        assert max_output_tokens_for(model) >= 32000
    assert max_output_tokens_for("gpt-4o-mini") == 16384  # non-reasoning, documented cap


def test_validate_data_with_llm_uses_full_budget(parser, events, monkeypatch):
    post = _mock_post(monkeypatch, _responses_payload(json.dumps({"issues": [], "summary": "ok",
                                                                  "recommendations": []})))
    result = parser.validate_data_with_llm("a,b\n1,2", {"a": {"type": "int"}})
    assert result["summary"] == "ok"
    assert post.call_args.kwargs["json"]["max_output_tokens"] >= 32000
    assert len(events) == 1
