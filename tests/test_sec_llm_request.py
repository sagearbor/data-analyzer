"""Security guarantees for the model request (ADP-76 store flag, ADP-73 bounds, ADP-83 redirects/https).

All offline: requests.post is mocked; no network, synthetic text only.
"""
import json
from unittest.mock import MagicMock

import pytest

from src.llm_client import LLMCallError, LLMDictionaryParser

ENDPOINT = "https://fake-endpoint.openai.azure.com/"


def _ok_payload(fields=None):
    return {"output": [{"type": "message", "content": [
        {"type": "output_text", "text": json.dumps({"fields": fields or [{"field_name": "age", "data_type": "int"}]})}]}]}


def _resp(status=200, payload=None):
    r = MagicMock()
    r.status_code = status
    r.raise_for_status.return_value = None
    r.json.return_value = payload if payload is not None else _ok_payload()
    return r


@pytest.fixture
def parser(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENTS", raising=False)
    for v in ("LLM_RESPONSES_STORE", "LLM_MAX_CHUNKS", "LLM_MAX_INPUT_CHARS", "LLM_ALLOW_INSECURE_ENDPOINT"):
        monkeypatch.delenv(v, raising=False)
    return LLMDictionaryParser()


def _post(monkeypatch, *responses):
    mock = MagicMock(side_effect=list(responses)) if len(responses) > 1 else MagicMock(return_value=responses[0])
    monkeypatch.setattr("src.llm_client.requests.post", mock)
    return mock


def test_model_request_asks_service_not_to_store_by_default(parser, monkeypatch):
    post = _post(monkeypatch, _resp())
    parser.extract_fields_using_responses_api("age: integer")
    assert post.call_args.kwargs["json"]["store"] is False


def test_store_flag_can_be_omitted_with_one_env_var(parser, monkeypatch):
    monkeypatch.setenv("LLM_RESPONSES_STORE", "omit")
    post = _post(monkeypatch, _resp())
    parser.extract_fields_using_responses_api("age: integer")
    assert "store" not in post.call_args.kwargs["json"]


def test_model_request_never_follows_redirects(parser, monkeypatch):
    post = _post(monkeypatch, _resp())
    parser.extract_fields_using_responses_api("age: integer")
    assert post.call_args.kwargs["allow_redirects"] is False


def test_a_redirect_from_the_model_endpoint_is_a_failure(parser, monkeypatch):
    post = _post(monkeypatch, _resp(status=302))
    with pytest.raises(LLMCallError) as exc:
        parser.extract_fields_using_responses_api("age: integer")
    assert exc.value.error_class == "redirect_refused"
    assert post.call_count == 1


def test_plain_http_endpoint_is_refused(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "http://model.example.test/")
    monkeypatch.delenv("LLM_ALLOW_INSECURE_ENDPOINT", raising=False)
    with pytest.raises(ValueError):
        LLMDictionaryParser()


def test_plain_http_endpoint_allowed_only_with_explicit_local_dev_override(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "http://localhost:9999/")
    monkeypatch.setenv("LLM_ALLOW_INSECURE_ENDPOINT", "1")
    assert LLMDictionaryParser().endpoint.startswith("http://")


def test_http_endpoint_changed_after_startup_is_still_refused_at_call_time(parser, monkeypatch):
    post = _post(monkeypatch, _resp())
    parser.endpoint = "http://model.example.test"
    with pytest.raises(LLMCallError) as exc:
        parser.extract_fields_using_responses_api("age: integer")
    assert exc.value.error_class == "endpoint_not_https"
    post.assert_not_called()


def test_oversized_dictionary_is_rejected_before_any_model_call(parser, monkeypatch):
    monkeypatch.setenv("LLM_MAX_INPUT_CHARS", "1000")
    post = _post(monkeypatch, _resp())
    with pytest.raises(LLMCallError) as exc:
        parser.parse_dictionary("x" * 1001)
    assert exc.value.error_class == "input_too_large"
    post.assert_not_called()


def test_dictionary_needing_too_many_chunks_is_rejected_not_truncated(parser, monkeypatch):
    monkeypatch.setenv("LLM_MAX_CHUNKS", "2")
    post = _post(monkeypatch, _resp())
    monkeypatch.setattr(parser, "count_tokens", lambda text: 100000)  # force chunked mode
    monkeypatch.setattr(parser, "chunk_text", lambda text, max_tokens=4500: ["a", "b", "c"])
    with pytest.raises(LLMCallError) as exc:
        parser.parse_dictionary("anything")
    assert exc.value.error_class == "too_many_chunks"
    post.assert_not_called()


def test_chunk_count_at_the_limit_is_still_processed(parser, monkeypatch):
    monkeypatch.setenv("LLM_MAX_CHUNKS", "2")
    post = _post(monkeypatch, _resp())
    monkeypatch.setattr(parser, "count_tokens", lambda text: 100000)
    monkeypatch.setattr(parser, "chunk_text", lambda text, max_tokens=4500: ["a", "b"])
    result = parser.parse_dictionary("anything")
    assert result["metadata"]["chunks_processed"] == 2
    assert post.call_count == 2


def test_output_token_budget_is_never_below_the_40000_floor(parser, monkeypatch):
    post = _post(monkeypatch, _resp())
    parser.extract_fields_using_responses_api("age: integer")
    assert post.call_args.kwargs["json"]["max_output_tokens"] >= 40000
