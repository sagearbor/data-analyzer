"""
Tests for AZURE_OPENAI_DEPLOYMENTS parsing and per-call deployment override
plumbing in src/llm_client.py.

These tests never make real network calls - the Responses API call in
extract_fields_using_responses_api is mocked via requests.post.
"""
import json
from unittest.mock import patch, MagicMock

import pytest

from src.llm_client import LLMDictionaryParser, get_available_deployments


# ---------------------------------------------------------------------------
# get_available_deployments()
# ---------------------------------------------------------------------------

class TestGetAvailableDeployments:
    def test_defaults_to_single_deployment_when_unset(self, monkeypatch):
        """With neither env var set, falls back to the hardcoded default."""
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENTS", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

        assert get_available_deployments() == ["gpt-5-nano"]

    def test_falls_back_to_azure_openai_deployment_when_deployments_unset(self, monkeypatch):
        """AZURE_OPENAI_DEPLOYMENTS unset -> use AZURE_OPENAI_DEPLOYMENT as the sole entry."""
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENTS", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")

        assert get_available_deployments() == ["gpt-4o-mini"]

    def test_parses_comma_separated_list(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENTS", "gpt-5-nano,gpt-5-mini,gpt-4o-mini")

        assert get_available_deployments() == ["gpt-5-nano", "gpt-5-mini", "gpt-4o-mini"]

    def test_strips_whitespace_around_entries(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENTS", " gpt-5-nano , gpt-5-mini ,gpt-4o-mini ")

        assert get_available_deployments() == ["gpt-5-nano", "gpt-5-mini", "gpt-4o-mini"]

    def test_deduplicates_preserving_order(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENTS", "gpt-5-nano,gpt-5-mini,gpt-5-nano")

        assert get_available_deployments() == ["gpt-5-nano", "gpt-5-mini"]

    def test_empty_string_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENTS", "")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")

        assert get_available_deployments() == ["gpt-5-mini"]

    def test_whitespace_only_string_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENTS", "   ,  ,")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")

        assert get_available_deployments() == ["gpt-5-mini"]

    def test_always_returns_at_least_one_entry(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENTS", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

        deployments = get_available_deployments()
        assert isinstance(deployments, list)
        assert len(deployments) >= 1


# ---------------------------------------------------------------------------
# Per-call deployment override plumbing
# ---------------------------------------------------------------------------

@pytest.fixture
def parser(monkeypatch):
    """LLMDictionaryParser constructed with fake credentials (no network)."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-endpoint.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fake-key")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
    return LLMDictionaryParser()


def _mock_responses_api(monkeypatch, fields=None):
    """Patch requests.post to return a minimal valid Responses API payload
    and return the Mock so callers can assert on the request body."""
    fields = fields if fields is not None else []
    response_payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": json.dumps({"fields": fields}),
                    }
                ],
            }
        ]
    }

    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = response_payload

    mock_post = MagicMock(return_value=mock_response)
    monkeypatch.setattr("src.llm_client.requests.post", mock_post)
    return mock_post


class TestDeploymentOverridePlumbing:
    def test_default_deployment_used_when_no_override(self, parser, monkeypatch):
        mock_post = _mock_responses_api(monkeypatch)

        parser.extract_fields_using_responses_api("some dictionary text")

        sent_body = mock_post.call_args.kwargs["json"]
        assert sent_body["model"] == "gpt-5-nano"

    def test_per_call_override_takes_precedence(self, parser, monkeypatch):
        mock_post = _mock_responses_api(monkeypatch)

        parser.extract_fields_using_responses_api("some dictionary text", deployment="gpt-5-mini")

        sent_body = mock_post.call_args.kwargs["json"]
        assert sent_body["model"] == "gpt-5-mini"

    def test_override_does_not_mutate_client_default(self, parser, monkeypatch):
        """A per-call override must not leak into the shared client's default
        deployment - later calls without an override should still use the
        configured default."""
        _mock_responses_api(monkeypatch)

        parser.extract_fields_using_responses_api("chunk 1", deployment="gpt-5-mini")

        assert parser.deployment == "gpt-5-nano"

    def test_extract_fields_from_chunk_forwards_override(self, parser, monkeypatch):
        mock_post = _mock_responses_api(monkeypatch)

        parser.extract_fields_from_chunk("some text", deployment="gpt-4o-mini")

        sent_body = mock_post.call_args.kwargs["json"]
        assert sent_body["model"] == "gpt-4o-mini"

    def test_parse_dictionary_single_call_mode_forwards_override(self, parser, monkeypatch):
        mock_post = _mock_responses_api(monkeypatch, fields=[
            {"field_name": "age", "data_type": "int"}
        ])

        result = parser.parse_dictionary("short dictionary text", deployment="gpt-5-mini")

        sent_body = mock_post.call_args.kwargs["json"]
        assert sent_body["model"] == "gpt-5-mini"
        assert result["metadata"]["mode"] == "single-call"
        assert len(result["fields"]) == 1

    def test_parse_dictionary_without_override_uses_default(self, parser, monkeypatch):
        mock_post = _mock_responses_api(monkeypatch)

        parser.parse_dictionary("short dictionary text")

        sent_body = mock_post.call_args.kwargs["json"]
        assert sent_body["model"] == "gpt-5-nano"

    def test_parse_dictionary_chunked_mode_forwards_override(self, parser, monkeypatch):
        """Large dictionaries fall into the chunked branch - the override
        must be threaded through each chunk's call too."""
        mock_post = _mock_responses_api(monkeypatch)

        # Force chunked mode regardless of actual token count of the input.
        monkeypatch.setattr(parser, "count_tokens", lambda text: 90000)
        monkeypatch.setattr(parser, "chunk_text", lambda text, max_tokens=4500: ["chunk one", "chunk two"])

        result = parser.parse_dictionary("irrelevant text", deployment="gpt-5-mini")

        assert result["metadata"]["mode"] == "chunked"
        assert mock_post.call_count == 2
        for call in mock_post.call_args_list:
            assert call.kwargs["json"]["model"] == "gpt-5-mini"

    def test_max_output_tokens_uses_override_deployment_limits(self, parser, monkeypatch):
        """MODEL_OUTPUT_LIMITS lookup should key off the override, not the
        client's configured default, since different deployments can have
        different output limits."""
        mock_post = _mock_responses_api(monkeypatch)

        parser.extract_fields_using_responses_api("text", deployment="gpt-4o-mini")

        sent_body = mock_post.call_args.kwargs["json"]
        assert sent_body["max_output_tokens"] == 16384  # gpt-4o-mini limit
