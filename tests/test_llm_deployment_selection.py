"""
Tests for AZURE_OPENAI_DEPLOYMENTS parsing and per-call deployment override
plumbing in src/llm_client.py.

These tests never make real network calls - the Responses API call in
extract_fields_using_responses_api is mocked via requests.post.
"""
import json
from unittest.mock import patch, MagicMock

import pytest

from src.llm_client import (
    LLMCallError,
    LLMDictionaryParser,
    get_available_deployments,
    max_output_tokens_for,
    resolve_deployment_choice,
)


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
    """LLMDictionaryParser with a fake endpoint (no network; token faked in conftest)."""
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-endpoint.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
    # ADP-52: an override must be on the allow-list to be used at all.
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENTS", "gpt-5-nano,gpt-5-mini,gpt-4o-mini")
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
        mock_post = _mock_responses_api(monkeypatch, fields=[{"field_name": "age", "data_type": "int"}])

        parser.parse_dictionary("short dictionary text")

        sent_body = mock_post.call_args.kwargs["json"]
        assert sent_body["model"] == "gpt-5-nano"

    def test_parse_dictionary_chunked_mode_forwards_override(self, parser, monkeypatch):
        """Large dictionaries fall into the chunked branch - the override
        must be threaded through each chunk's call too."""
        mock_post = _mock_responses_api(monkeypatch, fields=[{"field_name": "age", "data_type": "int"}])

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


# ---------------------------------------------------------------------------
# ADP-52: the allow-list is enforced where the call is made
# ---------------------------------------------------------------------------

ALLOWED = "gpt-5-nano,gpt-5.4-mini,gpt-5.5"


@pytest.fixture
def allowlisted_parser(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-endpoint.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENTS", ALLOWED)
    return LLMDictionaryParser()


class TestDeploymentAllowList:
    def test_override_not_on_the_allow_list_is_refused_before_any_http_call(
            self, allowlisted_parser, monkeypatch):
        mock_post = _mock_responses_api(monkeypatch)

        with pytest.raises(LLMCallError) as excinfo:
            allowlisted_parser.extract_fields_using_responses_api("text", deployment="gpt-9-unlisted")

        mock_post.assert_not_called()
        assert excinfo.value.error_class == "deployment_not_allowed"
        assert excinfo.value.model == "gpt-9-unlisted"
        assert "gpt-9-unlisted" in str(excinfo.value)

    def test_parse_dictionary_refuses_a_disallowed_override_without_calling_out(
            self, allowlisted_parser, monkeypatch):
        mock_post = _mock_responses_api(monkeypatch)

        with pytest.raises(LLMCallError) as excinfo:
            allowlisted_parser.parse_dictionary("short dictionary text", deployment="not-listed")

        mock_post.assert_not_called()
        assert excinfo.value.error_class == "deployment_not_allowed"

    def test_chunked_parse_refuses_a_disallowed_override_without_calling_out(
            self, allowlisted_parser, monkeypatch):
        mock_post = _mock_responses_api(monkeypatch)
        monkeypatch.setattr(allowlisted_parser, "count_tokens", lambda text: 90000)
        monkeypatch.setattr(allowlisted_parser, "chunk_text", lambda text, max_tokens=4500: ["a", "b"])

        with pytest.raises(LLMCallError):
            allowlisted_parser.parse_dictionary("x", deployment="not-listed")

        mock_post.assert_not_called()

    def test_extract_fields_from_chunk_refuses_a_disallowed_override(
            self, allowlisted_parser, monkeypatch):
        mock_post = _mock_responses_api(monkeypatch)

        with pytest.raises(LLMCallError):
            allowlisted_parser.extract_fields_from_chunk("text", deployment="not-listed")

        mock_post.assert_not_called()

    def test_refusal_is_logged_once_as_llm_failure_without_leaking_control_characters(
            self, allowlisted_parser, monkeypatch, caplog):
        _mock_responses_api(monkeypatch)
        hostile = "evil\nllm_failure operation=fake model=x"

        with caplog.at_level("ERROR"):
            with pytest.raises(LLMCallError):
                allowlisted_parser.extract_fields_using_responses_api("t", deployment=hostile)

        lines = [r.getMessage() for r in caplog.records if "llm_failure" in r.getMessage()]
        assert len(lines) == 1
        assert "\n" not in lines[0]
        assert "error_class=deployment_not_allowed" in lines[0]

    @pytest.mark.parametrize("allowed", ["gpt-5-nano", "gpt-5.4-mini", "gpt-5.5"])
    def test_override_on_the_allow_list_is_sent_to_the_api(
            self, allowlisted_parser, monkeypatch, allowed):
        mock_post = _mock_responses_api(monkeypatch)

        allowlisted_parser.extract_fields_using_responses_api("text", deployment=allowed)

        assert mock_post.call_count == 1
        assert mock_post.call_args.kwargs["json"]["model"] == allowed

    def test_no_override_still_uses_the_configured_default(self, allowlisted_parser, monkeypatch):
        mock_post = _mock_responses_api(monkeypatch)

        allowlisted_parser.extract_fields_using_responses_api("text")

        assert mock_post.call_args.kwargs["json"]["model"] == "gpt-5-nano"

    def test_allow_list_follows_the_environment_at_call_time(self, allowlisted_parser, monkeypatch):
        mock_post = _mock_responses_api(monkeypatch)
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENTS", "gpt-5-nano,brand-new-model")

        allowlisted_parser.extract_fields_using_responses_api("text", deployment="brand-new-model")

        assert mock_post.call_args.kwargs["json"]["model"] == "brand-new-model"

    def test_with_only_the_single_deployment_configured_other_models_are_refused(
            self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-endpoint.openai.azure.com/")
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENTS", raising=False)
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
        mock_post = _mock_responses_api(monkeypatch)

        with pytest.raises(LLMCallError):
            LLMDictionaryParser().extract_fields_using_responses_api("t", deployment="gpt-5.5")

        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# ADP-52: output budget floor (HARD RULE: reasoning models never below 40000)
# ---------------------------------------------------------------------------

LISTABLE_DEPLOYMENTS = [
    "gpt-5-nano", "gpt-5.4-nano", "gpt-5.4-mini", "gpt-5.5",
    "gpt-5.6-sol", "gpt-5.6-luna", "gpt-5.6-terra",
    "grok-4-1-fast-reasoning", "grok-4.3",
    "Mistral-Large-3", "Kimi-K2.6", "DeepSeek-V4-Pro", "DeepSeek-V4-Flash",
]


@pytest.mark.parametrize("deployment", LISTABLE_DEPLOYMENTS)
def test_output_budget_is_at_least_40000_for_every_listable_deployment(deployment):
    assert max_output_tokens_for(deployment) >= 40000


@pytest.mark.parametrize("deployment", ["Mistral-Large-3", "GROK-4.3", "some-future-model", "o4-mini"])
def test_output_budget_floor_applies_to_names_not_in_the_limits_table_and_any_case(deployment):
    assert max_output_tokens_for(deployment) >= 40000


def test_output_budget_floor_beats_a_low_limit_for_a_non_gpt_reasoning_model(monkeypatch):
    """If someone later adds a small cap for a reasoning model to MODEL_OUTPUT_LIMITS,
    the 40000 floor still wins."""
    monkeypatch.setitem(__import__("src.llm_client", fromlist=["x"]).MODEL_OUTPUT_LIMITS,
                        "deepseek-v4-pro", 8192)
    assert max_output_tokens_for("DeepSeek-V4-Pro") == 40000


def test_documented_non_reasoning_limit_for_gpt_4o_mini_is_not_changed():
    assert max_output_tokens_for("gpt-4o-mini") == 16384


def test_existing_higher_limits_are_not_lowered():
    assert max_output_tokens_for("gpt-5-nano") == 128000
    assert max_output_tokens_for("gpt-5.5") == 128000


def test_the_request_sent_for_a_listed_model_carries_a_budget_of_at_least_40000(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-endpoint.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENTS", "gpt-5-nano,Kimi-K2.6")
    mock_post = _mock_responses_api(monkeypatch)

    LLMDictionaryParser().extract_fields_using_responses_api("t", deployment="Kimi-K2.6")

    assert mock_post.call_args.kwargs["json"]["max_output_tokens"] >= 40000


# ---------------------------------------------------------------------------
# ADP-52: UI selection helper (stale session value)
# ---------------------------------------------------------------------------

class TestResolveDeploymentChoice:
    OPTIONS = ["gpt-5-nano", "gpt-5.4-mini", "gpt-5.5"]

    def test_stale_value_no_longer_in_the_list_falls_back_to_the_first_entry(self):
        assert resolve_deployment_choice("retired-model", self.OPTIONS) == "gpt-5-nano"

    def test_missing_value_falls_back_to_the_first_entry(self):
        assert resolve_deployment_choice(None, self.OPTIONS) == "gpt-5-nano"
        assert resolve_deployment_choice("", self.OPTIONS) == "gpt-5-nano"

    def test_valid_value_is_kept(self):
        assert resolve_deployment_choice("gpt-5.5", self.OPTIONS) == "gpt-5.5"

    def test_empty_options_gives_none(self):
        assert resolve_deployment_choice("gpt-5.5", []) is None


class TestModelSelectboxInTheApp:
    """Render web_app headlessly (Streamlit AppTest) with a controlled allow-list."""

    @staticmethod
    def _run(monkeypatch, session=None):
        from pathlib import Path
        from streamlit.testing.v1 import AppTest

        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://fake-endpoint.openai.azure.com/")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENTS", ALLOWED)
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "web_app.py"),
                               default_timeout=90)
        for k, v in (session or {}).items():
            at.session_state[k] = v
        at.run()
        assert not at.exception
        return at

    def test_default_is_the_first_entry_and_the_page_names_the_model_in_use(self, monkeypatch):
        at = self._run(monkeypatch)

        box = at.selectbox(key="llm_deployment_select")
        assert box.options == ["gpt-5-nano", "gpt-5.4-mini", "gpt-5.5"]
        assert box.value == "gpt-5-nano"
        assert any("Model in use" in c.value and "gpt-5-nano" in c.value for c in at.caption)
        assert at.session_state["llm_deployment"] == "gpt-5-nano"

    def test_stale_session_value_falls_back_to_the_first_entry_without_error(self, monkeypatch):
        at = self._run(monkeypatch, {"llm_deployment_select": "retired-model",
                                     "llm_deployment": "retired-model"})

        assert at.selectbox(key="llm_deployment_select").value == "gpt-5-nano"
        assert at.session_state["llm_deployment"] == "gpt-5-nano"

    def test_a_still_listed_choice_is_kept_across_reruns(self, monkeypatch):
        at = self._run(monkeypatch, {"llm_deployment_select": "gpt-5.5"})

        assert at.selectbox(key="llm_deployment_select").value == "gpt-5.5"
        assert any("gpt-5.5" in c.value for c in at.caption if "Model in use" in c.value)


@pytest.mark.parametrize("manifest", ["containerapp.yaml", "containerapp-ui.yaml"])
def test_manifest_template_lists_the_allowed_models_as_a_plain_value(manifest):
    from pathlib import Path
    import yaml

    doc = yaml.safe_load((Path(__file__).resolve().parent.parent / "deploy" / "aca" / manifest).read_text())
    env = {e["name"]: e for c in doc["properties"]["template"]["containers"] for e in c["env"]}

    entry = env["AZURE_OPENAI_DEPLOYMENTS"]
    assert entry["value"] == "gpt-5-nano,gpt-5.4-mini,gpt-5.5"
    assert "secretRef" not in entry
