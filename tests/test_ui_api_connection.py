"""
UI -> API connection must work on Azure and fail LOUDLY when misconfigured.

Covers (all offline; requests.post is mocked):
  * deployed env without DATA_ANALYZER_API_URL -> one-line config error, no
    localhost fallback, no HTTP call; local env keeps the localhost default
  * API cold start: 90 s timeout, ONE retry with a "starting..." notice
  * logic-validation failures are surfaced in the result and logged
  * the "LLM ACTIVE" model label comes from config, not a hard-coded name
"""
import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

import web_app
from src.runtime_env import ApiConfigError
from web_app import DataQualityAnalyzer

GOOD_REPORT = {
    "summary": {"total_rows": 2, "total_columns": 2, "issues_found": 0,
                "critical_issues": 0, "warnings": 0, "data_types": {}, "completeness": 100.0},
    "issues": [], "recommendations": [], "quality_checks": {}, "summary_stats": {},
}


def _resp(status=200, body=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = body if body is not None else GOOD_REPORT
    r.text = ""
    return r


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("CONTAINER_APP_NAME", "DATA_ANALYZER_API_URL", "DATA_ANALYZER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setattr("web_app.time.sleep", lambda s: None)


# ---------------------------------------------------------------------------
# Missing API URL in a deployed environment
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env", [{"APP_ENV": "prod"}, {"APP_ENV": "staging"},
                                 {"APP_ENV": "dev", "CONTAINER_APP_NAME": "data-analyzer-ui"}])
@pytest.mark.asyncio
async def test_deployed_without_api_url_is_a_clear_config_error(monkeypatch, df, caplog, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    with caplog.at_level(logging.ERROR, logger="data_analyzer.ui"):
        analyzer = DataQualityAnalyzer()
    assert analyzer.api_base_url is None
    assert "DATA_ANALYZER_API_URL is not set" in analyzer.config_error
    assert "\n" not in analyzer.config_error  # one line, no traceback
    assert "ui_config_error" in caplog.text

    with patch("web_app.requests.post") as post:
        with pytest.raises(ApiConfigError, match="DATA_ANALYZER_API_URL"):
            await analyzer.analyze_data_quality(df, None)
    post.assert_not_called()  # never silently tries localhost


@pytest.mark.asyncio
async def test_local_env_keeps_localhost_default(df):
    analyzer = DataQualityAnalyzer()
    assert analyzer.config_error is None
    with patch("web_app.requests.post", return_value=_resp()) as post:
        await analyzer.analyze_data_quality(df, None)
    assert post.call_args.args[0] == "http://localhost:8000/api/v1/analyze"


@pytest.mark.asyncio
async def test_deployed_with_url_and_key_calls_api(monkeypatch, df):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("DATA_ANALYZER_API_URL", "https://data-analyzer-api.example.test/")
    monkeypatch.setenv("DATA_ANALYZER_API_KEY", "k")
    analyzer = DataQualityAnalyzer()
    with patch("web_app.requests.post", return_value=_resp()) as post:
        await analyzer.analyze_data_quality(df, None)
    assert post.call_args.args[0] == "https://data-analyzer-api.example.test/api/v1/analyze"
    assert post.call_args.kwargs["headers"] == {"X-API-Key": "k"}


# ---------------------------------------------------------------------------
# Cold start: long timeout + one retry with a user-visible notice
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_uses_90s_timeout(df):
    with patch("web_app.requests.post", return_value=_resp()) as post:
        await DataQualityAnalyzer().analyze_data_quality(df, None)
    assert post.call_args.kwargs["timeout"] == 90


@pytest.mark.parametrize("first", [
    requests.exceptions.ConnectionError("refused"),
    requests.exceptions.ReadTimeout("slow"),
    _resp(503, {"detail": "starting"}),
    _resp(504, {"detail": "gateway timeout"}),
])
@pytest.mark.asyncio
async def test_cold_start_retries_once_and_notifies(df, first):
    analyzer = DataQualityAnalyzer()
    notices = []
    analyzer.on_status = notices.append
    with patch("web_app.requests.post", side_effect=[first, _resp()]) as post:
        result = await analyzer.analyze_data_quality(df, None)
    assert post.call_count == 2
    assert result["summary"]["total_rows"] == 2
    assert len(notices) == 1 and "Starting the analysis service" in notices[0]


@pytest.mark.asyncio
async def test_second_failure_raises_one_line_error_and_logs(df, caplog):
    analyzer = DataQualityAnalyzer()
    err = requests.exceptions.ConnectionError("Connection refused")
    with caplog.at_level(logging.ERROR, logger="data_analyzer.ui"):
        with patch("web_app.requests.post", side_effect=[err, err]) as post:
            with pytest.raises(RuntimeError) as exc:
                await analyzer.analyze_data_quality(df, None)
    assert post.call_count == 2  # exactly one retry
    assert "Could not connect" in str(exc.value) and "Retried once" in str(exc.value)
    assert "\n" not in str(exc.value)
    assert caplog.text.count("api_call_failed") == 2


@pytest.mark.asyncio
async def test_auth_error_is_not_retried(df):
    with patch("web_app.requests.post", return_value=_resp(401, {"detail": "Authentication required."})) as post:
        with pytest.raises(RuntimeError, match="HTTP 401"):
            await DataQualityAnalyzer().analyze_data_quality(df, None)
    assert post.call_count == 1


# ---------------------------------------------------------------------------
# Logic validation failures are surfaced, not just printed
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_logic_validation_error_is_surfaced_and_logged(monkeypatch, df, caplog):
    if not web_app.PROGRAM_CACHE_AVAILABLE:
        pytest.skip("logic engine not importable")

    class Boom:
        def extract_rules_from_fields(self, *a, **k):
            raise ValueError("bad branching expression")

    monkeypatch.setattr(web_app, "RuleExtractor", Boom)
    with caplog.at_level(logging.ERROR, logger="data_analyzer.ui"):
        with patch("web_app.requests.post", return_value=_resp()):
            result = await DataQualityAnalyzer().analyze_data_quality(df, {"fields": [{"field_name": "a"}]})
    assert "Conditional-logic checks could not run" in result["logic_validation_error"]
    assert "ValueError" in result["logic_validation_error"]
    assert "logic_validation_failed" in caplog.text


@pytest.mark.asyncio
async def test_no_logic_error_when_logic_validation_ok(df):
    with patch("web_app.requests.post", return_value=_resp()):
        result = await DataQualityAnalyzer().analyze_data_quality(df, None)
    assert result["logic_validation_error"] is None


# ---------------------------------------------------------------------------
# Model label reads config
# ---------------------------------------------------------------------------

def test_model_label_reads_deployment_from_config(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENTS", raising=False)
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")
    assert web_app.llm_model_label() == "gpt-5-nano"
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "some-other-deployment")
    assert web_app.llm_model_label() == "some-other-deployment"


def test_no_hardcoded_model_name_in_llm_banner():
    src = open(web_app.__file__, encoding="utf-8").read()
    assert "GPT-4 for intelligent" not in src
    assert "llm_model_label()" in src
