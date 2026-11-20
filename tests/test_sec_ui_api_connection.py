"""ADP-83 UI part: the API key is never sent along a redirect, and the API URL
is validated (https when deployed, loud failure when malformed)."""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import web_app
from src.runtime_env import ApiConfigError, resolve_api_base_url, ui_config_problems
from web_app import DataQualityAnalyzer


def _resp(status, headers=None):
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.text = ""
    r.json.return_value = {}
    return r


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("CONTAINER_APP_NAME", "DATA_ANALYZER_API_URL", "DATA_ANALYZER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setattr("web_app.time.sleep", lambda s: None)


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2]})


@pytest.mark.asyncio
async def test_api_call_with_the_key_disables_redirect_following(df, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_API_KEY", "secret-test-key")
    good = {"summary": {}, "issues": [], "recommendations": [], "quality_checks": {}, "summary_stats": {}}
    resp = _resp(200)
    resp.json.return_value = good
    with patch("web_app.requests.post", return_value=resp) as post:
        await DataQualityAnalyzer().analyze_data_quality(df, None)
    assert post.call_args.kwargs["allow_redirects"] is False
    assert post.call_args.kwargs["headers"] == {"X-API-Key": "secret-test-key"}


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.asyncio
async def test_a_redirect_from_the_api_is_an_error_and_is_not_followed(df, monkeypatch, status):
    monkeypatch.setenv("DATA_ANALYZER_API_KEY", "secret-test-key")
    with patch("web_app.requests.post",
               return_value=_resp(status, {"Location": "https://other.invalid/"})) as post:
        with pytest.raises(RuntimeError, match="redirect"):
            await DataQualityAnalyzer().analyze_data_quality(df, None)
    assert post.call_count == 1          # no retry, no second request to Location


def test_every_requests_call_in_the_ui_disables_redirects():
    import ast
    tree = ast.parse(open("web_app.py", encoding="utf-8").read())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
             and n.func.value.id == "requests" and n.func.attr in ("get", "post", "put", "delete", "request")]
    assert calls, "expected at least the analyze call"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert isinstance(kw.get("allow_redirects"), ast.Constant) and kw["allow_redirects"].value is False


# --- URL validation -----------------------------------------------------------
DEPLOYED = {"APP_ENV": "prod", "DATA_ANALYZER_API_KEY": "k"}


def test_deployed_http_to_a_public_style_host_is_refused():
    env = {**DEPLOYED, "DATA_ANALYZER_API_URL": "http://api.example.invalid"}
    with pytest.raises(ApiConfigError, match="https"):
        resolve_api_base_url(env)
    assert any("https" in p for p in ui_config_problems(env))


def test_deployed_https_is_accepted():
    env = {**DEPLOYED, "DATA_ANALYZER_API_URL": "https://api.example.invalid/"}
    assert resolve_api_base_url(env) == "https://api.example.invalid"
    assert ui_config_problems(env) == []


def test_http_stays_allowed_locally_and_for_a_compose_service_name():
    assert resolve_api_base_url({"APP_ENV": "dev", "DATA_ANALYZER_API_URL": "http://api.example.invalid:8000"}) \
        == "http://api.example.invalid:8000"
    assert resolve_api_base_url({"APP_ENV": "dev", "DATA_ANALYZER_API_URL": "http://localhost:8000"}) \
        == "http://localhost:8000"
    assert resolve_api_base_url({**DEPLOYED, "DATA_ANALYZER_API_URL": "http://api:8000"}) == "http://api:8000"


@pytest.mark.parametrize("url", [
    "ftp://api.example.invalid", "https://", "https://user:pw@api.example.invalid",
    "https://api.example.invalid/x?token=1", "https://api.example.invalid#frag",
    "https://api.example.invalid:notaport", "https://api .example.invalid",
    "file:///etc/passwd", "javascript:alert(1)", "https:/api.example.invalid",
])
@pytest.mark.parametrize("env", [{"APP_ENV": "dev"}, DEPLOYED])
def test_a_malformed_api_url_fails_loudly_in_every_environment(env, url):
    with pytest.raises(ApiConfigError):
        resolve_api_base_url({**env, "DATA_ANALYZER_API_URL": url})


def test_a_malformed_url_makes_the_startup_check_fail():
    env = {**DEPLOYED, "DATA_ANALYZER_API_URL": "ftp://x.example.invalid"}
    assert ui_config_problems(env)
