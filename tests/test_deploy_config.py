"""
Deploy-blocking checks: if any of these fail, the image/manifest must not ship.

  * src.runtime_env decides "deployed" the same way for API and UI
  * API refuses to start without DATA_ANALYZER_API_KEY in every non-local env
    (including ACA with APP_ENV forgotten)
  * UI container refuses to start without its API URL/key (entrypoint check)
  * committed ACA templates wire the UI -> API connection and hold no real
    secrets or infra IDs
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src.runtime_env import (LOCAL_API_URL, ApiConfigError, is_deployed_env, main,
                             resolve_api_base_url, ui_config_problems)

ROOT = Path(__file__).resolve().parent.parent
ACA = ROOT / "deploy" / "aca"


# ---------------------------------------------------------------------------
# runtime_env
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("env,deployed", [
    ({}, False),
    ({"APP_ENV": "dev"}, False),
    ({"APP_ENV": "local"}, False),
    ({"APP_ENV": "test"}, False),
    ({"APP_ENV": "development"}, False),
    ({"APP_ENV": "prod"}, True),
    ({"APP_ENV": "staging"}, True),
    ({"APP_ENV": "PROD"}, True),
    ({"APP_ENV": "dev", "CONTAINER_APP_NAME": "data-analyzer-ui"}, True),
    ({"CONTAINER_APP_NAME": "data-analyzer-api"}, True),
])
def test_is_deployed_env(env, deployed):
    assert is_deployed_env(env) is deployed


def test_resolve_api_base_url():
    assert resolve_api_base_url({}) == LOCAL_API_URL
    assert resolve_api_base_url({"APP_ENV": "prod", "DATA_ANALYZER_API_URL": "https://x/"}) == "https://x"
    with pytest.raises(ApiConfigError):
        resolve_api_base_url({"APP_ENV": "prod"})
    with pytest.raises(ApiConfigError):
        resolve_api_base_url({"CONTAINER_APP_NAME": "ui", "DATA_ANALYZER_API_URL": "  "})


def test_ui_config_problems_and_startup_check(monkeypatch):
    assert ui_config_problems({}) == []
    assert len(ui_config_problems({"APP_ENV": "prod"})) == 2
    assert ui_config_problems({"APP_ENV": "prod", "DATA_ANALYZER_API_URL": "https://x",
                               "DATA_ANALYZER_API_KEY": "k"}) == []
    for k in ("CONTAINER_APP_NAME", "DATA_ANALYZER_API_URL", "DATA_ANALYZER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("APP_ENV", "prod")
    assert main(["--check-ui"]) == 1
    monkeypatch.setenv("DATA_ANALYZER_API_URL", "https://x")
    monkeypatch.setenv("DATA_ANALYZER_API_KEY", "k")
    assert main(["--check-ui"]) == 0
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.delenv("DATA_ANALYZER_API_URL")
    assert main(["--check-ui"]) == 0
    assert main([]) == 2


def test_entrypoint_runs_ui_config_check_before_streamlit():
    text = (ROOT / "entrypoint.sh").read_text()
    check = text.index("python -m src.runtime_env --check-ui || exit 1")
    assert check < text.index("exec streamlit")


# ---------------------------------------------------------------------------
# API fail-closed guard (fresh interpreter per case: api_server reads env at import)
# ---------------------------------------------------------------------------

def _import_api_server(extra_env):
    env = {k: v for k, v in os.environ.items()
           if k not in ("APP_ENV", "CONTAINER_APP_NAME", "DATA_ANALYZER_API_KEY")}
    # Explicit empty key: load_dotenv() never overrides an existing var, so a
    # developer's .env can't mask the guard under test.
    env["DATA_ANALYZER_API_KEY"] = ""
    env["ALLOWED_ORIGINS"] = "https://ui.example.test"
    env.update(extra_env)
    return subprocess.run([sys.executable, "-c", "import api_server"], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("extra", [
    {"APP_ENV": "prod"},
    {"APP_ENV": "staging"},
    {"CONTAINER_APP_NAME": "data-analyzer-api"},                    # ACA, APP_ENV forgotten
    {"APP_ENV": "dev", "CONTAINER_APP_NAME": "data-analyzer-api"},  # ACA, APP_ENV wrong
])
def test_api_refuses_to_start_without_key_when_deployed(extra):
    proc = _import_api_server(extra)
    assert proc.returncode != 0
    assert "DATA_ANALYZER_API_KEY is not set" in proc.stderr


def test_api_starts_without_key_locally():
    proc = _import_api_server({"APP_ENV": "test"})
    assert proc.returncode == 0, proc.stderr[-2000:]


# ---------------------------------------------------------------------------
# Committed ACA templates
# ---------------------------------------------------------------------------

def _load(name):
    return yaml.safe_load((ACA / name).read_text())


def _container_env(manifest):
    return {e["name"]: e for e in manifest["properties"]["template"]["containers"][0]["env"]}


def test_ui_template_wires_api_url_and_key():
    m = _load("containerapp-ui.yaml")
    env = _container_env(m)
    assert env["DATA_ANALYZER_API_URL"]["value"].startswith("https://")
    assert env["DATA_ANALYZER_API_KEY"] == {"name": "DATA_ANALYZER_API_KEY",
                                            "secretRef": "data-analyzer-api-key"}
    secrets = {s["name"] for s in m["properties"]["configuration"]["secrets"]}
    assert "data-analyzer-api-key" in secrets
    assert env["APP_ENV"]["value"] == "prod"


def test_ui_and_api_share_the_api_key_secret_name():
    api_env = _container_env(_load("containerapp.yaml"))
    ui_env = _container_env(_load("containerapp-ui.yaml"))
    assert api_env["DATA_ANALYZER_API_KEY"]["secretRef"] == ui_env["DATA_ANALYZER_API_KEY"]["secretRef"]


_REAL_LOOKING = [
    re.compile(r"/subscriptions/[0-9a-f]{8}-", re.I),          # real resource IDs
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),  # GUIDs
    re.compile(r"\b[a-z0-9]+\.azurecr\.io\b", re.I),             # real ACR login server
    re.compile(r"\.azurecontainerapps\.io\b", re.I),             # real app FQDNs
    re.compile(r"\b[a-z0-9]+\.blob\.core\.windows\.net\b", re.I),
]


@pytest.mark.parametrize("name", ["containerapp.yaml", "containerapp-ui.yaml"])
def test_committed_templates_hold_no_real_infra_ids_or_secrets(name):
    text = (ACA / name).read_text()
    for pat in _REAL_LOOKING:
        assert not pat.search(text), f"{name} contains a real-looking value matching {pat.pattern}"
    for s in yaml.safe_load(text)["properties"]["configuration"]["secrets"]:
        assert s.get("value", "").startswith("<"), f"secret {s['name']} has a non-placeholder value"
