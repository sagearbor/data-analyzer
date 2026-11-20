"""Static guarantees about the build, CI and alternative-deploy files (ADP-74/80/81).

Parses files only; nothing is executed or built.
"""
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def _load(rel):
    return yaml.safe_load((ROOT / rel).read_text())


def _run_scripts(workflow):
    for job in workflow["jobs"].values():
        for step in job["steps"]:
            if "run" in step:
                yield step["run"]


def test_cloud_run_workflow_never_interpolates_dispatch_inputs_into_shell():
    for script in _run_scripts(_load(".github/workflows/deploy-cloudrun.yml")):
        assert "${{" not in script


def test_cloud_run_paths_do_not_allow_unauthenticated_access():
    for rel in ("deploy/cloudrun.sh", ".github/workflows/deploy-cloudrun.yml"):
        text = "\n".join(l for l in (ROOT / rel).read_text().splitlines()
                         if not l.lstrip().startswith("#"))  # comments may name the flag
        flags = re.findall(r"--(?:no-)?allow-unauthenticated", text)
        assert flags and all(f == "--no-allow-unauthenticated" for f in flags), rel


def test_cloud_run_paths_set_app_env_and_bind_the_api_key_secret():
    for rel in ("deploy/cloudrun.sh", ".github/workflows/deploy-cloudrun.yml"):
        text = (ROOT / rel).read_text()
        assert "APP_ENV=prod" in text and "DATA_ANALYZER_API_KEY=" in text, rel


def test_cloud_run_script_refuses_to_deploy_without_an_api_key_secret_binding():
    assert ': "${API_KEY_SECRET:?' in (ROOT / "deploy/cloudrun.sh").read_text()


def test_cloud_run_files_carry_a_not_used_for_the_release_banner():
    for rel in ("deploy/cloudrun.sh", ".github/workflows/deploy-cloudrun.yml"):
        assert "NOT USED FOR THE RELEASE" in (ROOT / rel).read_text()[:900], rel


def test_ci_workflow_runs_tests_and_has_a_separate_hygiene_job():
    wf = _load(".github/workflows/ci.yml")
    triggers = wf.get("on", wf.get(True))
    assert "pull_request" in triggers and "push" in triggers
    assert set(wf["jobs"]) == {"tests", "repo-hygiene"}


def test_gitignore_blocks_sidecar_and_key_files_but_not_env_example():
    lines = (ROOT / ".gitignore").read_text().splitlines()
    for pattern in ("*.pem", "*.key", ".env", ".env.*", "!.env.example"):
        assert pattern in lines
    assert any(l.endswith(":sec.endpointdlp") for l in lines)
    assert any(l.endswith(":Zone.Identifier") for l in lines)


def test_ui_image_runtime_stage_has_no_compilers_git_or_ssh_client():
    text = (ROOT / "Dockerfile").read_text()
    runtime = text.split("AS builder", 1)[1].split("\nFROM ", 1)[1]
    for tool in ("build-essential", "gcc", "git", "openssh-client"):
        assert not re.search(rf"\b{tool}\b", runtime.split("COPY --from=builder")[0]), tool
    assert "USER appuser" in runtime


def test_api_image_does_not_install_the_ui_requirements():
    text = (ROOT / "Dockerfile.api").read_text()
    installs = re.findall(r"-r (\S+)", text)
    assert "requirements.txt" not in installs
    assert {"api_requirements.txt", "mcp_requirements.txt"} <= set(installs)


def test_api_requirements_exclude_ui_only_packages():
    names = {re.split(r"[<>=\[ ]", l, 1)[0].lower()
             for l in (ROOT / "api_requirements.txt").read_text().splitlines()
             if l.strip() and not l.startswith("#")}
    assert not names & {"streamlit", "plotly", "matplotlib", "pypdf", "env-banner"}
    assert {"fastapi", "slowapi", "openai", "tiktoken", "azure-identity", "requests"} <= names


def test_pip_trusted_hosts_are_a_build_argument_defaulting_to_current_behaviour():
    for rel in ("Dockerfile", "Dockerfile.api"):
        text = (ROOT / rel).read_text()
        assert 'ARG PIP_TRUSTED_HOSTS="--trusted-host pypi.org --trusted-host files.pythonhosted.org"' in text, rel
        assert "${PIP_TRUSTED_HOSTS}" in text, rel
