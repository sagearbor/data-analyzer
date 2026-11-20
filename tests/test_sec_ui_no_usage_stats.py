"""The page must not send Streamlit usage statistics to an outside service.

Found on the deployed site on 20 Nov 2026: each user's browser was calling
webhooks.fivetran.com (Streamlit's usage statistics). An internal tool makes
no outside calls, so the setting is off in three places.
"""
import re
import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def test_streamlit_config_file_turns_usage_statistics_off():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text())
    assert config["browser"]["gatherUsageStats"] is False


def test_every_streamlit_start_command_in_the_entrypoint_turns_usage_statistics_off():
    text = (ROOT / "entrypoint.sh").read_text()
    starts = re.findall(r"exec streamlit run web_app\.py(?:[^\n]*\\\n)*[^\n]*", text)
    assert len(starts) >= 2, "expected the plain and the base-path start commands"
    for command in starts:
        assert "--browser.gatherUsageStats=false" in command


def test_page_deployment_template_turns_usage_statistics_off():
    manifest = yaml.safe_load((ROOT / "deploy" / "aca" / "containerapp-ui.yaml").read_text())
    env = {e["name"]: e.get("value") for e in manifest["properties"]["template"]["containers"][0]["env"]}
    assert env.get("STREAMLIT_BROWSER_GATHER_USAGE_STATS") == "false"
