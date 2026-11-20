"""Documentation must match the released service (ADP-87): no advertised A2A routes, no bare az update --yaml."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_readme_does_not_advertise_a2a_routes_as_available():
    text = (ROOT / "README.md").read_text()
    assert "POST /a2a/data-analyzer" not in text
    assert "not copied into the released" in text


def test_integration_guide_has_no_public_unauthenticated_deployment_instructions():
    text = (ROOT / "docs/INTEGRATION.md").read_text()
    assert "curl -X POST http://<host>:8003/a2a" not in text
    assert "must **never be exposed to a network without authentication" in text.replace("\n", " ")


def test_aca_readme_does_not_instruct_a_bare_update_yaml_and_points_to_the_guard():
    text = (ROOT / "deploy/aca/README.md").read_text()
    assert 'az containerapp update --name' not in text
    assert 'az containerapp create --name' not in text
    assert "aca-guard.sh" in text and "~/" not in text.split("## 3. Deploy")[1].split("##")[0]


def test_docs_record_that_openapi_pages_are_open_by_design():
    assert "open without an API key by design" in (ROOT / "README.md").read_text()
    assert "open\nwithout an API key by design" in (ROOT / "docs/INTEGRATION.md").read_text()
