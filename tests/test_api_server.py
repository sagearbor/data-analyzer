"""
tests/test_api_server.py — Tests for api_server.py FastAPI endpoints.

All tests use pytest.importorskip("fastapi") so the file degrades gracefully
when FastAPI is not installed (offline / minimal environments).

Tests that do NOT require pandas:
  - GET /health
  - GET /.well-known/agent.json

Tests that require pandas are gated with a further importorskip.

Run: ``pytest tests/test_api_server.py``
"""

from __future__ import annotations

import json
import os
import sys

import pytest

# Ensure repo root on path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Gate the entire module on fastapi being present.
fastapi = pytest.importorskip("fastapi", reason="fastapi not installed — skipping API tests")

from fastapi.testclient import TestClient  # noqa: E402 (after importorskip)

import api_server  # noqa: E402


@pytest.fixture(scope="module")
def client():
    """Return a TestClient wrapping the FastAPI app.

    We trigger the startup event so the A2A agent is registered before tests run.
    """
    with TestClient(api_server.app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

def test_health_status_ok(client):
    """GET /health returns HTTP 200 with status='ok'."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_health_service_name(client):
    """GET /health reports service='data-analyzer'."""
    data = client.get("/health").json()
    assert data["service"] == "data-analyzer"


def test_health_a2a_backend_present(client):
    """GET /health includes the a2a_backend field."""
    data = client.get("/health").json()
    assert "a2a_backend" in data
    assert isinstance(data["a2a_backend"], str)


def test_health_a2a_shim_is_bool(client):
    """GET /health a2a_shim field is a boolean."""
    data = client.get("/health").json()
    assert "a2a_shim" in data
    assert isinstance(data["a2a_shim"], bool)


# ---------------------------------------------------------------------------
# /.well-known/agent.json
# ---------------------------------------------------------------------------

def test_agent_card_status(client):
    """GET /.well-known/agent.json returns HTTP 200."""
    response = client.get("/.well-known/agent.json")
    assert response.status_code == 200


def test_agent_card_content_type(client):
    """AgentCard response has application/json content type."""
    response = client.get("/.well-known/agent.json")
    assert "application/json" in response.headers.get("content-type", "")


def test_agent_card_url_field(client):
    """AgentCard url field points at /a2a/data-analyzer."""
    data = client.get("/.well-known/agent.json").json()
    assert data["url"] == "/a2a/data-analyzer"


def test_agent_card_required_fields(client):
    """AgentCard contains all required top-level fields."""
    data = client.get("/.well-known/agent.json").json()
    for field in ("name", "description", "version", "url", "capabilities", "methods"):
        assert field in data, f"AgentCard missing field {field!r}"


def test_agent_card_not_internal_only(client):
    """AgentCard served from the public endpoint has internal_only=False."""
    data = client.get("/.well-known/agent.json").json()
    dcri = data.get("x-dcri", {})
    assert dcri.get("internal_only") is False


# ---------------------------------------------------------------------------
# /analyze and /data-info — require pandas
# ---------------------------------------------------------------------------

pandas = pytest.importorskip(
    "pandas",
    reason="pandas not installed — skipping analysis endpoint tests",
)

SAMPLE_CSV = "name,age,score\nAlice,30,95.5\nBob,25,88.0\nCarol,35,72.0\n"


def test_analyze_endpoint_returns_200(client):
    """POST /analyze with valid CSV returns HTTP 200."""
    response = client.post(
        "/analyze",
        json={"data_content": SAMPLE_CSV, "file_format": "csv"},
    )
    assert response.status_code == 200


def test_analyze_endpoint_returns_dict(client):
    """POST /analyze result body is a JSON object."""
    response = client.post(
        "/analyze",
        json={"data_content": SAMPLE_CSV, "file_format": "csv"},
    )
    data = response.json()
    assert isinstance(data, dict)


def test_analyze_missing_data_content(client):
    """POST /analyze with no data_content returns a JSON error dict, not 5xx."""
    response = client.post("/analyze", json={})
    assert response.status_code == 200     # service returns error in body
    data = response.json()
    assert "error" in data


def test_data_info_endpoint_shape(client):
    """POST /data-info returns shape, columns, and sample_data."""
    response = client.post(
        "/data-info",
        json={"data_content": SAMPLE_CSV, "file_format": "csv", "sample_rows": 2},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["shape"]["rows"] == 3
    assert data["shape"]["columns"] == 3
    assert len(data["sample_data"]) == 2


def test_data_info_bad_json_returns_400(client):
    """POST /data-info with non-JSON body returns HTTP 400."""
    response = client.post(
        "/data-info",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# /a2a/{agent_id}
# ---------------------------------------------------------------------------

from a2a_runtime import encode_message, decode_message  # noqa: E402


def test_a2a_endpoint_unknown_agent(client):
    """POST /a2a/no-such-agent returns HTTP 404."""
    msg = encode_message({}, correlation_id="c", cycle=1, agent_id="no-such-agent")
    response = client.post("/a2a/no-such-agent", json=msg)
    assert response.status_code == 404


def test_a2a_endpoint_bad_json_returns_400(client):
    """POST /a2a/data-analyzer with non-JSON body returns HTTP 400."""
    response = client.post(
        "/a2a/data-analyzer",
        content=b"not-json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400


def test_a2a_endpoint_unknown_operation(client):
    """POST /a2a/data-analyzer with unknown operation returns HTTP 200 error dict."""
    msg = encode_message(
        {"operation": "fly-to-moon"},
        correlation_id="corr-a2a-01",
        cycle=1,
        agent_id="data-analyzer",
    )
    response = client.post("/a2a/data-analyzer", json=msg)
    assert response.status_code == 200
    envelope = response.json()
    # The response is itself an A2A message; decode it.
    payload, meta = decode_message(envelope)
    assert "error" in payload
    assert payload.get("type") == "unknown_operation"


def test_a2a_endpoint_correlation_id_preserved(client):
    """POST /a2a/data-analyzer echoes back the correlation_id in the response envelope."""
    msg = encode_message(
        {"operation": "fly-to-moon"},
        correlation_id="my-unique-id-xyz",
        cycle=2,
        agent_id="data-analyzer",
    )
    response = client.post("/a2a/data-analyzer", json=msg)
    assert response.status_code == 200
    envelope = response.json()
    _, meta = decode_message(envelope)
    assert meta["correlation_id"] == "my-unique-id-xyz"
