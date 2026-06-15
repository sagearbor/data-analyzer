"""
tests/test_a2a_runtime.py — STDLIB-ONLY tests for the a2a_runtime package.

These tests MUST pass under plain ``python3`` with zero pip-installed packages.
They exercise:
  - encode_message / decode_message round-trip + cycle int-coercion
  - AgentRegistry: register, get, clear
  - A2AClient in-process dispatch (short-circuit, no network)
  - A2AClient remote LookupError when no remote_sender is configured
  - build_agent_card / serve_agent_card output shape
  - AGENT_PATH_TEMPLATE / WELL_KNOWN_PATH constants

No pandas, no fastapi, no mcp — only a2a_runtime (stdlib shim or real package).

Run with pytest: ``pytest tests/test_a2a_runtime.py``
Run standalone:  ``python3 tests/test_a2a_runtime.py``
"""

from __future__ import annotations

import asyncio
import json
import sys
import os

# Ensure repo root is on sys.path when run as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from a2a_runtime import (
    A2A_BACKEND,
    A2A_IMPLEMENTED_METHODS,
    A2A_MINIMAL_CAPABILITIES,
    AGENT_PATH_TEMPLATE,
    WELL_KNOWN_PATH,
    AgentRegistry,
    A2AClient,
    IS_A2A_SHIM,
    build_agent_card,
    decode_message,
    default_registry,
    encode_message,
    serve_agent_card,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine in a new event loop (works without pytest-asyncio)."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# encode_message / decode_message
# ---------------------------------------------------------------------------

def test_encode_decode_round_trip():
    """Payload survives encode -> decode without data loss."""
    payload = {"data_content": "col_a,col_b\n1,2", "file_format": "csv"}
    msg = encode_message(payload, correlation_id="corr-001", cycle=1, agent_id="data-analyzer")

    assert isinstance(msg, dict), "encode_message must return a dict"
    assert "payload" in msg
    assert "metadata" in msg

    recovered_payload, meta = decode_message(msg)
    assert recovered_payload == payload
    assert meta["correlation_id"] == "corr-001"
    assert meta["cycle"] == 1
    assert meta["agent_id"] == "data-analyzer"


def test_cycle_is_coerced_to_int():
    """decode_message must return ``cycle`` as int, even if stored as float."""
    payload = {"x": 1}
    msg = encode_message(payload, correlation_id="corr-002", cycle=3)
    # Manually corrupt cycle to float to simulate protobuf / JSON round-trip.
    msg["metadata"]["cycle"] = 3.0

    _, meta = decode_message(msg)
    assert isinstance(meta["cycle"], int), "cycle must be int after decode"
    assert meta["cycle"] == 3


def test_encode_empty_payload():
    """encode_message handles empty dict without raising."""
    msg = encode_message({}, correlation_id="corr-003", cycle=1)
    payload, meta = decode_message(msg)
    assert payload == {}
    assert meta["correlation_id"] == "corr-003"


def test_decode_missing_metadata_fields():
    """decode_message returns sensible defaults when metadata keys are absent."""
    bare_msg = {"payload": {"a": 1}, "metadata": {}}
    payload, meta = decode_message(bare_msg)
    assert payload == {"a": 1}
    assert meta["correlation_id"] is None
    assert meta["cycle"] == 1   # default


def test_encode_non_dict_payload_raises():
    """encode_message raises TypeError for non-dict, non-model-dump payloads."""
    try:
        encode_message("not-a-dict", correlation_id="c", cycle=1)  # type: ignore[arg-type]
        assert False, "Expected TypeError"
    except TypeError:
        pass


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------

class _DummyAgent:
    agent_id = "dummy"

    async def run(self, payload: dict) -> dict:
        return {"echo": payload}


def test_registry_register_and_get():
    """register + get returns the same agent instance."""
    reg = AgentRegistry()
    agent = _DummyAgent()
    reg.register(agent)
    assert reg.get("dummy") is agent


def test_registry_get_missing_returns_none():
    """get on an absent agent_id returns None."""
    reg = AgentRegistry()
    assert reg.get("no-such-agent") is None


def test_registry_clear():
    """clear removes all registered agents."""
    reg = AgentRegistry()
    reg.register(_DummyAgent())
    reg.clear()
    assert reg.get("dummy") is None


def test_registry_overwrite_is_idempotent():
    """Registering the same agent_id twice replaces the old entry."""
    reg = AgentRegistry()
    a1 = _DummyAgent()
    a2 = _DummyAgent()
    reg.register(a1)
    reg.register(a2)
    assert reg.get("dummy") is a2


# ---------------------------------------------------------------------------
# A2AClient — in-process dispatch
# ---------------------------------------------------------------------------

def test_a2a_client_in_process_call():
    """A2AClient.call dispatches to a locally registered agent."""
    reg = AgentRegistry()
    reg.register(_DummyAgent())
    client = A2AClient(registry=reg)

    result = _run(
        client.call("dummy", {"hello": "world"}, correlation_id="corr-010", cycle=1)
    )
    assert result == {"echo": {"hello": "world"}}


def test_a2a_client_remote_lookup_error():
    """A2AClient.call raises LookupError for unknown agent with no remote_sender."""
    reg = AgentRegistry()
    client = A2AClient(registry=reg)

    try:
        _run(client.call("unknown-agent", {}, correlation_id="c", cycle=1))
        assert False, "Expected LookupError"
    except LookupError:
        pass


# ---------------------------------------------------------------------------
# AgentCard
# ---------------------------------------------------------------------------

def test_build_agent_card_url():
    """build_agent_card produces a url matching AGENT_PATH_TEMPLATE."""
    card = build_agent_card(
        "data-analyzer",
        "Data Analyzer",
        "Test agent",
        version="1.0.0",
        internal_only=False,
    )
    expected_url = AGENT_PATH_TEMPLATE.format(agent_id="data-analyzer")
    assert card["url"] == expected_url, f"Expected {expected_url!r}, got {card['url']!r}"


def test_build_agent_card_shape():
    """build_agent_card returns all required top-level keys."""
    card = build_agent_card(
        "data-analyzer",
        "Data Analyzer",
        "Description",
        version="1.0.0",
    )
    for key in ("name", "description", "version", "url", "capabilities", "methods"):
        assert key in card, f"Missing key {key!r} in AgentCard"


def test_serve_agent_card_returns_json():
    """serve_agent_card output is valid JSON and content-type is application/json."""
    card = build_agent_card("data-analyzer", "DA", "desc", version="1.0.0")
    body, headers, content_type = serve_agent_card(card)
    assert content_type == "application/json"
    parsed = json.loads(body)   # must not raise
    assert parsed["url"] == AGENT_PATH_TEMPLATE.format(agent_id="data-analyzer")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_well_known_path_starts_with_slash():
    assert WELL_KNOWN_PATH.startswith("/"), "WELL_KNOWN_PATH must be an absolute path"


def test_agent_path_template_contains_placeholder():
    assert "{agent_id}" in AGENT_PATH_TEMPLATE


def test_a2a_backend_is_string():
    assert isinstance(A2A_BACKEND, str) and A2A_BACKEND, "A2A_BACKEND must be a non-empty string"


def test_is_a2a_shim_is_bool():
    assert isinstance(IS_A2A_SHIM, bool)


# ---------------------------------------------------------------------------
# Standalone runner (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for fn in _tests:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {fn.__name__}: {exc}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {passed + failed} tests.")
    sys.exit(0 if failed == 0 else 1)
