"""
tests/test_a2a_agent.py — Tests for the DataAnalyzerAgent A2A wrapper.

Tests that need real analysis (pandas + core classes) are gated with
``pytest.importorskip("pandas")``.  The unknown-operation error path is
tested without pandas because it short-circuits before any data loading.

Run: ``pytest tests/test_a2a_agent.py``
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest

# Ensure repo root on path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from a2a_runtime import AgentRegistry, default_registry


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def test_register_populates_default_registry():
    """register() places a DataAnalyzerAgent into default_registry."""
    import a2a_agent

    # Clear any prior registration to make this test independent.
    default_registry.clear()
    a2a_agent.register()

    agent = default_registry.get("data-analyzer")
    assert agent is not None, "DataAnalyzerAgent should be registered"
    assert agent.agent_id == "data-analyzer"


def test_register_is_idempotent():
    """Calling register() twice does not raise and leaves a valid agent."""
    import a2a_agent

    default_registry.clear()
    a2a_agent.register()
    a2a_agent.register()  # second call — must not raise

    assert default_registry.get("data-analyzer") is not None


def test_register_into_custom_registry():
    """register(registry=...) targets a user-supplied registry, not the default."""
    import a2a_agent

    custom_reg = AgentRegistry()
    default_registry.clear()
    a2a_agent.register(registry=custom_reg)

    assert custom_reg.get("data-analyzer") is not None
    # Must NOT have registered into the default registry.
    assert default_registry.get("data-analyzer") is None


# ---------------------------------------------------------------------------
# build_card
# ---------------------------------------------------------------------------

def test_build_card_url_matches_agent_id():
    """build_card() returns a card whose url contains the agent id."""
    import a2a_agent

    card = a2a_agent.build_card(internal_only=False)
    assert "data-analyzer" in card["url"]


def test_build_card_internal_only_default_is_false():
    """build_card() with no arguments defaults to internal_only=False."""
    import a2a_agent

    card = a2a_agent.build_card()
    # The x-dcri extension carries the flag.
    assert card.get("x-dcri", {}).get("internal_only") is False


def test_build_card_can_be_internal():
    """build_card(internal_only=True) sets the flag in x-dcri."""
    import a2a_agent

    card = a2a_agent.build_card(internal_only=True)
    assert card.get("x-dcri", {}).get("internal_only") is True


# ---------------------------------------------------------------------------
# Unknown-operation error (no pandas needed)
# ---------------------------------------------------------------------------

def test_unknown_operation_returns_error_dict():
    """DataAnalyzerAgent.run returns an error dict for unknown operations."""
    import a2a_agent

    agent = a2a_agent.DataAnalyzerAgent()
    result = _run(agent.run({"operation": "fly-to-the-moon"}))

    assert "error" in result, "Expected an 'error' key in result"
    assert result.get("type") == "unknown_operation"


def test_missing_operation_defaults_to_analyze():
    """When 'operation' key is absent, the agent defaults to 'analyze'.

    Since pandas is not available in offline mode the call will return an
    error dict — but it must NOT return 'unknown_operation'; it must attempt
    the analyze path (which fails for a different reason: missing deps).
    """
    import a2a_agent

    agent = a2a_agent.DataAnalyzerAgent()
    # No data_content either — analysis_service.analyze returns a validation error.
    result = _run(agent.run({}))

    assert result.get("type") != "unknown_operation", (
        "Missing 'operation' should default to 'analyze', not 'unknown_operation'"
    )


# ---------------------------------------------------------------------------
# Full analysis path (requires pandas)
# ---------------------------------------------------------------------------

pandas = pytest.importorskip("pandas", reason="pandas not installed — skipping analysis tests")


SAMPLE_CSV = "name,age,score\nAlice,30,95.5\nBob,25,88.0\nCarol,35,72.0\n"


def test_agent_run_analyze():
    """DataAnalyzerAgent.run with operation='analyze' returns quality results."""
    import a2a_agent

    agent = a2a_agent.DataAnalyzerAgent()
    payload = {
        "operation": "analyze",
        "data_content": SAMPLE_CSV,
        "file_format": "csv",
    }
    result = _run(agent.run(payload))

    assert isinstance(result, dict)
    # The quality pipeline result should not be a plain error.
    assert "error" not in result or result.get("type") != "analysis_error", (
        f"Got unexpected error: {result}"
    )


def test_agent_run_get_info():
    """DataAnalyzerAgent.run with operation='get_info' returns shape/column info."""
    import a2a_agent

    agent = a2a_agent.DataAnalyzerAgent()
    payload = {
        "operation": "get_info",
        "data_content": SAMPLE_CSV,
        "file_format": "csv",
    }
    result = _run(agent.run(payload))

    assert isinstance(result, dict)
    assert "shape" in result, f"Expected 'shape' key; got: {result}"
    assert result["shape"]["rows"] == 3
    assert result["shape"]["columns"] == 3


def test_agent_run_data_info_alias():
    """'data_info' is an accepted alias for 'get_info'."""
    import a2a_agent

    agent = a2a_agent.DataAnalyzerAgent()
    payload = {
        "operation": "data_info",
        "data_content": SAMPLE_CSV,
        "file_format": "csv",
    }
    result = _run(agent.run(payload))

    assert "shape" in result, f"Expected shape in result; got: {result}"


def test_agent_run_missing_data_content():
    """analyze with no data_content returns a validation error dict, not a raise."""
    import a2a_agent

    agent = a2a_agent.DataAnalyzerAgent()
    result = _run(agent.run({"operation": "analyze"}))

    assert "error" in result
    # Must not propagate as an unhandled exception.
