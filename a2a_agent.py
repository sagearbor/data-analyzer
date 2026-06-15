"""
a2a_agent.py — A2A-protocol agent wrapper for the data-analyzer service.

This module is importable under plain python3 with zero heavy dependencies.
All heavy work (analysis_service, pandas) is imported lazily inside ``run()``.

Usage
-----
    from a2a_agent import DataAnalyzerAgent, build_card, register

    # Register into the default in-process registry (idempotent)
    register()

    # Build the AgentCard for /.well-known/agent.json
    card = build_card(internal_only=False)

    # Use the agent directly (async)
    agent = DataAnalyzerAgent()
    result = await agent.run({"operation": "analyze", "data_content": "..."})
"""

from __future__ import annotations

import logging
from typing import Optional

from a2a_runtime import (
    AgentRegistry,
    build_agent_card,
    default_registry,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_AGENT_ID = "data-analyzer"
_AGENT_NAME = "Data Analyzer"
_AGENT_DESCRIPTION = (
    "Runs data-quality checks (type validation, range checks, missing-value "
    "analysis, duplicate detection) on CSV/JSON/Excel/Parquet data. "
    "Accepts base64 or plain-text data_content. "
    "Dispatch via operation='analyze' (full pipeline) or operation='get_info' "
    "(lightweight shape/column/sample info)."
)
_AGENT_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class DataAnalyzerAgent:
    """A2A-protocol runnable agent that delegates to analysis_service.

    Implements the ``RunnableAgent`` structural protocol required by
    ``AgentRegistry``: exposes ``agent_id`` (str attribute) and
    ``run(payload)`` (async method returning dict).
    """

    agent_id: str = _AGENT_ID

    async def run(self, payload: dict) -> dict:  # noqa: D102
        """Dispatch the payload to the appropriate analysis_service function.

        Parameters
        ----------
        payload:
            A plain dict decoded from the A2A envelope. Expected keys:
            - operation (str, optional): "analyze" (default) | "get_info" | "data_info"
            - data_content, file_format, schema, rules, min_rows, encoding,
              sample_rows — forwarded verbatim to analysis_service.

        Returns
        -------
        dict
            JSON-serialisable result from analysis_service, or an error dict.
        """
        # Lazy import so this module loads with stdlib only.
        import analysis_service  # noqa: PLC0415

        operation: str = payload.get("operation", "analyze").lower()

        logger.debug("DataAnalyzerAgent.run: operation=%r", operation)

        if operation == "analyze":
            return analysis_service.analyze(payload)

        if operation in ("get_info", "data_info"):
            return analysis_service.get_info(payload)

        # Unknown operation — return a consistent error dict (never raise).
        logger.warning("DataAnalyzerAgent: unknown operation %r", operation)
        return {
            "error": (
                f"Unknown operation {operation!r}. "
                "Supported values: 'analyze', 'get_info'."
            ),
            "type": "unknown_operation",
        }


# ---------------------------------------------------------------------------
# AgentCard helper
# ---------------------------------------------------------------------------

def build_card(internal_only: bool = False) -> dict:
    """Build the AgentCard dict for /.well-known/agent.json.

    Parameters
    ----------
    internal_only:
        Pass ``False`` when serving publicly (e.g. from ``api_server.py``).
        The default is ``False`` to match the typical external-facing use-case;
        callers that want the card to be marked internal-only must opt-in.

    Returns
    -------
    dict
        AgentCard dict ready to serialise via ``serve_agent_card()``.
    """
    return build_agent_card(
        _AGENT_ID,
        _AGENT_NAME,
        _AGENT_DESCRIPTION,
        version=_AGENT_VERSION,
        internal_only=internal_only,
    )


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

def register(registry: Optional[AgentRegistry] = None) -> None:
    """Register a ``DataAnalyzerAgent`` into a registry (idempotent).

    If ``registry`` is ``None``, registers into ``a2a_runtime.default_registry``.
    Calling this function multiple times is safe — it always overwrites the
    previous entry with a fresh ``DataAnalyzerAgent`` instance.

    Parameters
    ----------
    registry:
        Target ``AgentRegistry``. Defaults to ``default_registry``.
    """
    target: AgentRegistry = registry if registry is not None else default_registry
    agent = DataAnalyzerAgent()
    target.register(agent)
    logger.info("Registered DataAnalyzerAgent (id=%r) into registry %r", _AGENT_ID, target)
