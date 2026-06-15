"""A2A runtime: prefer the real ``dcri-a2a-core``, fall back to a local shim.

Every adapter in this repo (REST / MCP / A2A) imports its A2A primitives from
``a2a_runtime`` rather than importing ``dcri_a2a_core`` directly. That single
indirection means:

  * When ``dcri-a2a-core`` is installed (the org's published foundation
    package), the real, versioned wire contract is used — no code change.
  * When it is not yet installed (offline sandbox, fresh clone), the pure-stdlib
    ``_shim`` provides the identical public names so the service still runs.

Check ``A2A_BACKEND`` / ``IS_A2A_SHIM`` at runtime to see which is active; the
``/health`` and AgentCard surfaces report it so operators know whether they are
on the real wire contract.

To go to the real contract: ``pip install dcri-a2a-core`` (and optionally delete
``a2a_runtime/_shim.py``); nothing else changes because the names match.
"""

from __future__ import annotations

try:  # Prefer the real, versioned foundation package.
    from dcri_a2a_core import (  # type: ignore
        A2A_IMPLEMENTED_METHODS,
        A2A_MINIMAL_CAPABILITIES,
        AGENT_PATH_TEMPLATE,
        PAYLOAD_MEDIA_TYPE,
        WELL_KNOWN_PATH,
        A2AClient,
        AgentRegistry,
        RunnableAgent,
        build_agent_card,
        correlation_id_of,
        cycle_of,
        decode_message,
        default_registry,
        encode_message,
        serve_agent_card,
    )

    IS_A2A_SHIM = False
    A2A_BACKEND = "dcri-a2a-core"
except ImportError:  # Fall back to the bundled stdlib shim.
    from ._shim import (  # type: ignore
        A2A_IMPLEMENTED_METHODS,
        A2A_MINIMAL_CAPABILITIES,
        AGENT_PATH_TEMPLATE,
        PAYLOAD_MEDIA_TYPE,
        WELL_KNOWN_PATH,
        A2AClient,
        AgentRegistry,
        RunnableAgent,
        build_agent_card,
        correlation_id_of,
        cycle_of,
        decode_message,
        default_registry,
        encode_message,
        serve_agent_card,
    )

    IS_A2A_SHIM = True
    A2A_BACKEND = "shim"

__all__ = [
    "IS_A2A_SHIM",
    "A2A_BACKEND",
    "A2A_IMPLEMENTED_METHODS",
    "A2A_MINIMAL_CAPABILITIES",
    "AGENT_PATH_TEMPLATE",
    "PAYLOAD_MEDIA_TYPE",
    "WELL_KNOWN_PATH",
    "A2AClient",
    "AgentRegistry",
    "RunnableAgent",
    "build_agent_card",
    "correlation_id_of",
    "cycle_of",
    "decode_message",
    "default_registry",
    "encode_message",
    "serve_agent_card",
]
