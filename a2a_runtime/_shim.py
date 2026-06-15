"""Pure-stdlib fallback that mirrors the ``dcri-a2a-core`` public API.

This module is a **temporary stand-in** for the sibling foundation package
``dcri-a2a-core`` (https://github.com/sagearbor/dcri-a2a-core). That package is
the canonical A2A wire contract every DCRI agent imports, but it depends on
``a2a-sdk`` + ``pydantic`` and is distributed as a path/index dependency that
may not be installed yet (e.g. in an offline sandbox).

``a2a_runtime/__init__.py`` prefers the real package and falls back to this
shim, so this repo's REST / MCP / A2A adapters import the *same names* either
way. The names and call signatures here intentionally match the real package's
``__all__`` so swapping in the real one is a no-op:

    encode_message / decode_message / correlation_id_of / cycle_of
    AgentRegistry / A2AClient / default_registry / RunnableAgent
    build_agent_card / serve_agent_card
    AGENT_PATH_TEMPLATE / WELL_KNOWN_PATH
    A2A_MINIMAL_CAPABILITIES / A2A_IMPLEMENTED_METHODS / PAYLOAD_MEDIA_TYPE

Differences from the real package (all invisible to callers):
  * A "Message" here is a plain ``dict`` envelope, not an a2a-sdk protobuf type.
    There is therefore no protobuf float/Struct coercion to funnel — ``cycle``
    is still re-coerced to ``int`` in ``decode_message`` so the contract holds.
  * No real network transport: a remote target with no ``remote_sender`` raises
    the same ``LookupError`` the real package raises until its HTTP client lands.

DELETE this file once ``dcri-a2a-core`` is installable; nothing else changes.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Optional, Protocol, Tuple, runtime_checkable

# Marker so callers/tests can detect whether the shim or the real package is live.
IS_A2A_SHIM = True

# ---------------------------------------------------------------------------
# Constants (must match dcri_a2a_core.envelope)
# ---------------------------------------------------------------------------
A2A_MINIMAL_CAPABILITIES = {"streaming": False, "pushNotifications": False}
A2A_IMPLEMENTED_METHODS = ("message/send", "tasks/get")
PAYLOAD_MEDIA_TYPE = "application/json"

AGENT_PATH_TEMPLATE = "/a2a/{agent_id}"
WELL_KNOWN_PATH = "/.well-known/agent.json"

# A "payload" is a JSON dict or any object exposing ``.model_dump()`` (pydantic).
Payload = Any
Message = dict


# ---------------------------------------------------------------------------
# Envelope seam
# ---------------------------------------------------------------------------
def _as_dict(payload: Payload) -> dict:
    """Coerce a payload (dict or pydantic-like model) to a plain JSON dict."""
    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    model_dump = getattr(payload, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    raise TypeError(
        f"A2A payload must be a dict or expose .model_dump(); got {type(payload)!r}"
    )


def encode_message(
    payload: Payload,
    *,
    correlation_id: str,
    cycle: int = 1,
    agent_id: Optional[str] = None,
) -> Message:
    """Wrap ``payload`` in the canonical A2A envelope.

    The payload rides as a JSON data part; ``correlation_id`` / ``cycle`` /
    ``agent_id`` ride in the message metadata. JSON round-trip here mirrors the
    serialization the real wire performs.
    """
    data_part = json.loads(json.dumps(_as_dict(payload)))
    return {
        "payload": data_part,
        "metadata": {
            "correlation_id": correlation_id,
            "cycle": int(cycle),
            "agent_id": agent_id,
        },
    }


def decode_message(message: Message) -> Tuple[dict, dict]:
    """Return ``(payload_dict, metadata_dict)`` from an envelope.

    ``cycle`` is re-coerced to ``int`` so callers see a stable contract
    regardless of how the transport stored it (matches the real package, which
    undoes protobuf's float coercion here).
    """
    payload = dict(message.get("payload", {}))
    meta_in = dict(message.get("metadata", {}))
    metadata = {
        "correlation_id": meta_in.get("correlation_id"),
        "cycle": int(meta_in.get("cycle", 1)),
        "agent_id": meta_in.get("agent_id"),
    }
    return payload, metadata


def correlation_id_of(message: Message) -> Optional[str]:
    """Metadata accessor for the correlation id."""
    return message.get("metadata", {}).get("correlation_id")


def cycle_of(message: Message) -> int:
    """Metadata accessor for the cycle (coerced to ``int``)."""
    return int(message.get("metadata", {}).get("cycle", 1))


# ---------------------------------------------------------------------------
# Registry + uniform caller
# ---------------------------------------------------------------------------
@runtime_checkable
class RunnableAgent(Protocol):
    """Structural type the registry can dispatch to: ``agent_id`` + ``run``."""

    agent_id: str

    async def run(self, payload: dict) -> dict: ...


class AgentRegistry:
    """In-process directory of locally-registered agents (id -> agent)."""

    def __init__(self) -> None:
        self._agents: dict[str, RunnableAgent] = {}

    def register(self, agent: RunnableAgent) -> None:
        self._agents[agent.agent_id] = agent

    def get(self, agent_id: str) -> Optional[RunnableAgent]:
        return self._agents.get(agent_id)

    def clear(self) -> None:
        self._agents.clear()


default_registry = AgentRegistry()

RemoteSender = Callable[[str, Any], Awaitable[Any]]


class A2AClient:
    """Uniform inter-agent caller: in-process short-circuit, else remote seam."""

    def __init__(
        self,
        registry: Optional[AgentRegistry] = None,
        *,
        remote_sender: Optional[RemoteSender] = None,
        bearer_token: Optional[str] = None,
    ) -> None:
        self._registry = registry if registry is not None else default_registry
        self._remote_sender = remote_sender
        self._bearer_token = bearer_token

    async def call(
        self,
        agent_id: str,
        payload: Payload,
        *,
        correlation_id: str,
        cycle: int = 1,
    ) -> dict:
        # Encode even locally so the wire-shape is exercised identically.
        request = encode_message(
            payload, correlation_id=correlation_id, cycle=cycle, agent_id=agent_id
        )
        local = self._registry.get(agent_id)
        if local is not None:
            request_payload, _ = decode_message(request)
            return await local.run(request_payload)
        return await self._call_remote(agent_id, request)

    async def _call_remote(self, agent_id: str, request: Any) -> dict:
        if self._remote_sender is None:
            raise LookupError(
                f"Agent {agent_id!r} is not registered locally and no remote "
                "sender is configured. Register the agent for the in-process "
                "short-circuit, or install dcri-a2a-core for the real HTTP hop."
            )
        response = await self._remote_sender(agent_id, request)
        payload, _ = decode_message(response)
        return payload


# ---------------------------------------------------------------------------
# AgentCard helpers
# ---------------------------------------------------------------------------
def build_agent_card(
    agent_id: str,
    name: str,
    description: str,
    *,
    version: str,
    payload_schema_url: Optional[str] = None,
    internal_only: bool = True,
) -> dict:
    """Build the ``/.well-known/agent.json`` AgentCard dict (transport-only)."""
    card: dict[str, Any] = {
        "name": name,
        "description": description,
        "version": version,
        "url": AGENT_PATH_TEMPLATE.format(agent_id=agent_id),
        "capabilities": dict(A2A_MINIMAL_CAPABILITIES),
        "methods": list(A2A_IMPLEMENTED_METHODS),
        "defaultInputModes": [PAYLOAD_MEDIA_TYPE],
        "defaultOutputModes": [PAYLOAD_MEDIA_TYPE],
        "x-dcri": {"agent_id": agent_id, "internal_only": internal_only},
    }
    if payload_schema_url is not None:
        card["x-dcri"]["payload_schema_url"] = payload_schema_url
    return card


def serve_agent_card(card: dict) -> Tuple[str, dict, str]:
    """Render an AgentCard for serving at :data:`WELL_KNOWN_PATH`."""
    body = json.dumps(card, indent=2, sort_keys=True)
    headers = {"Cache-Control": "public, max-age=300"}
    return body, headers, "application/json"


__all__ = [
    "IS_A2A_SHIM",
    "A2A_MINIMAL_CAPABILITIES",
    "A2A_IMPLEMENTED_METHODS",
    "PAYLOAD_MEDIA_TYPE",
    "encode_message",
    "decode_message",
    "correlation_id_of",
    "cycle_of",
    "AgentRegistry",
    "A2AClient",
    "default_registry",
    "RunnableAgent",
    "build_agent_card",
    "serve_agent_card",
    "AGENT_PATH_TEMPLATE",
    "WELL_KNOWN_PATH",
]
