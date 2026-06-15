"""Real over-the-network A2A client — the hop ``dcri-a2a-core`` leaves as a stub.

``dcri_a2a_core.A2AClient._call_remote`` is a documented TODO: the foundation
package can receive A2A calls but cannot yet *make* one over HTTP. This module
implements that missing hop so a caller (e.g. a ``dcri-ct-graph`` node, or any
org service) can actually reach a deployed agent.

It speaks the **canonical DCRI JSON envelope** — the same shape this repo's
``/a2a/{agent_id}`` endpoint accepts — by POSTing the encoded message to
``{base_url}/a2a/{agent_id}`` and decoding the reply. It is **stdlib-only**
(``urllib``) on purpose: no ``requests``/``httpx`` dependency, importable in any
environment, and trivial to run from a CI smoke test.

Usage — direct call::

    from a2a_runtime.http_client import call_agent_http
    result = call_agent_http(
        "https://data-analyzer-xxxx.run.app", "data-analyzer",
        {"operation": "analyze", "data_content": "<base64-csv>"},
        correlation_id="c1",
    )

Usage — as the remote_sender for the uniform ``A2AClient`` (so local and remote
callers use the identical ``client.call(...)`` path)::

    from a2a_runtime import A2AClient
    from a2a_runtime.http_client import make_http_remote_sender
    sender = make_http_remote_sender({"data-analyzer": "https://...run.app"})
    client = A2AClient(remote_sender=sender)
    result = await client.call("data-analyzer", payload, correlation_id="c1")

NOTE on the real package: when ``dcri-a2a-core`` (a2a-sdk based) is installed,
``encode_message`` returns a protobuf ``Message`` rather than a dict. This module
serializes either form to the canonical JSON envelope via ``model_dump()``. The
remaining parity point to verify against the real package is whether its inbound
route expects the a2a-sdk JSON-RPC ``message/send`` framing; see docs/INTEGRATION.md.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Optional, Union

from . import decode_message, encode_message

__all__ = ["call_agent_http", "make_http_remote_sender", "A2AHttpError"]


class A2AHttpError(RuntimeError):
    """Raised when the remote A2A call fails at the transport level."""


def _to_json_envelope(message: Any) -> dict:
    """Coerce an encoded message (shim dict OR a2a-sdk Message) to a JSON dict."""
    if isinstance(message, dict):
        return message
    model_dump = getattr(message, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    raise TypeError(
        f"Cannot serialize encoded A2A message of type {type(message)!r} to JSON."
    )


def _endpoint(base_url: str, agent_id: str) -> str:
    return base_url.rstrip("/") + "/a2a/" + agent_id


def _post_envelope(
    url: str, envelope: dict, *, bearer_token: Optional[str], timeout: float
) -> dict:
    """POST a JSON envelope and return the decoded JSON response body."""
    body = json.dumps(envelope).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:  # 4xx/5xx
        detail = exc.read().decode("utf-8", "replace") if exc.fp else ""
        raise A2AHttpError(f"A2A call to {url} failed: HTTP {exc.code} {detail}") from exc
    except urllib.error.URLError as exc:  # DNS / connection
        raise A2AHttpError(f"A2A call to {url} failed: {exc.reason}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise A2AHttpError(f"A2A reply from {url} was not JSON: {raw[:200]!r}") from exc


def call_agent_http(
    base_url: str,
    agent_id: str,
    payload: dict,
    *,
    correlation_id: str,
    cycle: int = 1,
    bearer_token: Optional[str] = None,
    timeout: float = 60.0,
) -> dict:
    """Call a deployed agent over HTTP and return its response payload dict.

    Encodes ``payload`` into the canonical envelope, POSTs it to
    ``{base_url}/a2a/{agent_id}``, then decodes and returns the reply payload.
    Raises :class:`A2AHttpError` on any transport-level failure.
    """
    request_msg = encode_message(
        payload, correlation_id=correlation_id, cycle=cycle, agent_id=agent_id
    )
    response_envelope = _post_envelope(
        _endpoint(base_url, agent_id),
        _to_json_envelope(request_msg),
        bearer_token=bearer_token,
        timeout=timeout,
    )
    result_payload, _meta = decode_message(response_envelope)
    return result_payload


def make_http_remote_sender(
    resolve_base_url: Union[Mapping[str, str], Callable[[str], str]],
    *,
    bearer_token: Optional[str] = None,
    timeout: float = 60.0,
):
    """Build an async ``remote_sender`` for :class:`a2a_runtime.A2AClient`.

    ``resolve_base_url`` maps an ``agent_id`` to its base URL — either a plain
    ``{agent_id: base_url}`` dict or a callable ``agent_id -> base_url``. The
    returned coroutine matches the ``A2AClient`` remote-seam contract:
    ``async (agent_id, encoded_request) -> encoded_response``.
    """
    if callable(resolve_base_url):
        resolver = resolve_base_url
    else:
        mapping = dict(resolve_base_url)

        def resolver(agent_id: str) -> str:
            try:
                return mapping[agent_id]
            except KeyError as exc:
                raise A2AHttpError(
                    f"No base URL configured for agent {agent_id!r}."
                ) from exc

    async def _sender(agent_id: str, request_message: Any) -> dict:
        url = _endpoint(resolver(agent_id), agent_id)
        envelope = _to_json_envelope(request_message)
        return await asyncio.to_thread(
            _post_envelope, url, envelope, bearer_token=bearer_token, timeout=timeout
        )

    return _sender
