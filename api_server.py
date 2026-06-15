"""
api_server.py — FastAPI REST + A2A gateway for the data-analyzer service.

Endpoints
---------
GET  /health                  Liveness probe; reports A2A backend in use.
POST /analyze                 Full quality-pipeline analysis.
POST /data-info               Lightweight dataset info (shape, columns, sample).
GET  /.well-known/agent.json  A2A AgentCard (JSON).
POST /a2a/{agent_id}          A2A message/send endpoint.

Run
---
    python api_server.py            # listens on 0.0.0.0:8003
    API_PORT=9000 python api_server.py

Environment
-----------
API_PORT   int  Port to listen on (default 8003).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

import a2a_agent
import analysis_service
from a2a_runtime import (
    A2A_BACKEND,
    IS_A2A_SHIM,
    WELL_KNOWN_PATH,
    decode_message,
    default_registry,
    encode_message,
    serve_agent_card,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Data Analyzer API",
    description=(
        "Data-quality analysis service exposed over REST, MCP (stdio), and A2A. "
        "Core logic lives in analysis_service.py; all three transports delegate there."
    ),
    version="1.0.0",
)


@app.on_event("startup")
async def _startup() -> None:
    """Register the A2A agent so the /a2a route can dispatch locally."""
    a2a_agent.register()
    logger.info(
        "data-analyzer API started — A2A backend: %s (shim=%s)", A2A_BACKEND, IS_A2A_SHIM
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Liveness probe.

    Returns a small JSON dict that reports whether the real ``dcri-a2a-core``
    package is installed or the bundled shim is active.
    """
    return {
        "status": "ok",
        "service": "data-analyzer",
        "a2a_backend": A2A_BACKEND,
        "a2a_shim": IS_A2A_SHIM,
    }


# ---------------------------------------------------------------------------
# Analysis endpoints
# ---------------------------------------------------------------------------

@app.post("/analyze")
async def analyze(request: Request) -> JSONResponse:
    """Run the full data-quality pipeline.

    Request body (JSON):
        data_content  str   Required. Raw CSV text, base64, or data-URL.
        file_format   str   Optional (default "csv").
        schema        dict  Optional column-type map.
        rules         dict  Optional validation-rule map.
        min_rows      int   Optional (default 1).
        encoding      str   Optional (default "utf-8").

    Returns:
        Quality pipeline result dict, or ``{"error": "...", "type": "..."}``
        if the input is invalid (always HTTP 200 — errors are in the payload
        to match the MCP tool handler behaviour).
    """
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")

    result = analysis_service.analyze(body)
    return JSONResponse(content=result)


@app.post("/data-info")
async def data_info(request: Request) -> JSONResponse:
    """Return lightweight dataset info (shape, columns, dtypes, sample rows).

    Request body (JSON):
        data_content  str  Required.
        file_format   str  Optional (default "csv").
        sample_rows   int  Optional (default 5).
        encoding      str  Optional (default "utf-8").

    Returns:
        Info dict with keys: format, shape, columns, dtypes, sample_data,
        missing_values, duplicate_rows.
    """
    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")

    result = analysis_service.get_info(body)
    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# AgentCard
# ---------------------------------------------------------------------------

@app.get(WELL_KNOWN_PATH)
async def agent_card() -> Response:
    """Serve the A2A AgentCard at the well-known path.

    Returns the canonical ``/.well-known/agent.json`` document describing this
    agent's capabilities, supported methods, and A2A endpoint URL.
    Clients (e.g. ``dcri-ct-graph`` pipeline nodes) discover this agent by
    fetching this document and pointing their ``a2a_endpoint`` at the ``url``
    field.
    """
    card = a2a_agent.build_card(internal_only=False)
    body, headers, content_type = serve_agent_card(card)
    return Response(content=body, media_type=content_type, headers=headers)


# ---------------------------------------------------------------------------
# A2A message/send endpoint
# ---------------------------------------------------------------------------

@app.post("/a2a/{agent_id}")
async def a2a_message_send(agent_id: str, request: Request) -> JSONResponse:
    """Handle an A2A ``message/send`` call for *agent_id*.

    The request body must be a JSON-serialised A2A Message envelope as produced
    by ``encode_message()``. The server decodes it, dispatches to the registered
    agent, and returns the result wrapped in a new envelope.

    Path parameters:
        agent_id  The target agent identifier. Currently only "data-analyzer"
                  is registered; any other value returns HTTP 404.

    Returns:
        Encoded A2A Message envelope containing the agent's result dict.
    """
    if agent_id != "data-analyzer":
        raise HTTPException(
            status_code=404,
            detail=f"Agent {agent_id!r} not found on this server.",
        )

    try:
        body: dict = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be a valid A2A Message JSON")

    # Decode the incoming A2A envelope.
    try:
        payload, meta = decode_message(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode A2A message: {exc}")

    # Retrieve the locally-registered agent and run it.
    agent = default_registry.get(agent_id)
    if agent is None:
        # Shouldn't happen after startup, but be defensive.
        raise HTTPException(
            status_code=503,
            detail=f"Agent {agent_id!r} is registered but not yet available.",
        )

    try:
        result: dict = await agent.run(payload)
    except Exception as exc:
        logger.error("A2A agent.run error: %s", exc, exc_info=True)
        result = {"error": str(exc), "type": "agent_error"}

    # Wrap the result in an A2A response envelope.
    response_message = encode_message(
        result,
        correlation_id=meta.get("correlation_id", ""),
        cycle=meta.get("cycle", 1),
        agent_id=agent_id,
    )
    return JSONResponse(content=response_message)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("API_PORT", "8003"))
    uvicorn.run(app, host="0.0.0.0", port=port)
