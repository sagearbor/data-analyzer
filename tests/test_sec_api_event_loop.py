"""Security: heavy work must not freeze the API (ADP-73: A-01).

Each test starts a deliberately slow, blocking job through the real ASGI app and
asks /api/v1/health at the same time. If the job ran on the event loop the
health answer would have to wait for it.
"""

import asyncio
import os
import sys
import time
from unittest.mock import MagicMock

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY = "sec-test-key-loop"
SLOW_SECONDS = 1.5
HEALTH_DEADLINE_SECONDS = 0.7


@pytest.fixture
def api(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "API_KEY", KEY)
    return api_server


async def _health_during(app, slow_request):
    """Run `slow_request(client)` and, while it is in flight, time a health call."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        slow = asyncio.create_task(slow_request(client))
        await asyncio.sleep(0.3)  # let the slow job start running
        started = time.monotonic()
        health = await client.get("/api/v1/health")
        health_seconds = time.monotonic() - started
        slow_done_before_health = slow.done()
        slow_response = await slow
    return health, health_seconds, slow_done_before_health, slow_response


def test_health_answers_promptly_while_an_analysis_is_running(api, monkeypatch):
    def slow_checks(self, min_rows=1):
        time.sleep(SLOW_SECONDS)  # blocking, like pandas/tiktoken/SQLite work
        return {"issues": [], "checks": {}, "summary_stats": {}}

    monkeypatch.setattr(api.mcp_server.QualityPipeline, "run_all_checks", slow_checks)

    async def slow(client):
        return await client.post(
            "/api/v1/analyze",
            files={"data_file": ("d.csv", b"a\n1\n", "text/csv")},
            headers={"X-API-Key": KEY},
        )

    health, seconds, slow_done_early, slow_resp = asyncio.run(_health_during(api.app, slow))

    assert health.status_code == 200
    assert seconds < HEALTH_DEADLINE_SECONDS, f"health took {seconds:.2f}s while analysis ran"
    assert slow_done_early is False  # the analysis really was still running
    assert slow_resp.status_code == 200


def test_health_answers_promptly_while_inline_json_analysis_is_running(api, monkeypatch):
    def slow_checks(self, min_rows=1):
        time.sleep(SLOW_SECONDS)
        return {"issues": [], "checks": {}, "summary_stats": {}}

    monkeypatch.setattr(api.mcp_server.QualityPipeline, "run_all_checks", slow_checks)

    async def slow(client):
        return await client.post(
            "/api/v1/analyze/json",
            json={"data_content": "a\n1\n", "format": "csv"},
            headers={"X-API-Key": KEY},
        )

    health, seconds, slow_done_early, slow_resp = asyncio.run(_health_during(api.app, slow))

    assert health.status_code == 200
    assert seconds < HEALTH_DEADLINE_SECONDS
    assert slow_done_early is False
    assert slow_resp.status_code == 200


def test_health_answers_promptly_while_a_dictionary_is_being_parsed(api, monkeypatch):
    program = MagicMock(program_id="pid", num_fields=1, num_basic_rules=0, num_logic_rules=0,
                        generated_code="", schema={}, dictionary_format="csv", model_used="m")
    program.name = "n"

    manager = MagicMock()

    def slow_create(**kwargs):
        time.sleep(SLOW_SECONDS)  # stands in for the blocking LLM call path
        return program

    manager.create_program_from_dictionary.side_effect = slow_create
    monkeypatch.setattr(api, "program_manager", manager)

    async def slow(client):
        return await client.post(
            "/api/v1/dictionary/parse",
            files={"dictionary_file": ("d.csv", b"a,b\n1,2\n", "text/csv")},
            headers={"X-API-Key": KEY},
        )

    health, seconds, slow_done_early, slow_resp = asyncio.run(_health_during(api.app, slow))

    assert health.status_code == 200
    assert seconds < HEALTH_DEADLINE_SECONDS
    assert slow_done_early is False
    assert slow_resp.status_code == 200


def test_health_answers_promptly_while_a_dictionary_lookup_is_running(api, monkeypatch):
    manager = MagicMock()

    def slow_load(dict_id):
        time.sleep(SLOW_SECONDS)
        return None

    manager.db.load_program.side_effect = slow_load
    monkeypatch.setattr(api, "program_manager", manager)

    async def slow(client):
        return await client.get("/api/v1/dictionary/abc", headers={"X-API-Key": KEY})

    health, seconds, slow_done_early, slow_resp = asyncio.run(_health_during(api.app, slow))

    assert health.status_code == 200
    assert seconds < HEALTH_DEADLINE_SECONDS
    assert slow_done_early is False
    assert slow_resp.status_code == 404
