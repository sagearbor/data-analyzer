"""ADP-85: one client identifier (peer address, or the Nth X-Forwarded-For entry
from the right behind a trusted proxy) feeds BOTH the rate limiter and the
wrong-key throttle; only failed attempts are throttled."""
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY = "sec-test-key-client-id"


@pytest.fixture
def api(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "API_KEY", KEY)
    monkeypatch.setenv("DATA_ANALYZER_AUTH_FAIL_LIMIT", "3")
    monkeypatch.setenv("DATA_ANALYZER_AUTH_FAIL_WINDOW_SECONDS", "60")
    monkeypatch.delenv("DATA_ANALYZER_TRUSTED_PROXY_HOPS", raising=False)
    api_server.limiter.reset()
    yield api_server
    api_server.limiter.reset()


@pytest.fixture
def client(api):
    return TestClient(api.app)


def _post(client, key, xff=None):
    headers = {}
    if key is not None:
        headers["X-API-Key"] = key
    if xff is not None:
        headers["X-Forwarded-For"] = xff
    return client.post("/api/v1/analyze", files={"data_file": ("d.csv", b"a\n1\n", "text/csv")}, headers=headers)


def _req(api, xff=None, host="198.51.100.7"):
    from starlette.requests import Request

    headers = [(b"x-forwarded-for", xff.encode())] if xff is not None else []
    return Request({"type": "http", "headers": headers, "client": (host, 1234), "method": "GET", "path": "/"})


def test_with_the_setting_unset_a_forged_forwarded_for_header_does_not_change_the_client_id(api):
    assert api.get_client_id(_req(api, "1.2.3.4")) == "198.51.100.7"
    assert api.get_client_id(_req(api, "1.2.3.4, 5.6.7.8")) == "198.51.100.7"


def test_with_the_setting_unset_forged_headers_do_not_escape_a_lockout(client):
    for i in range(3):
        assert _post(client, f"w{i}", xff=f"10.0.0.{i}").status_code == 403
    assert _post(client, "w-again", xff="10.9.9.9").status_code == 429


def test_with_one_trusted_hop_the_rightmost_forwarded_for_entry_is_the_client_id(api, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_TRUSTED_PROXY_HOPS", "1")
    assert api.get_client_id(_req(api, "6.6.6.6, 203.0.113.5")) == "203.0.113.5"


def test_with_two_trusted_hops_the_second_entry_from_the_right_is_used(api, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_TRUSTED_PROXY_HOPS", "2")
    assert api.get_client_id(_req(api, "6.6.6.6, 203.0.113.5, 10.0.0.1")) == "203.0.113.5"


def test_with_trusted_hops_a_missing_or_too_short_header_falls_back_to_the_peer_address(api, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_TRUSTED_PROXY_HOPS", "2")
    assert api.get_client_id(_req(api)) == "198.51.100.7"
    assert api.get_client_id(_req(api, "203.0.113.5")) == "198.51.100.7"


@pytest.mark.parametrize("value", ["0", "-1", "abc", ""])
def test_a_zero_negative_or_invalid_hop_setting_means_the_peer_address(api, monkeypatch, value):
    monkeypatch.setenv("DATA_ANALYZER_TRUSTED_PROXY_HOPS", value)
    assert api.get_client_id(_req(api, "203.0.113.5")) == "198.51.100.7"


def test_a_forwarded_for_entry_cannot_inject_log_or_storage_characters(api, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_TRUSTED_PROXY_HOPS", "1")
    cid = api.get_client_id(_req(api, "1.1.1.1, 9.9.9.9\r\nFAKE LOG LINE;drop " + "x" * 500))
    assert all(ch in "0123456789abcdefABCDEF:.[]%" for ch in cid)
    assert len(cid) <= 64
    # nothing valid left in the entry -> falls back to the peer address
    assert api.get_client_id(_req(api, "1.1.1.1, ;;;;")) == "198.51.100.7"


def test_the_setting_is_read_at_call_time(api, monkeypatch):
    req = _req(api, "6.6.6.6, 203.0.113.5")
    assert api.get_client_id(req) == "198.51.100.7"
    monkeypatch.setenv("DATA_ANALYZER_TRUSTED_PROXY_HOPS", "1")
    assert api.get_client_id(req) == "203.0.113.5"


def test_with_one_trusted_hop_a_lockout_applies_only_to_the_caller_who_earned_it(client, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_TRUSTED_PROXY_HOPS", "1")
    for i in range(3):
        assert _post(client, f"w{i}", xff="203.0.113.1").status_code == 403
    assert _post(client, "w-more", xff="203.0.113.1").status_code == 429
    # a different caller (different rightmost entry) is untouched, even with a wrong key
    assert _post(client, "w-other", xff="203.0.113.2").status_code == 403


def test_a_client_cannot_escape_its_lockout_by_prepending_fake_left_entries(client, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_TRUSTED_PROXY_HOPS", "1")
    for i in range(3):
        _post(client, f"w{i}", xff="203.0.113.1")
    for fake in ("1.1.1.1, 203.0.113.1", "8.8.8.8, 7.7.7.7, 203.0.113.1"):
        assert _post(client, "w-more", xff=fake).status_code == 429


def test_the_rate_limiter_uses_the_same_client_id_as_the_throttle(api, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_TRUSTED_PROXY_HOPS", "1")
    req = _req(api, "6.6.6.6, 203.0.113.5")
    assert api.limiter._key_func(req) == api.get_client_id(req) == "203.0.113.5"


def test_after_the_limit_a_wrong_key_gets_429_but_the_correct_key_from_the_same_client_succeeds(client):
    for i in range(3):
        assert _post(client, f"w{i}").status_code == 403
    wrong = _post(client, "w-more")
    assert wrong.status_code == 429 and wrong.headers["Retry-After"] == "60"
    assert _post(client, KEY).status_code == 200


def test_a_missing_key_is_still_401_during_a_lockout(client):
    for i in range(3):
        _post(client, f"w{i}")
    assert _post(client, None).status_code == 401


def test_admin_password_wrong_attempts_are_throttled_but_the_correct_password_still_works(api, monkeypatch):
    from fastapi import HTTPException
    import asyncio

    monkeypatch.setattr(api, "ADMIN_PASSWORD", "sec-test-admin")
    req = _req(api)

    def call(pw):
        return asyncio.run(api.verify_admin_password(pw, req))

    for _ in range(3):
        with pytest.raises(HTTPException) as e:
            call("wrong")
        assert e.value.status_code == 403
    with pytest.raises(HTTPException) as e:
        call("wrong")
    assert e.value.status_code == 429 and e.value.headers["Retry-After"] == "60"
    assert call("sec-test-admin") == "sec-test-admin"
    with pytest.raises(HTTPException) as e:
        call(None)
    assert e.value.status_code == 401
