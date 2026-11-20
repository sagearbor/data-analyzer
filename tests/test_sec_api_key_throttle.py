"""Security: wrong API keys cannot be tried without limit, and odd keys give a
clean error (ADP-85: A-04, A-12; ADP-78: A-08)."""

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

KEY = "sec-test-key-throttle"


@pytest.fixture
def api(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "API_KEY", KEY)
    monkeypatch.setenv("DATA_ANALYZER_AUTH_FAIL_LIMIT", "5")
    monkeypatch.setenv("DATA_ANALYZER_AUTH_FAIL_WINDOW_SECONDS", "60")
    api_server.limiter.reset()  # also clears the failed-attempt counters
    yield api_server
    api_server.limiter.reset()


@pytest.fixture
def client(api):
    return TestClient(api.app)


def _post(client, key):
    return client.post(
        "/api/v1/analyze",
        files={"data_file": ("d.csv", b"a\n1\n", "text/csv")},
        headers={"X-API-Key": key} if key is not None else {},
    )


def test_repeated_wrong_keys_are_answered_with_429_after_the_configured_number_of_failures(client):
    statuses = [_post(client, f"wrong-{i}").status_code for i in range(8)]

    assert statuses[:5] == [403] * 5
    assert statuses[5:] == [429] * 3


def test_the_429_for_too_many_failures_tells_the_client_when_to_retry(client):
    for i in range(5):
        _post(client, f"wrong-{i}")

    resp = _post(client, "wrong-again")

    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "60"


def test_a_locked_out_client_is_refused_even_with_the_correct_key(client):
    # Identical 429 for a right or wrong key during a lockout: no guessing oracle.
    for i in range(5):
        _post(client, f"wrong-{i}")

    right, wrong = _post(client, KEY), _post(client, "wrong-again")
    assert right.status_code == wrong.status_code == 429
    assert right.json() == wrong.json() or right.json()["error"] == wrong.json()["error"]
    assert right.headers["Retry-After"] == wrong.headers["Retry-After"] == "60"


def test_correct_keys_never_count_as_failures(client):
    statuses = [_post(client, KEY).status_code for _ in range(12)]

    assert statuses == [200] * 12


def test_missing_keys_do_not_lock_the_client_out(client):
    assert [_post(client, None).status_code for _ in range(8)] == [401] * 8
    assert _post(client, KEY).status_code == 200


def test_failures_are_counted_per_client_address(api, client):
    from fastapi import HTTPException

    for i in range(5):
        _post(client, f"wrong-{i}")  # all from the test client's peer address

    def fake_request(host):
        return type("R", (), {"client": type("C", (), {"host": host})})()

    # A different peer address has its own, untouched allowance...
    api._reject_if_auth_locked_out("api_key", fake_request("203.0.113.9"))  # must not raise
    # ...while the offending one is locked out.
    with pytest.raises(HTTPException) as exc:
        api._reject_if_auth_locked_out("api_key", fake_request("testclient"))
    assert exc.value.status_code == 429


def test_the_failure_limit_and_window_are_configurable(api, monkeypatch, client):
    monkeypatch.setenv("DATA_ANALYZER_AUTH_FAIL_LIMIT", "2")
    api.limiter.reset()

    statuses = [_post(client, f"w{i}").status_code for i in range(4)]

    assert statuses == [403, 403, 429, 429]


def test_the_health_endpoint_is_not_affected_by_a_lockout(client):
    for i in range(6):
        _post(client, f"wrong-{i}")

    assert client.get("/api/v1/health").status_code == 200


def test_non_ascii_api_key_gets_403_not_a_server_error(client):
    resp = client.post(
        "/api/v1/analyze",
        files={"data_file": ("d.csv", b"a\n1\n", "text/csv")},
        headers={"X-API-Key": "café".encode("latin-1")},
    )

    assert resp.status_code == 403
    assert resp.json()["error"] == "Invalid credentials provided."


def test_failed_attempt_log_line_names_the_client_but_not_the_key(client, caplog):
    with caplog.at_level("WARNING"):
        _post(client, "very-secret-wrong-guess")

    text = " ".join(r.getMessage() for r in caplog.records)
    assert "auth_failure" in text
    assert "very-secret-wrong-guess" not in text
