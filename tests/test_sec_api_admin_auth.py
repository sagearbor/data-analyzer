"""Security: no default admin password, and admin checks fail closed (ADP-84:
A-06, A-07, B-01)."""

import asyncio
import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.program_cache import ProgramDatabase


@pytest.fixture
def db(tmp_path):
    return ProgramDatabase(db_path=tmp_path / "programs.db")


def test_the_old_hardcoded_admin_password_is_not_accepted_when_nothing_is_configured(db, monkeypatch):
    monkeypatch.delenv("DATA_ANALYZER_ADMIN_PASSWORD", raising=False)
    monkeypatch.delenv("ADMIN_PW", raising=False)

    assert db._verify_admin("dataAnalyzerAdmin666") is False


def test_the_legacy_ADMIN_PW_variable_alone_does_not_enable_admin_access(db, monkeypatch):
    monkeypatch.delenv("DATA_ANALYZER_ADMIN_PASSWORD", raising=False)
    monkeypatch.setenv("ADMIN_PW", "legacy-name-password")

    assert db._verify_admin("legacy-name-password") is False


def test_the_documented_admin_password_variable_is_the_one_that_is_checked(db, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_ADMIN_PASSWORD", "correct-horse")

    assert db._verify_admin("correct-horse") is True
    assert db._verify_admin("wrong") is False


def test_an_empty_admin_password_never_matches_an_empty_configuration(db, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_ADMIN_PASSWORD", "")

    assert db._verify_admin("") is False


def test_non_ascii_admin_password_attempts_are_rejected_without_raising(db, monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_ADMIN_PASSWORD", "correct-horse")

    assert db._verify_admin("café") is False


def test_admin_comparison_is_constant_time():
    text = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "program_cache.py")).read()
    assert "secrets.compare_digest" in text
    assert "dataAnalyzerAdmin666" not in text


def test_admin_dependency_refuses_with_503_when_unconfigured_in_a_local_environment(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "ADMIN_PASSWORD", None)

    with pytest.raises(HTTPException) as exc:
        asyncio.run(api_server.verify_admin_password(admin_password="anything"))
    assert exc.value.status_code == 503


def test_admin_dependency_refuses_with_503_when_unconfigured_in_a_deployed_environment(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "ADMIN_PASSWORD", "")
    monkeypatch.setenv("CONTAINER_APP_NAME", "some-app")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(api_server.verify_admin_password(admin_password=None))
    assert exc.value.status_code == 503


def test_admin_dependency_accepts_only_the_configured_password(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "ADMIN_PASSWORD", "s3cret-admin")

    assert asyncio.run(api_server.verify_admin_password(admin_password="s3cret-admin")) == "s3cret-admin"
    with pytest.raises(HTTPException) as exc:
        asyncio.run(api_server.verify_admin_password(admin_password="nope"))
    assert exc.value.status_code == 403


def test_non_ascii_admin_header_gives_403_not_a_crash(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "ADMIN_PASSWORD", "s3cret-admin")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(api_server.verify_admin_password(admin_password="éé"))
    assert exc.value.status_code == 403


def test_api_key_check_refuses_with_503_when_unconfigured_in_a_deployed_environment(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "API_KEY", None)
    monkeypatch.setenv("APP_ENV", "prod")

    with pytest.raises(HTTPException) as exc:
        asyncio.run(api_server.verify_api_key(api_key=None))
    assert exc.value.status_code == 503
