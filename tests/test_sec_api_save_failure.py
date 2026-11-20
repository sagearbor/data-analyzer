"""Integrity: a failed save of a parsed dictionary is never reported as a
successful save (ADP-87 item 3: A-14)."""

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.program_cache import ProgramDatabase
from src.program_manager import ProgramManager, ProgramSaveError

KEY = "sec-test-key-save"
HEADERS = {"X-API-Key": KEY}

PARSED = {
    "fields": [{"name": "age", "type": "int"}],
    "schema": {"age": "int"},
    "metadata": {"total_fields": 1, "dictionary_name": "Demo"},
}


@pytest.fixture
def manager(tmp_path, monkeypatch):
    llm = MagicMock()
    llm.parse_dictionary.return_value = PARSED
    llm.deployment = "test-model"
    # Keep the manager's own database inside the test's temp dir (not ~/.data_analyzer).
    monkeypatch.setattr(
        "src.program_manager.ProgramDatabase",
        lambda: ProgramDatabase(db_path=tmp_path / "programs.db"),
    )
    return ProgramManager(llm)


def test_saving_two_programs_with_the_same_name_raises_instead_of_returning_a_phantom_id(manager):
    manager.create_program_from_dictionary("age,int\n", program_name="Same Name")

    with pytest.raises(ProgramSaveError):
        manager.create_program_from_dictionary("age,int\n", program_name="Same Name")


def test_a_program_that_was_really_saved_can_be_loaded_back_by_its_id(manager):
    program = manager.create_program_from_dictionary("age,int\n")

    assert manager.db.load_program(program.program_id) is not None


def test_a_custom_program_name_is_stored_by_the_single_insert(manager):
    program = manager.create_program_from_dictionary("age,int\n", program_name="PatientIntake")

    assert program.name == "PatientIntake"
    assert manager.db.load_program("PatientIntake").program_id == program.program_id


def test_not_saving_still_works_and_stores_nothing(manager):
    program = manager.create_program_from_dictionary("age,int\n", save=False)

    assert manager.db.load_program(program.program_id) is None


def test_database_refusing_the_save_is_raised_not_swallowed(manager):
    manager.db = MagicMock()
    manager.db.save_program.return_value = False

    with pytest.raises(ProgramSaveError):
        manager.create_program_from_dictionary("age,int\n")


def test_failed_insert_leaves_no_orphaned_code_file(manager):
    manager.create_program_from_dictionary("age,int\n", program_name="Dup")
    before = sorted(p.name for p in manager.db.programs_dir.iterdir())

    with pytest.raises(ProgramSaveError):
        manager.create_program_from_dictionary("age,int\n", program_name="Dup")

    assert sorted(p.name for p in manager.db.programs_dir.iterdir()) == before


def test_api_reports_a_failed_save_as_an_error_not_as_a_program_id(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "API_KEY", KEY)
    pm = MagicMock()
    pm.create_program_from_dictionary.side_effect = ProgramSaveError("Program could not be saved")
    monkeypatch.setattr(api_server, "program_manager", pm)

    resp = TestClient(api_server.app).post(
        "/api/v1/dictionary/parse",
        files={"dictionary_file": ("d.csv", b"a,b\n1,2\n", "text/csv")},
        data={"save_program": "true", "program_name": "X"},
        headers=HEADERS,
    )

    assert resp.status_code == 500
    assert "program_id" not in resp.json()
    assert "could not be saved" in resp.json()["error"]


def test_api_passes_the_custom_name_into_the_single_save_instead_of_renaming_afterwards(monkeypatch):
    import api_server

    monkeypatch.setattr(api_server, "API_KEY", KEY)
    program = MagicMock(program_id="pid", num_fields=1, num_basic_rules=0, num_logic_rules=0,
                        generated_code="", schema={}, dictionary_format="csv", model_used="m")
    program.name = "Custom"
    pm = MagicMock()
    pm.create_program_from_dictionary.return_value = program
    monkeypatch.setattr(api_server, "program_manager", pm)

    resp = TestClient(api_server.app).post(
        "/api/v1/dictionary/parse",
        files={"dictionary_file": ("d.csv", b"a,b\n1,2\n", "text/csv")},
        data={"save_program": "true", "program_name": "Custom"},
        headers=HEADERS,
    )

    assert resp.status_code == 200
    assert pm.create_program_from_dictionary.call_args.kwargs["program_name"] == "Custom"
    pm.db.save_program.assert_not_called()  # no second INSERT that could fail silently
