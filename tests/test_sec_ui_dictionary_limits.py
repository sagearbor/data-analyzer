"""ADP-73 UI parts / B-03 / B-04: linear-time manual PDF pattern extraction and
bounded page/character counts for dictionary parsing. Synthetic data only."""
import time

import pytest

from web_app import (DictionaryTooLargeError, check_dictionary_text_size,
                     extract_pdf_manual_rules, extract_pdf_text, max_upload_bytes)


class _Page:
    def __init__(self, text):
        self._t = text

    def extract_text(self):
        return self._t


class _Reader:
    def __init__(self, texts):
        self.pages = [_Page(t) for t in texts]


# --- B-03 -------------------------------------------------------------------
@pytest.mark.parametrize("payload", [
    "a" * 200_000,                       # one huge word, no keyword (cubic in the old regex)
    ("a" * 50_000) + " range " + "1" * 50_000,
    "word " * 40_000,
    ("x1 " * 30_000) + "between",
    "\n".join("a" * 5_000 for _ in range(200)),
])
def test_manual_pdf_rule_extraction_is_fast_on_adversarial_text(payload):
    started = time.perf_counter()
    extract_pdf_manual_rules(payload)
    # The old patterns needed minutes at this size; linear code needs milliseconds.
    assert time.perf_counter() - started < 3.0


def test_manual_pdf_rules_still_find_date_and_range_fields():
    text = "birth_date is a date field\nage valid range 18 to 85\nnotes free text"
    rules = extract_pdf_manual_rules(text)
    assert rules["birth_date"] == {"type": "date"}
    assert rules["age"] == {"min": 18, "max": 85}
    assert "notes" not in rules


def test_manual_pdf_rules_ignore_a_range_without_two_numbers():
    assert extract_pdf_manual_rules("score range 10") == {}


# --- B-04 (b) ---------------------------------------------------------------
def test_upload_limit_defaults_to_25_mb_and_is_configurable(monkeypatch):
    monkeypatch.delenv("UI_MAX_UPLOAD_MB", raising=False)
    assert max_upload_bytes() == 25 * 1024 * 1024
    monkeypatch.setenv("UI_MAX_UPLOAD_MB", "10")
    assert max_upload_bytes() == 10 * 1024 * 1024


@pytest.mark.parametrize("bad", ["", "abc", "0", "-5", "1.5"])
def test_a_bad_upload_limit_value_falls_back_to_the_default_never_unlimited(monkeypatch, bad):
    monkeypatch.setenv("UI_MAX_UPLOAD_MB", bad)
    assert max_upload_bytes() == 25 * 1024 * 1024


def test_default_upload_limit_is_below_the_apis_100_mb_default():
    assert 25 * 1024 * 1024 < 100 * 1024 * 1024


# --- B-04 (c) ---------------------------------------------------------------
def test_pdf_with_too_many_pages_is_refused_with_a_clear_message():
    with pytest.raises(DictionaryTooLargeError, match=r"6 pages; the limit is 5"):
        extract_pdf_text(_Reader(["x"] * 6), max_pages=5, max_chars=1000)


def test_pdf_with_too_much_text_is_refused_with_a_clear_message():
    with pytest.raises(DictionaryTooLargeError, match="characters"):
        extract_pdf_text(_Reader(["y" * 600, "y" * 600]), max_pages=5, max_chars=1000)


def test_pdf_within_limits_returns_all_text(monkeypatch):
    assert extract_pdf_text(_Reader(["one", "two"]), max_pages=5, max_chars=1000) == "one\ntwo\n"


def test_pdf_limits_come_from_environment(monkeypatch):
    monkeypatch.setenv("UI_PDF_MAX_PAGES", "2")
    with pytest.raises(DictionaryTooLargeError):
        extract_pdf_text(_Reader(["a", "b", "c"]))


def test_non_pdf_dictionary_text_has_the_same_character_limit(monkeypatch):
    monkeypatch.setenv("UI_DICT_MAX_CHARS", "100")
    with pytest.raises(DictionaryTooLargeError):
        check_dictionary_text_size("z" * 101)
    assert check_dictionary_text_size("z" * 100) == "z" * 100
