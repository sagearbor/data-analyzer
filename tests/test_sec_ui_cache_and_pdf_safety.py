"""UI hardening: the dictionary cache refuses unsafe directories/files and falls
back to a private folder; PDF pages with oversized content are refused before
text extraction. No network; synthetic data only."""
import io
import os
import stat

import pypdf
import pytest
from pypdf.generic import DecodedStreamObject, NameObject

import web_app
from web_app import DictionaryCache, DictionaryTooLargeError, extract_pdf_text

SID = "c" * 32


@pytest.fixture(autouse=True)
def _reset_fallback():
    DictionaryCache._fallback_dir = None
    yield
    DictionaryCache._fallback_dir = None


def _works(cache, tmp_path):
    key = cache.key_for(b"x")
    cache.put(key, {"ok": 1})
    assert cache.get(key) == {"ok": 1}
    return cache


def test_a_symlinked_cache_base_is_refused_and_the_cache_still_works_via_a_private_fallback(tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir(mode=0o700)
    base = tmp_path / "base"
    base.symlink_to(victim)
    cache = _works(DictionaryCache(base, SID), tmp_path)
    assert not cache.base.is_symlink() and cache.base != base
    assert list(victim.iterdir()) == []          # nothing was written through the link
    assert stat.S_IMODE(os.lstat(cache.base).st_mode) & 0o077 == 0


def test_a_base_with_group_or_other_permission_bits_is_refused(tmp_path):
    base = tmp_path / "shared"
    base.mkdir()
    os.chmod(base, 0o777)
    cache = _works(DictionaryCache(base, SID), tmp_path)
    assert cache.base != base
    assert list(base.iterdir()) == []


def test_a_symlinked_session_directory_is_refused(tmp_path):
    base = tmp_path / "base"
    base.mkdir(mode=0o700)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (base / SID).symlink_to(elsewhere)
    cache = _works(DictionaryCache(base, SID), tmp_path)
    assert cache.base != base and list(elsewhere.iterdir()) == []


def test_a_private_base_is_used_as_is(tmp_path):
    base = tmp_path / "base"
    cache = _works(DictionaryCache(base, SID), tmp_path)
    assert cache.base == base


def test_a_symlinked_entry_file_is_not_read_and_not_written_through(tmp_path):
    cache = DictionaryCache(tmp_path / "base", SID)
    key = cache.key_for(b"entry")
    target = tmp_path / "target.json"
    target.write_text('{"stolen": true}')
    (cache.dir / f"{key}.json").symlink_to(target)
    assert cache.get(key) is None
    (cache.dir / f"{key}.json").symlink_to(target)   # get() removed the link; plant it again for put()
    cache.put(key, {"fresh": 1})
    assert target.read_text() == '{"stolen": true}'
    assert cache.get(key) == {"fresh": 1}


def test_a_non_regular_entry_file_is_refused(tmp_path):
    cache = DictionaryCache(tmp_path / "base", SID)
    key = cache.key_for(b"fifo")
    os.mkfifo(cache.dir / f"{key}.json")
    assert cache.get(key) is None        # must neither hang nor crash
    cache.put(key, {"x": 1})             # nor hang on write


# ---- PDF page size ---------------------------------------------------------
def _pdf(content: bytes, pages=1):
    w = pypdf.PdfWriter()
    for _ in range(pages):
        page = w.add_blank_page(200, 200)
        stream = DecodedStreamObject()
        stream.set_data(content)
        page[NameObject("/Contents")] = w._add_object(stream)
    buf = io.BytesIO()
    w.write(buf)
    return pypdf.PdfReader(io.BytesIO(buf.getvalue()))


def test_a_page_with_content_over_the_per_page_byte_limit_is_refused_before_extraction(monkeypatch):
    monkeypatch.setenv("UI_PDF_MAX_PAGE_BYTES", "100")
    reader = _pdf(b"BT (" + b"a" * 500 + b") Tj ET")
    calls = []
    monkeypatch.setattr(type(reader.pages[0]), "extract_text", lambda self, *a, **k: calls.append(1) or "")
    with pytest.raises(DictionaryTooLargeError) as e:
        extract_pdf_text(reader)
    assert "Page 1" in str(e.value) and "100" in str(e.value)
    assert calls == []


def test_a_small_page_is_still_extracted(monkeypatch):
    monkeypatch.setenv("UI_PDF_MAX_PAGE_BYTES", "10000")
    assert isinstance(extract_pdf_text(_pdf(b"BT ET")), str)


def test_a_page_without_content_and_an_array_of_streams_are_handled():
    blank = pypdf.PdfWriter()
    blank.add_blank_page(10, 10)
    assert web_app._page_content_bytes(blank.pages[0]) == 0
    reader = _pdf(b"x" * 40)
    assert web_app._page_content_bytes(reader.pages[0]) == 40
