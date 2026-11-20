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


# ---- PDF page size (decoded, fail closed) ----------------------------------
import zlib

from pypdf.generic import ArrayObject, DictionaryObject


def _pdf(content: bytes, pages=1, flate=False, form=False, filter_name=None):
    """In-memory PDF. flate: compress the content stream. form: put the content in
    a Form XObject drawn by a tiny page stream instead."""
    w = pypdf.PdfWriter()
    for _ in range(pages):
        page = w.add_blank_page(200, 200)
        stream = DecodedStreamObject()
        stream.set_data(content)
        if flate:
            stream = stream.flate_encode()
        if filter_name:
            stream[NameObject("/Filter")] = NameObject(filter_name)
        if form:
            stream[NameObject("/Type")] = NameObject("/XObject")
            stream[NameObject("/Subtype")] = NameObject("/Form")
            from pypdf.generic import RectangleObject
            stream[NameObject("/BBox")] = RectangleObject([0, 0, 10, 10])
            xo = DictionaryObject({NameObject("/Fm0"): w._add_object(stream)})
            page[NameObject("/Resources")] = DictionaryObject({NameObject("/XObject"): xo})
            small = DecodedStreamObject()
            small.set_data(b"/Fm0 Do")
            page[NameObject("/Contents")] = w._add_object(small)
        else:
            page[NameObject("/Contents")] = w._add_object(stream)
    buf = io.BytesIO()
    w.write(buf)
    return pypdf.PdfReader(io.BytesIO(buf.getvalue()))


def _spy_extract(monkeypatch, reader):
    calls = []
    monkeypatch.setattr(type(reader.pages[0]), "extract_text", lambda self, *a, **k: calls.append(1) or "")
    return calls


def test_a_page_with_content_over_the_per_page_byte_limit_is_refused_before_extraction(monkeypatch):
    monkeypatch.setenv("UI_PDF_MAX_PAGE_BYTES", "100")
    reader = _pdf(b"BT (" + b"a" * 500 + b") Tj ET")
    calls = _spy_extract(monkeypatch, reader)
    with pytest.raises(DictionaryTooLargeError) as e:
        extract_pdf_text(reader)
    assert "Page 1" in str(e.value) and "100" in str(e.value)
    assert calls == []


def test_a_small_compressed_stream_that_expands_past_the_limit_is_refused_without_extraction(monkeypatch):
    monkeypatch.setenv("UI_PDF_MAX_PAGE_BYTES", "5000")
    reader = _pdf(b"0" * 2_000_000, flate=True)   # stored size (~2 KB) is under the limit
    assert len(reader.pages[0]["/Contents"].get_object()._data) < 5000
    calls = _spy_extract(monkeypatch, reader)
    with pytest.raises(DictionaryTooLargeError) as e:
        extract_pdf_text(reader)
    assert "once decoded" in str(e.value)
    assert calls == []


def test_a_form_xobject_carrying_the_large_content_is_refused(monkeypatch):
    monkeypatch.setenv("UI_PDF_MAX_PAGE_BYTES", "1000")
    reader = _pdf(b"0" * 500_000, flate=True, form=True)
    calls = _spy_extract(monkeypatch, reader)
    with pytest.raises(DictionaryTooLargeError):
        extract_pdf_text(reader)
    assert calls == []


def test_a_stream_with_an_encoding_that_cannot_be_bounded_is_refused(monkeypatch):
    monkeypatch.setenv("UI_PDF_MAX_PAGE_BYTES", "100000")
    reader = _pdf(b"abc", filter_name="/LZWDecode")
    calls = _spy_extract(monkeypatch, reader)
    with pytest.raises(DictionaryTooLargeError) as e:
        extract_pdf_text(reader)
    assert "cannot be size-checked" in str(e.value)
    assert calls == []


def test_a_measurement_error_refuses_the_page_instead_of_treating_it_as_empty(monkeypatch):
    reader = _pdf(b"BT ET")
    calls = _spy_extract(monkeypatch, reader)

    def boom(*a, **k):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(web_app, "_bounded_stream_size", boom)
    with pytest.raises(DictionaryTooLargeError) as e:
        extract_pdf_text(reader)
    assert "could not be size-checked" in str(e.value)
    assert calls == []


def test_a_normal_small_pdf_still_extracts(monkeypatch):
    monkeypatch.setenv("UI_PDF_MAX_PAGE_BYTES", "10000")
    assert isinstance(extract_pdf_text(_pdf(b"BT ET")), str)
    assert isinstance(extract_pdf_text(_pdf(b"BT ET", flate=True)), str)
    assert isinstance(extract_pdf_text(_pdf(b"BT ET", flate=True, form=True)), str)


def test_a_page_without_content_measures_zero_and_streams_in_an_array_add_up():
    blank = pypdf.PdfWriter()
    blank.add_blank_page(10, 10)
    assert web_app._page_content_bytes(blank.pages[0], 100) == 0
    reader = _pdf(b"x" * 40)
    assert web_app._page_content_bytes(reader.pages[0], 100) == 40
    with pytest.raises(web_app._PageSizeError):
        web_app._page_content_bytes(reader.pages[0], 39)
