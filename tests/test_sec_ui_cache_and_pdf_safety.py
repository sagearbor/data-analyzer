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
    fifo = cache.dir / f"{key}.json"
    assert cache.get(key) is None        # must neither hang nor crash
    cache.put(key, {"x": 1})             # nor hang on write
    # The write was refused: nothing was stored, and the special file was
    # neither replaced by a normal file nor turned into a readable entry.
    assert stat.S_ISFIFO(os.stat(fifo).st_mode)
    assert cache.get(key) is None


# ---- PDF extraction in a bounded child process ------------------------------
import sys
import textwrap

from tests.test_sec_ui_dictionary_limits import _Reader as make_pdf

PYTHON = sys.executable


def _substitute_worker(monkeypatch, script: str):
    """Tests only: replace the worker command with an inline script."""
    monkeypatch.setattr(web_app, "_pdf_worker_command",
                        lambda *a: [PYTHON, "-c", textwrap.dedent(script)])


def test_a_normal_small_pdf_extracts_through_the_real_subprocess():
    text, pages = web_app.extract_pdf_text_and_pages(make_pdf(["one", "two"]))
    assert (text, pages) == ("one\ntwo\n", 2)


def test_a_pdf_whose_text_exceeds_a_tiny_character_limit_is_refused():
    with pytest.raises(DictionaryTooLargeError, match="characters"):
        extract_pdf_text(make_pdf(["y" * 600, "y" * 600]), max_chars=1000)


def test_a_pdf_over_the_page_limit_is_refused_with_the_page_count():
    with pytest.raises(DictionaryTooLargeError, match=r"4 pages; the limit is 3"):
        extract_pdf_text(make_pdf(["a"] * 4), max_pages=3)


def test_a_worker_that_hangs_is_killed_at_the_timeout_and_the_user_is_told_it_took_too_long(monkeypatch):
    import time

    monkeypatch.setenv("UI_PDF_EXTRACT_TIMEOUT_SECONDS", "1")
    _substitute_worker(monkeypatch, "import time; time.sleep(60)")
    started = time.monotonic()
    with pytest.raises(DictionaryTooLargeError, match="took too long"):
        extract_pdf_text(make_pdf(["x"]))
    assert time.monotonic() - started < 10


@pytest.mark.parametrize("script", [
    "import sys; sys.exit(3)",
    "print('not json at all')",
    "import os, signal; os.kill(os.getpid(), signal.SIGKILL)",
    "print('{\"status\": \"ok\"}')",                       # ok without text: incomplete
    "print('{\"status\": \"something-else\"}')",
    "",
])
def test_a_worker_that_exits_badly_or_prints_garbage_is_refused(monkeypatch, script):
    _substitute_worker(monkeypatch, script)
    with pytest.raises(DictionaryTooLargeError):
        extract_pdf_text(make_pdf(["x"]))


def test_over_long_worker_output_is_refused_and_the_child_stopped(monkeypatch):
    _substitute_worker(monkeypatch, "import sys; sys.stdout.write('a' * 5_000_000); sys.stdout.flush()")
    with pytest.raises(DictionaryTooLargeError):
        extract_pdf_text(make_pdf(["x"]), max_chars=100)


def test_the_child_environment_has_no_api_key_or_azure_settings(monkeypatch):
    monkeypatch.setenv("DATA_ANALYZER_API_KEY", "SECRET-KEY-VALUE")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.invalid")
    _substitute_worker(monkeypatch, """
        import os, json
        bad = [k for k in os.environ if k.startswith(("DATA_ANALYZER", "AZURE", "LLM_"))]
        print(json.dumps({"status": "ok", "pages": 1, "text": "leaked:" + ",".join(bad)}))
    """)
    assert extract_pdf_text(make_pdf(["x"])) == "leaked:"


def test_the_worker_does_not_import_the_app_or_streamlit():
    import subprocess

    code = ("import runpy, sys; sys.argv=['w','5','100','400']; "
            "sys.stdin = open('/dev/null'); "
            "mod = runpy.run_path(r'%s', run_name='not_main'); "
            "print(any(m in sys.modules for m in ('web_app', 'streamlit')))" % web_app._PDF_WORKER_PATH)
    out = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "False"


@pytest.mark.skipif(sys.platform != "linux",
                    reason="RLIMIT_AS is only enforced on Linux (the container); unsupported/unreliable on macOS")
def test_an_over_limit_allocation_in_the_child_is_refused():
    import subprocess

    code = ("import runpy; m = runpy.run_path(r'%s', run_name='not_main'); "
            "m['apply_memory_limit'](100); "
            "\ntry:\n    bytearray(1024*1024*1024)\n    print('allocated')\n"
            "except MemoryError:\n    print('refused')" % web_app._PDF_WORKER_PATH)
    out = subprocess.run([PYTHON, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.stdout.strip() == "refused"


def test_the_memory_limit_helper_never_raises_even_when_the_platform_cannot_apply_it(monkeypatch):
    import runpy

    import resource

    mod = runpy.run_path(str(web_app._PDF_WORKER_PATH), run_name="not_main")

    # ADP-89: the old test accepted True or False and also really lowered this
    # process's address-space limit. Both outcomes are now forced with a stub.
    applied = []
    monkeypatch.setattr(resource, "setrlimit", lambda which, limits: applied.append((which, limits)))
    assert mod["apply_memory_limit"](512) is True
    which, (soft, hard) = applied[0]
    assert which == resource.RLIMIT_AS
    assert soft == hard >= 512 * 1024 * 1024   # baseline + 512 MB

    def refuse(which, limits):
        raise ValueError("platform cannot apply this limit")

    monkeypatch.setattr(resource, "setrlimit", refuse)
    assert mod["apply_memory_limit"](512) is False   # reported, not raised
