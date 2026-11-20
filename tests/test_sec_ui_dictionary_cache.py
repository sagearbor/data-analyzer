"""ADP-72 UI part / B-12: the parsed-dictionary disk cache is private to one
browser session, keyed by SHA-256, expires, and never exposes other users."""
import json
import os
import stat
import time

import pytest

from web_app import DictionaryCache

SID_A = "a" * 32
SID_B = "b" * 32


def test_a_new_session_cannot_read_entries_written_by_another_session(tmp_path):
    a = DictionaryCache(tmp_path, SID_A)
    key = a.key_for(b"dictionary bytes")
    a.put(key, {"rules": {"age": {"min": 1}}})
    b = DictionaryCache(tmp_path, SID_B)
    assert b.get(key) is None
    assert a.get(key) == {"rules": {"age": {"min": 1}}}


def test_clear_removes_only_the_current_sessions_entries(tmp_path):
    a, b = DictionaryCache(tmp_path, SID_A), DictionaryCache(tmp_path, SID_B)
    ka, kb = a.key_for(b"one"), b.key_for(b"two")
    a.put(ka, {"x": 1})
    b.put(kb, {"y": 2})
    assert a.clear() == 1
    assert a.count() == 0
    assert b.get(kb) == {"y": 2}


def test_count_reports_only_the_current_session(tmp_path):
    a, b = DictionaryCache(tmp_path, SID_A), DictionaryCache(tmp_path, SID_B)
    a.put(a.key_for(b"1"), {})
    b.put(b.key_for(b"2"), {})
    b.put(b.key_for(b"3"), {})
    assert (a.count(), b.count()) == (1, 2)


def test_old_shared_cache_entries_in_the_legacy_layout_are_never_read(tmp_path):
    # The legacy layout was one flat folder of <md5>.json files.
    import hashlib
    content = b"legacy dictionary"
    (tmp_path / f"{hashlib.md5(content).hexdigest()}.json").write_text(json.dumps({"rules": {"evil": {}}}))
    (tmp_path / f"{hashlib.sha256(content).hexdigest()}.json").write_text(json.dumps({"rules": {"evil": {}}}))
    store = DictionaryCache(tmp_path, SID_A)
    assert store.get(store.key_for(content)) is None
    assert store.count() == 0


def test_keys_are_sha256_not_md5():
    import hashlib
    k = DictionaryCache.key_for(b"abc")
    assert k == hashlib.sha256(b"abc").hexdigest() and len(k) == 64
    assert DictionaryCache.key_for(b"abc", llm=True) == k + "_llm"


def test_expired_entries_are_not_returned_and_are_deleted(tmp_path):
    store = DictionaryCache(tmp_path, SID_A, ttl_seconds=60)
    key = store.key_for(b"x")
    store.put(key, {"a": 1})
    path = store.dir / f"{key}.json"
    old = time.time() - 3600
    os.utime(path, (old, old))
    assert store.get(key) is None
    assert not path.exists()


def test_expired_entries_of_other_sessions_are_pruned_but_fresh_ones_stay(tmp_path):
    a = DictionaryCache(tmp_path, SID_A, ttl_seconds=60)
    b = DictionaryCache(tmp_path, SID_B, ttl_seconds=60)
    stale, fresh = b.key_for(b"stale"), b.key_for(b"fresh")
    b.put(stale, {})
    b.put(fresh, {})
    old = time.time() - 3600
    os.utime(b.dir / f"{stale}.json", (old, old))
    assert a.prune() == 1
    assert not (b.dir / f"{stale}.json").exists()
    assert (b.dir / f"{fresh}.json").exists()


def test_ttl_is_configurable_by_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("UI_CACHE_TTL_HOURS", "2")
    assert DictionaryCache(tmp_path, SID_A).ttl == 2 * 3600


@pytest.mark.parametrize("bad", ["../etc", "short", "G" * 32, "", None])
def test_session_ids_that_could_escape_the_cache_folder_are_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        DictionaryCache(tmp_path, bad)


def test_cache_keys_that_could_escape_the_session_folder_are_rejected(tmp_path):
    store = DictionaryCache(tmp_path, SID_A)
    for bad in ("../../x", "abc", "a" * 64 + "/../x"):
        with pytest.raises(ValueError):
            store.put(bad, {})


def test_cache_files_are_owner_only(tmp_path):
    store = DictionaryCache(tmp_path, SID_A)
    key = store.key_for(b"x")
    store.put(key, {})
    mode = stat.S_IMODE((store.dir / f"{key}.json").stat().st_mode)
    assert mode & 0o077 == 0


def test_a_corrupt_entry_is_a_miss_not_a_crash(tmp_path):
    store = DictionaryCache(tmp_path, SID_A)
    key = store.key_for(b"x")
    (store.dir / f"{key}.json").write_text("{not json")
    assert store.get(key) is None


def test_the_ui_never_exposes_the_cache_path_or_uses_the_shared_folder():
    src = open("web_app.py", encoding="utf-8").read()
    assert ".data_analyzer_cache" not in src
    assert "Cache location" not in src
    assert "Clear All Cache" not in src
    assert "md5" not in src.lower()
    assert "pickle" not in src and ".pkl" not in src
