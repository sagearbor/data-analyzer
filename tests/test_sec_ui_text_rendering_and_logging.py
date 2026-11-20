"""ADP-78 UI parts / B-10 / B-11 / B-14: untrusted text is shown as text, the
script-based console helper is gone, and logs carry no file names."""
import ast
import logging
import re
from pathlib import Path

import pytest

from web_app import (field_markdown, file_fingerprint, md_escape,
                     validate_dictionary_shape)

SRC = Path("web_app.py").read_text(encoding="utf-8")


def _unescaped(markdown: str) -> str:
    """What a CommonMark renderer shows after removing backslash escapes."""
    return re.sub(r"\\([!-/:-@\[-`{-~])", r"\1", markdown)


@pytest.mark.parametrize("hostile", [
    "![x](https://attacker.invalid/p.png?x=1)",
    "[click](javascript:alert(1))",
    "<img src=x onerror=alert(1)>",
    "https://attacker.invalid/auto-link",
    "$$\\frac{1}{2}$$ :smile: **bold** `code` # head",
])
def test_markdown_escape_leaves_no_active_markdown_syntax(hostile):
    out = md_escape(hostile)
    assert _unescaped(out) == hostile                    # same text shown to the user
    for ch in "!()[]<>*`#$:":                            # every active char is escaped
        for i, c in enumerate(out):
            if c == ch:
                assert out[i - 1] == "\\" , f"unescaped {ch!r} in {out!r}"


def test_markdown_escape_accepts_non_strings():
    assert md_escape(None) == "None" and md_escape(12) == "12"


def test_field_markdown_escapes_every_dictionary_supplied_string():
    field = {"field_name": "![a](http://x.invalid/i.png)", "data_type": "<b>int</b>",
             "description": "see [here](http://x.invalid)", "required": True,
             "min_value": "1", "max_value": "[2]", "allowed_values": ["![y](http://y.invalid)", 3]}
    out = field_markdown(field, detailed=True)
    assert "![" not in out.replace("\\![", "")
    assert "](http" not in out.replace("\\](http", "")
    assert "*[Required]*" in out


@pytest.mark.parametrize("field", [{}, {"field_name": 5}, {"description": 7, "allowed_values": "abc"},
                                   {"min_value": None, "max_value": 3, "description": ["x"]}])
def test_field_markdown_never_crashes_on_odd_dictionary_entries(field):
    assert isinstance(field_markdown(field), str)
    assert isinstance(field_markdown(field, detailed=True), str)


@pytest.mark.parametrize("bad", [[], "text", 5, {"rules": []}, {"fields": {}}, {"fields": ["a"]},
                                 {"fields": [1, 2]}])
def test_a_malformed_json_dictionary_is_rejected_with_a_clear_error(bad):
    with pytest.raises(ValueError):
        validate_dictionary_shape(bad)


def test_a_well_formed_json_dictionary_is_accepted():
    good = {"rules": {"age": {"min": 1}}, "fields": [{"field_name": "age"}]}
    assert validate_dictionary_shape(good) is good


def test_no_script_element_or_html_component_is_built_from_upload_data():
    assert "log_to_browser_console" not in SRC
    assert "<script" not in SRC.lower()
    assert "components.html" not in SRC


def test_web_app_has_no_print_calls_that_could_write_file_names_to_container_logs():
    calls = [n for n in ast.walk(ast.parse(SRC))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "print"]
    assert calls == []


def test_file_fingerprint_hides_the_file_name():
    fp = file_fingerprint("SITE-017_participants.csv", 1234)
    assert "SITE-017" not in fp and "participants" not in fp
    assert "ext=csv" in fp and "size=1234" in fp and re.search(r"name_sha256=[0-9a-f]{8}", fp)


def test_file_fingerprint_is_stable_and_distinguishes_names():
    assert file_fingerprint("a.csv", 1) == file_fingerprint("a.csv", 1)
    assert file_fingerprint("a.csv", 1) != file_fingerprint("b.csv", 1)


def test_failed_analysis_logs_only_the_error_class_not_the_upstream_text(caplog):
    """The log call in the Analyze handler must not interpolate the exception text."""
    assert 'message=%r", type(e).__name__, str(e)' not in SRC
    tree = ast.parse(SRC)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("error", "warning", "info")
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "logger"):
            fmt = ast.unparse(node.args[0]) if node.args else ""
            if "analysis_failed" in fmt:
                assert "str(e)" not in ast.unparse(node)


def test_every_user_visible_error_that_embeds_a_value_escapes_it():
    # st.error/st.warning f-strings may only embed untrusted values via md_escape.
    for m in re.finditer(r"st\.(?:error|warning)\(f\"([^\n]*)", SRC):
        text = m.group(1)
        for expr in re.findall(r"\{([^}]*)\}", text):
            if re.search(r"\b(e|issue|rec|dict_file|uploaded_file)\b", expr):
                assert "md_escape" in expr or "type(e).__name__" in expr or "size" in expr or "_escape" in expr, text
