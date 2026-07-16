"""
Regression tests for web_app's dictionary session-state resolution logic.

Bug: selecting a demo dictionary (e.g. REDCap), analyzing, then resetting
the "Or load demo dictionary:" selectbox back to "None" left
`st.session_state.dictionary` untouched - the demo dictionary's dict-driven
critical issues kept showing up on every subsequent analysis because nothing
ever cleared the stale state. Root cause: the selectbox handling in
web_app.py had `if demo_dict != "None": st.session_state.dictionary = {...}`
with no `else` branch to clear it back to None.

The fix extracts the decision of "what should st.session_state.dictionary be
this render pass" into a pure function, `resolve_effective_dictionary`
(web_app.py, module scope, defined near the top of the file above
`st.set_page_config`), so it's directly testable without a Streamlit
runtime - same import-safety pattern as tests/test_upload.py and
tests/test_refactor.py, which both `from web_app import ...` directly.
"""

from web_app import resolve_effective_dictionary


REDCAP_DICT = {
    "source": "Demo Dictionary",
    "filename": "REDCap - Clinical (synthetic, with branching logic)",
    "rules": {"age": {"type": "int", "min": 18, "max": 85}},
}

WESTERN_DICT = {
    "source": "Demo Dictionary",
    "filename": "Western",
    "rules": {"employee_id": {"type": "int"}},
}

UPLOADED_DICT = {
    "source": "CSV",
    "filename": "my_uploaded_dict.csv",
    "rules": {"custom_field": {"type": "str"}},
}


class TestDemoDictionarySelectedThenAnalyze:
    """(a) Selecting a demo dictionary makes it the effective dictionary."""

    def test_demo_dictionary_selection_is_used(self):
        result = resolve_effective_dictionary(
            demo_dict_selection=REDCAP_DICT["filename"],
            demo_dictionary_built=REDCAP_DICT,
            dict_file_uploaded=False,
            previous_session_dictionary=None,
        )
        assert result == REDCAP_DICT

    def test_demo_dictionary_selection_replaces_previous_none(self):
        result = resolve_effective_dictionary(
            demo_dict_selection=REDCAP_DICT["filename"],
            demo_dictionary_built=REDCAP_DICT,
            dict_file_uploaded=False,
            previous_session_dictionary=None,
        )
        assert result is not None
        assert result["rules"] == REDCAP_DICT["rules"]


class TestDemoDictionaryResetToNone:
    """(b) Resetting the demo selector to "None" (no upload present) clears
    the dictionary - this is the exact bug scenario from the report."""

    def test_reset_to_none_clears_stale_demo_dictionary(self):
        # Simulate: REDCap was previously loaded and is sitting in session
        # state from a prior render pass (this is the stale state that used
        # to linger forever before this fix).
        result = resolve_effective_dictionary(
            demo_dict_selection="None",
            demo_dictionary_built=None,
            dict_file_uploaded=False,
            previous_session_dictionary=REDCAP_DICT,
        )
        assert result is None

    def test_reset_to_none_with_no_prior_dictionary_stays_none(self):
        result = resolve_effective_dictionary(
            demo_dict_selection="None",
            demo_dictionary_built=None,
            dict_file_uploaded=False,
            previous_session_dictionary=None,
        )
        assert result is None


class TestSwitchingBetweenDemoDictionaries:
    """(c) Switching from one demo dictionary to another fully replaces the
    first - no merged/leftover rules."""

    def test_switch_from_redcap_to_western_fully_replaces(self):
        # First render pass: REDCap selected.
        after_redcap = resolve_effective_dictionary(
            demo_dict_selection=REDCAP_DICT["filename"],
            demo_dictionary_built=REDCAP_DICT,
            dict_file_uploaded=False,
            previous_session_dictionary=None,
        )
        assert after_redcap == REDCAP_DICT

        # Second render pass: selector switched to Western. The caller is
        # responsible for building the freshly-parsed Western dict for this
        # pass (web_app.py does this every render); previous_session_dictionary
        # is what the first pass left behind (REDCap).
        after_western = resolve_effective_dictionary(
            demo_dict_selection=WESTERN_DICT["filename"],
            demo_dictionary_built=WESTERN_DICT,
            dict_file_uploaded=False,
            previous_session_dictionary=after_redcap,
        )

        assert after_western == WESTERN_DICT
        # No merge/append: none of REDCap's rule keys leak into the result.
        assert set(after_western["rules"]).isdisjoint(set(REDCAP_DICT["rules"]))
        assert after_western["filename"] == WESTERN_DICT["filename"]


class TestUploadedDictionaryTakesPrecedenceOverDemoNone:
    """(d) An uploaded dictionary file should not be clobbered by "None" in
    the demo selector.

    Design decision: file upload is treated as the higher-precedence,
    intentional user action. The demo selectbox defaults to "None" whenever
    no demo entry has been explicitly chosen, so if resetting the demo
    selector to "None" always cleared the dictionary, uploading a file would
    be immediately wiped out on the very next rerun (Streamlit reruns the
    whole script on every widget interaction). `dict_file_uploaded` reflects
    whether a file is currently present in the uploader widget for this
    render pass, so as long as the uploaded file stays in the widget, it
    wins regardless of the demo selector's value.
    """

    def test_uploaded_dictionary_wins_when_demo_selector_is_none(self):
        result = resolve_effective_dictionary(
            demo_dict_selection="None",
            demo_dictionary_built=None,
            dict_file_uploaded=True,
            previous_session_dictionary=UPLOADED_DICT,
        )
        assert result == UPLOADED_DICT

    def test_uploaded_dictionary_wins_even_if_demo_selector_has_stale_value(self):
        # Defensive: even if the demo selector widget somehow still reports
        # a non-"None" value while a file is uploaded, the upload wins per
        # documented precedence (upload > demo selection > None).
        result = resolve_effective_dictionary(
            demo_dict_selection=REDCAP_DICT["filename"],
            demo_dictionary_built=REDCAP_DICT,
            dict_file_uploaded=True,
            previous_session_dictionary=UPLOADED_DICT,
        )
        assert result == UPLOADED_DICT

    def test_removing_uploaded_file_and_demo_none_clears_dictionary(self):
        # File removed from uploader (dict_file_uploaded=False) and demo
        # selector is "None" -> dictionary should clear, not keep the old
        # uploaded dict around either.
        result = resolve_effective_dictionary(
            demo_dict_selection="None",
            demo_dictionary_built=None,
            dict_file_uploaded=False,
            previous_session_dictionary=UPLOADED_DICT,
        )
        assert result is None


class TestReturnValueIsNotMutatedCopy:
    """Sanity check: the function is a pure selector, not a merger - it
    returns one of its inputs unchanged rather than constructing a new
    merged dict."""

    def test_demo_dictionary_object_identity_preserved(self):
        result = resolve_effective_dictionary(
            demo_dict_selection=REDCAP_DICT["filename"],
            demo_dictionary_built=REDCAP_DICT,
            dict_file_uploaded=False,
            previous_session_dictionary=None,
        )
        assert result is REDCAP_DICT

    def test_uploaded_dictionary_object_identity_preserved(self):
        result = resolve_effective_dictionary(
            demo_dict_selection="None",
            demo_dictionary_built=None,
            dict_file_uploaded=True,
            previous_session_dictionary=UPLOADED_DICT,
        )
        assert result is UPLOADED_DICT
