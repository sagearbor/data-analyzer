"""
Tests for the demo dictionary/dataset catalog (demo_dictionaries.py,
web_app.load_demo_data), focused on the REDCap synthetic clinical demo added
to both "load demo dictionary" and "load demo data" dropdowns.

These are pure data/parsing assertions - no Streamlit runtime is required
(same pattern as tests/test_upload.py and tests/test_refactor.py, which
import from web_app directly).
"""

import io

import pandas as pd
import pytest

from demo_dictionaries import DEMO_DICTIONARIES, get_demo_dictionary
from web_app import load_demo_data


REDCAP_DICT_LABEL = "REDCap - Clinical (synthetic, with branching logic)"


class TestRedcapDemoDictionary:
    """The REDCap demo dictionary entry loads and parses like the others."""

    def test_redcap_dictionary_registered(self):
        assert REDCAP_DICT_LABEL in DEMO_DICTIONARIES

    def test_redcap_dictionary_not_empty(self):
        csv_string = get_demo_dictionary(REDCAP_DICT_LABEL)
        assert csv_string
        assert "Column" in csv_string

    def test_redcap_dictionary_parses_as_csv(self):
        csv_string = get_demo_dictionary(REDCAP_DICT_LABEL)
        df = pd.read_csv(io.StringIO(csv_string))
        # Generic demo-dictionary format used by every entry in
        # DEMO_DICTIONARIES (see web_app.py's "load demo dictionary" parser).
        assert list(df.columns) == [
            "Column", "Type", "Required", "Min", "Max", "Description", "Allowed_Values"
        ]

    def test_redcap_dictionary_has_16_fields(self):
        csv_string = get_demo_dictionary(REDCAP_DICT_LABEL)
        df = pd.read_csv(io.StringIO(csv_string))
        assert len(df) == 16

    def test_redcap_dictionary_field_names_match_source(self):
        """Field names should match tests/test_data/dictionaries/synthetic/
        redcap_clinical_with_logic.csv (the original synthetic REDCap CSV
        this demo entry is derived from)."""
        csv_string = get_demo_dictionary(REDCAP_DICT_LABEL)
        df = pd.read_csv(io.StringIO(csv_string))
        expected_fields = {
            "subject_id", "age", "gender", "pregnant", "weeks_pregnant",
            "due_date", "treatment_arm", "dosage_mg", "placebo_type",
            "diabetes", "diabetes_type", "insulin_dependent", "lab_glucose",
            "adverse_event", "ae_description", "ae_severity",
        }
        assert set(df["Column"]) == expected_fields

    def test_redcap_dictionary_preserves_branching_logic_context(self):
        """Branching-logic conditions from the source REDCap CSV should be
        preserved (in Description) even though this quick-load path doesn't
        mechanically enforce them - see comments in demo_dictionaries.py."""
        csv_string = get_demo_dictionary(REDCAP_DICT_LABEL)
        assert "Branching" in csv_string
        assert "gender='Female'" in csv_string

    def test_redcap_dictionary_loads_via_ui_quick_parser(self):
        """Exercise the same parsing logic used by the "load demo dictionary"
        dropdown handler in web_app.py (~line 1289): generic Column/Type/
        Required/Min/Max/Description/Allowed_Values header -> rules dict."""
        csv_string = get_demo_dictionary(REDCAP_DICT_LABEL)
        df = pd.read_csv(io.StringIO(csv_string))
        rules = {}
        for _, row in df.iterrows():
            field_name = row.get('Column')
            if field_name:
                rule = {}
                rule['type'] = str(row.get('Type'))
                rule['min'] = row.get('Min')
                rule['max'] = row.get('Max')
                rule['required'] = row.get('Required')
                allowed = row.get('Allowed_Values')
                if allowed and not pd.isna(allowed):
                    rule['allowed_values'] = [v.strip() for v in str(allowed).split(',')]
                rules[field_name] = rule

        assert len(rules) == 16
        assert rules['gender']['allowed_values'] == ['Male', 'Female', 'Other']
        assert rules['age']['min'] == 18
        assert rules['age']['max'] == 85


class TestRedcapDemoDataset:
    """The matching synthetic REDCap dataset loads and contains the expected
    quality issues, including a branching-logic violation."""

    @pytest.fixture()
    def df(self):
        return load_demo_data('redcap_clinical')

    def test_loads_dataframe(self, df):
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 15
        assert len(df) <= 25

    def test_columns_match_dictionary_fields(self, df):
        csv_string = get_demo_dictionary(REDCAP_DICT_LABEL)
        dict_df = pd.read_csv(io.StringIO(csv_string))
        expected_fields = set(dict_df["Column"])
        assert expected_fields.issubset(set(df.columns))

    def test_contains_age_out_of_range(self, df):
        # Dictionary max for age is 85
        assert (df['age'] > 85).any()

    def test_contains_invalid_gender_value(self, df):
        allowed = {'Male', 'Female', 'Other'}
        assert (~df['gender'].isin(allowed)).any()

    def test_contains_missing_required_lab_glucose_for_diabetic(self, df):
        diabetic_missing_glucose = df[(df['diabetes'] == 'Yes') & (df['lab_glucose'].isna())]
        assert len(diabetic_missing_glucose) >= 1

    def test_contains_dosage_out_of_range(self, df):
        # Dictionary max for dosage_mg is 500
        non_null_dosage = df['dosage_mg'].dropna()
        assert (non_null_dosage > 500).any()

    def test_contains_branching_logic_violation(self, df):
        """At least one row must have pregnant='Yes' while gender != 'Female'
        - a violation of the dictionary's branching logic
        ([pregnant] shown only if [gender]='Female')."""
        violations = df[(df['pregnant'] == 'Yes') & (df['gender'] != 'Female')]
        assert len(violations) >= 1

    def test_demo_selector_dropdown_maps_to_dataset(self):
        """The label used in the 'load demo data' dropdown must map to the
        'redcap_clinical' key understood by load_demo_data (see web_app.py's
        dataset_map for the "Or load demo data" selectbox)."""
        df = load_demo_data('redcap_clinical')
        assert df is not load_demo_data('western')
        assert 'subject_id' in df.columns
