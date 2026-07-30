"""
Tests for the Logic Validation Engine (Option B — structured, non-executable rules).

Tests cover:
- Condition tree: Compare / And / Or / Not / Const evaluation (vectorized)
- ConditionalRule / LogicViolation dataclasses + serialization round-trips
- REDCap branching-logic parser (text -> structured Condition, never code)
- RuleExtractor parsing of REDCap, FHIR, business-rule, and LLM logic
- LogicValidator end-to-end validation
- SECURITY: malicious data-dictionary input must NEVER execute code
  (regression for the exec()/pd.read_pickle RCE that this rewrite eliminates)
"""
import os
import sys
import json
import pickle
import pytest
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.logic_engine import (
    Condition, Compare, And, Or, Not, Const,
    ConditionalRule, LogicViolation, LogicValidator, RuleExtractor,
    parse_redcap_expression,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_rule():
    return ConditionalRule(
        rule_id="pregnancy_gender_check",
        rule_type="skip_if",
        condition=Compare("gender", "in", ["male", "m", "1"]),
        action="must_be_blank",
        affected_fields=["pregnant", "weeks_pregnant"],
        description="Skip pregnancy questions for male subjects",
        source="[gender]='1'",
        severity="error",
        confidence=1.0,
    )


@pytest.fixture
def sample_rules():
    return [
        ConditionalRule(
            rule_id="pregnancy_male_skip",
            rule_type="skip_if",
            condition=Compare("gender", "in", ["male", "m", "1"]),
            action="must_be_blank",
            affected_fields=["pregnant"],
            description="Skip pregnancy for males",
            severity="error",
            confidence=1.0,
        ),
        ConditionalRule(
            rule_id="dose_age_check",
            rule_type="required_if",
            condition=Compare("age", "ge", 18),
            action="must_be_filled",
            affected_fields=["consent_signed"],
            description="Consent required for adults",
            severity="error",
            confidence=1.0,
        ),
    ]


@pytest.fixture
def sample_df_valid():
    return pd.DataFrame({
        'subject_id': ['001', '002', '003'],
        'gender': ['Male', 'Female', 'Female'],
        'age': [35, 28, 45],
        'pregnant': [None, 'Yes', 'No'],       # Male has blank, as expected
        'weeks_pregnant': [None, 12, None],
        'consent_signed': ['Yes', 'Yes', 'Yes'],
    })


@pytest.fixture
def sample_df_violations():
    return pd.DataFrame({
        'subject_id': ['001', '002'],
        'gender': ['Male', 'Male'],
        'age': [35, 28],
        'pregnant': ['Yes', 'No'],             # Males should be blank!
        'weeks_pregnant': [12, None],
    })


@pytest.fixture
def sample_fields_redcap():
    return [
        {'field_name': 'subject_id', 'data_type': 'str', 'required': True},
        {'field_name': 'gender', 'data_type': 'int', 'allowed_values': [1, 2]},
        {'field_name': 'pregnant', 'data_type': 'int',
         'branching_logic': "[gender]='2'", 'allowed_values': [0, 1]},
        {'field_name': 'weeks_pregnant', 'data_type': 'int',
         'branching_logic': "[pregnant]='1'"},
    ]


# ============================================================================
# CONDITION TREE
# ============================================================================

class TestConditionEvaluation:
    def test_compare_eq_string_normalized(self):
        df = pd.DataFrame({'g': ['Male', 'female', 'MALE']})
        res = Compare('g', 'eq', 'male').evaluate(df)
        assert list(res) == [True, False, True]

    def test_compare_in_list(self):
        df = pd.DataFrame({'g': ['1', 'm', 'x', 'MALE']})
        res = Compare('g', 'in', ['male', 'm', '1']).evaluate(df)
        assert list(res) == [True, True, False, True]

    def test_compare_numeric(self):
        df = pd.DataFrame({'age': [10, 18, 25, None, 'abc']})
        res = Compare('age', 'ge', 18).evaluate(df)
        assert list(res) == [False, True, True, False, False]

    def test_compare_missing_column_is_false(self):
        df = pd.DataFrame({'a': [1, 2]})
        assert list(Compare('nonexistent', 'eq', 'x').evaluate(df)) == [False, False]

    def test_is_blank_and_not_blank(self):
        df = pd.DataFrame({'f': ['x', None, '   ', '']})
        assert list(Compare('f', 'is_blank').evaluate(df)) == [False, True, True, True]
        assert list(Compare('f', 'not_blank').evaluate(df)) == [True, False, False, False]

    def test_contains(self):
        df = pd.DataFrame({'notes': ['Diabetic patient', 'healthy', None]})
        assert list(Compare('notes', 'contains', 'diabet').evaluate(df)) == [True, False, False]

    def test_and_or_not(self):
        df = pd.DataFrame({'g': ['m', 'm', 'f'], 'age': [20, 10, 30]})
        c = And([Compare('g', 'eq', 'm'), Compare('age', 'ge', 18)])
        assert list(c.evaluate(df)) == [True, False, False]
        c2 = Or([Compare('g', 'eq', 'f'), Compare('age', 'ge', 18)])
        assert list(c2.evaluate(df)) == [True, False, True]
        assert list(Not(Compare('g', 'eq', 'm')).evaluate(df)) == [False, False, True]

    def test_const(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        assert list(Const(True).evaluate(df)) == [True, True, True]
        assert list(Const(False).evaluate(df)) == [False, False, False]

    def test_invalid_op_rejected(self):
        with pytest.raises(ValueError):
            Compare('f', 'totally_bogus', 1)


class TestConditionSerialization:
    def test_roundtrip(self):
        cond = And([
            Compare('gender', 'eq', '2'),
            Or([Compare('age', 'ge', 18), Not(Compare('x', 'is_blank'))]),
        ])
        restored = Condition.from_dict(cond.to_dict())
        df = pd.DataFrame({'gender': ['2', '2'], 'age': [20, 5], 'x': ['v', None]})
        assert list(restored.evaluate(df)) == list(cond.evaluate(df))

    def test_legacy_string_condition_is_neutralized(self):
        # An old cache entry stored a Python expression string here; it must
        # become Const(False) and NEVER be executed.
        cond = Condition.from_dict("os.system('rm -rf /')")
        assert isinstance(cond, Const) and cond.value is False

    def test_unknown_type_is_neutralized(self):
        assert isinstance(Condition.from_dict({'type': 'weird'}), Const)
        assert isinstance(Condition.from_dict(None), Const)


# ============================================================================
# DATACLASSES
# ============================================================================

class TestConditionalRule:
    def test_creation(self, sample_rule):
        assert sample_rule.rule_id == "pregnancy_gender_check"
        assert isinstance(sample_rule.condition, Condition)

    def test_string_condition_coerced_not_executed(self):
        rule = ConditionalRule(
            rule_id="x", rule_type="skip_if",
            condition="__import__('os').system('boom')",   # legacy/malicious
            action="must_be_blank", affected_fields=["f"], description="d",
        )
        assert isinstance(rule.condition, Const) and rule.condition.value is False

    def test_to_from_dict_roundtrip(self, sample_rule):
        restored = ConditionalRule.from_dict(sample_rule.to_dict())
        assert restored.rule_id == sample_rule.rule_id
        assert restored.condition.to_dict() == sample_rule.condition.to_dict()

    def test_str_condition_is_readable(self, sample_rule):
        # test_dictionary.py prints rule.condition — must render, not crash.
        assert "gender" in str(sample_rule.condition).lower()


class TestLogicViolation:
    def test_to_dict(self):
        v = LogicViolation(
            rule_id="r", rule_description="d", row_index=0,
            affected_fields=["f"], actual_values={"f": "x"},
            expected_behavior="blank", severity="error",
        )
        d = v.to_dict()
        assert d["rule_id"] == "r" and d["row_index"] == 0


# ============================================================================
# REDCap PARSER
# ============================================================================

class TestRedcapParser:
    def test_simple_equality(self):
        df = pd.DataFrame({'gender': ['2', '1']})
        cond = parse_redcap_expression("[gender]='2'")
        assert list(cond.evaluate(df)) == [True, False]

    def test_and(self):
        cond = parse_redcap_expression("[diabetes]='1' and [age] >= 18")
        df = pd.DataFrame({'diabetes': ['1', '1', '0'], 'age': [20, 10, 20]})
        assert list(cond.evaluate(df)) == [True, False, False]

    def test_or(self):
        cond = parse_redcap_expression("[status]='complete' or [status]='partial'")
        df = pd.DataFrame({'status': ['complete', 'partial', 'none']})
        assert list(cond.evaluate(df)) == [True, True, False]

    def test_parentheses(self):
        cond = parse_redcap_expression("([a]='1' or [a]='2') and [b]>=5")
        df = pd.DataFrame({'a': ['1', '2', '3'], 'b': [9, 1, 9]})
        assert list(cond.evaluate(df)) == [True, False, False]

    def test_unparseable_is_const_false(self):
        # Not valid REDCap grammar -> fail closed, no execution.
        cond = parse_redcap_expression("pd.read_pickle('/tmp/evil.pkl')")
        assert isinstance(cond, Const) and cond.value is False

    def test_empty(self):
        assert isinstance(parse_redcap_expression(""), Const)


# ============================================================================
# RULE EXTRACTOR
# ============================================================================

class TestRuleExtractor:
    def test_empty_input(self):
        assert RuleExtractor().extract_rules_from_fields([]) == []

    def test_redcap_branching(self):
        rule = RuleExtractor()._parse_redcap_branching("pregnant", "[gender]='2'")
        assert rule is not None
        assert rule.affected_fields == ["pregnant"]
        # show_if -> blank when NOT female
        df = pd.DataFrame({'gender': ['1'], 'pregnant': ['1']})
        assert list(rule.condition.evaluate(df)) == [True]   # condition "must be blank" fires for male

    def test_extract_from_fields(self, sample_fields_redcap):
        rules = RuleExtractor().extract_rules_from_fields(sample_fields_redcap)
        rule_ids = [r.rule_id for r in rules]
        assert any("pregnant" in rid for rid in rule_ids)
        assert any("weeks_pregnant" in rid for rid in rule_ids)

    def test_business_rule_male_skip(self):
        rule = RuleExtractor()._parse_business_rule("pregnant", "If male, skip this field")
        assert rule is not None and rule.action == "must_be_blank"
        df = pd.DataFrame({'gender': ['male', 'female']})
        assert list(rule.condition.evaluate(df)) == [True, False]

    def test_business_rule_female_required(self):
        rule = RuleExtractor()._parse_business_rule("preg_test", "Required for female subjects")
        assert rule is not None and rule.action == "must_be_filled"

    def test_fhir_enable_when(self):
        ew = {"question": "gender", "operator": "=", "answerCoding": {"code": "female"}}
        rule = RuleExtractor()._parse_fhir_enable_when("pregnant", ew)
        assert rule is not None
        df = pd.DataFrame({'gender': ['female'], 'pregnant': ['1']})
        # enabled when female -> Not(enabled) is False for female
        assert list(rule.condition.evaluate(df)) == [False]

    def test_llm_rules(self):
        field = {
            'field_name': 'pregnant',
            'conditional_rules': [
                {'rule_type': 'skip_if', 'condition_text': 'gender is male',
                 'action': 'must_be_blank'},
            ],
        }
        rules = RuleExtractor()._extract_llm_rules(field)
        assert len(rules) == 1
        df = pd.DataFrame({'gender': ['male', 'female']})
        assert list(rules[0].condition.evaluate(df)) == [True, False]

    def test_natural_language_variants(self):
        ex = RuleExtractor()
        df = pd.DataFrame({'age': [20, 10], 'status': ['active', 'closed'],
                           'notes': ['has diabetes', 'none']})
        assert list(ex._convert_natural_language_condition("age >= 18").evaluate(df)) == [True, False]
        assert list(ex._convert_natural_language_condition("status is not active").evaluate(df)) == [False, True]
        assert list(ex._convert_natural_language_condition("notes contains diabetes").evaluate(df)) == [True, False]


# ============================================================================
# LOGIC VALIDATOR (end to end)
# ============================================================================

class TestLogicValidator:
    def test_no_rules(self):
        assert LogicValidator().validate([], pd.DataFrame({'a': [1]})) == []

    def test_empty_df(self, sample_rules):
        assert LogicValidator().validate(sample_rules, pd.DataFrame()) == []

    def test_valid_data_no_violations(self, sample_rules, sample_df_valid):
        violations = LogicValidator().validate(sample_rules, sample_df_valid)
        assert violations == []

    def test_violations_detected(self, sample_df_violations):
        rule = ConditionalRule(
            rule_id="preg_male_skip", rule_type="skip_if",
            condition=Compare("gender", "in", ["male", "m", "1"]),
            action="must_be_blank", affected_fields=["pregnant", "weeks_pregnant"],
            description="Skip pregnancy for males",
        )
        violations = LogicValidator().validate([rule], sample_df_violations)
        # row0: pregnant='Yes' + weeks=12 -> 2 violations; row1: pregnant='No' -> 1
        assert len(violations) == 3
        assert all(v.severity == "error" for v in violations)

    def test_required_if_violation(self):
        rule = ConditionalRule(
            rule_id="consent", rule_type="required_if",
            condition=Compare("age", "ge", 18), action="must_be_filled",
            affected_fields=["consent"], description="Adults need consent",
        )
        df = pd.DataFrame({'age': [25, 10], 'consent': [None, None]})
        violations = LogicValidator().validate([rule], df)
        assert len(violations) == 1 and violations[0].row_index == 0

    def test_one_bad_rule_does_not_abort_others(self, sample_df_violations):
        good = ConditionalRule(
            rule_id="g", rule_type="skip_if",
            condition=Compare("gender", "eq", "Male"), action="must_be_blank",
            affected_fields=["pregnant"], description="d",
        )
        violations = LogicValidator().validate([good], sample_df_violations)
        assert len(violations) >= 1


# ============================================================================
# SECURITY REGRESSION  (the whole point of Option B)
# ============================================================================

class TestSecurityNoCodeExecution:
    """
    These tests reproduce the original RCE vector (a malicious data dictionary
    whose 'logic' is really a code payload) and assert it CANNOT execute.
    """

    def test_pickle_gadget_in_branching_logic_does_not_execute(self, tmp_path):
        marker = tmp_path / "PWNED.txt"
        evil_pkl = tmp_path / "evil.pkl"

        class Evil:
            def __reduce__(self):
                return (os.system, (f"echo pwned > {marker}",))

        with open(evil_pkl, "wb") as fh:
            pickle.dump(Evil(), fh)

        # The exact shape of the original exploit: dictionary field whose
        # branching_logic is a pd.read_pickle() call.
        fields = [{"field_name": "dose", "branching_logic": f"pd.read_pickle('{evil_pkl}')"}]
        rules = RuleExtractor().extract_rules_from_fields(fields, format_type="REDCap")

        df = pd.DataFrame({"dose": [1, 2]})
        # Must run without executing the payload.
        LogicValidator().validate(rules, df)

        assert not marker.exists(), "SECURITY: dictionary payload executed code!"
        # The condition must have been neutralized to Const(False) (inside Not()).
        assert rules and isinstance(rules[0].condition, Not)
        assert isinstance(rules[0].condition.operand, Const)
        assert rules[0].condition.operand.value is False

    def test_os_system_in_llm_condition_does_not_execute(self, tmp_path):
        marker = tmp_path / "PWNED2.txt"
        payload = f"__import__('os').system('echo x > {marker}')"
        field = {
            "field_name": "f",
            "conditional_rules": [
                {"rule_type": "skip_if", "condition_text": payload, "action": "must_be_blank"},
            ],
        }
        rules = RuleExtractor().extract_rules_from_fields([field], format_type="Custom")
        LogicValidator().validate(rules, pd.DataFrame({"f": ["x", "y"]}))
        assert not marker.exists(), "SECURITY: LLM condition payload executed code!"

    def test_no_exec_eval_in_module(self):
        # Belt-and-suspenders: the module must not contain an exec()/eval()
        # call path anymore.
        import src.logic_engine as mod
        src = open(mod.__file__).read()
        assert "exec(" not in src
        assert "eval(" not in src

    def test_malformed_field_does_not_abort_other_fields(self):
        # A non-string (or otherwise malformed) branching_logic on one field must
        # NOT abort rule extraction for the other, legitimate fields — otherwise
        # one bad dictionary entry silently disables all logic validation.
        fields = [
            {"field_name": "legit_a", "branching_logic": "[gender]='2'"},
            {"field_name": "evil", "branching_logic": 123},          # non-string
            {"field_name": "legit_b", "branching_logic": "[age]>=18"},
            {"field_name": "also_bad", "branching_logic": ["not", "a", "string"]},
        ]
        rules = RuleExtractor().extract_rules_from_fields(fields, format_type="REDCap")
        rule_ids = [r.rule_id for r in rules]
        assert any("legit_a" in rid for rid in rule_ids)
        assert any("legit_b" in rid for rid in rule_ids)

    def test_non_dict_field_is_skipped(self):
        rules = RuleExtractor().extract_rules_from_fields(
            ["not a dict", None, {"field_name": "ok", "branching_logic": "[g]='1'"}],
            format_type="REDCap",
        )
        assert any("ok" in r.rule_id for r in rules)

    def test_malicious_condition_via_from_dict_is_inert(self):
        rule = ConditionalRule.from_dict({
            "rule_id": "x", "rule_type": "skip_if",
            "condition": "open('/etc/passwd').read()",   # legacy string
            "action": "must_be_blank", "affected_fields": ["f"],
            "description": "d",
        })
        assert isinstance(rule.condition, Const) and rule.condition.value is False
