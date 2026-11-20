"""Rule coverage is explicit: unsupported / failed rules are reported, never silently 'clean' (ADP-82).

Synthetic data only. The closed, non-executing interpreter is unchanged.
"""
import logging

import pandas as pd
import pytest

from src.logic_engine import (
    Compare, ConditionalRule, Condition, Const, LogicValidator, RuleExtractor, parse_redcap_expression,
)

SECRET_TEXT = "[zz_secret_marker_field] ~~ not a valid expression"


@pytest.fixture
def df():
    return pd.DataFrame({"gender": ["Male", "Female", "Male"], "pregnant": ["1", "", ""]})


def _good_rule():
    return ConditionalRule(
        rule_id="good", rule_type="skip_if", condition=Compare("gender", "eq", "Male"),
        action="must_be_blank", affected_fields=["pregnant"], description="d")


class _Boom(Condition):
    def evaluate(self, df):
        raise RuntimeError("synthetic failure containing " + SECRET_TEXT)

    def to_dict(self):
        return {}


def test_unparseable_branching_logic_is_reported_as_unsupported_not_clean(df):
    rule = RuleExtractor()._parse_redcap_branching("pregnant", SECRET_TEXT)
    violations, cov = LogicValidator().validate_with_coverage([rule], df)
    assert violations == []
    assert cov.complete is False
    assert cov.unsupported == [{"rule_id": "pregnant_show_if", "reason": "unsupported_condition"}]
    assert cov.evaluated == 0 and cov.summary() == "1 of 1 rules could not be checked"


def test_supported_and_unsupported_rule_together_give_an_explicit_partial_result(df):
    bad = RuleExtractor()._parse_redcap_branching("pregnant", SECRET_TEXT)
    violations, cov = LogicValidator().validate_with_coverage([_good_rule(), bad], df)
    assert [v.rule_id for v in violations] == ["good"]  # the good rule still ran and found its violation
    assert (cov.total, cov.evaluated, len(cov.unsupported), len(cov.failed)) == (2, 1, 1, 0)


def test_a_rule_that_errors_during_evaluation_is_listed_as_failed_and_others_still_run(df):
    broken = ConditionalRule(rule_id="broken", rule_type="skip_if", condition=_Boom(),
                             action="must_be_blank", affected_fields=["pregnant"], description="d")
    violations, cov = LogicValidator().validate_with_coverage([broken, _good_rule()], df)
    assert [v.rule_id for v in violations] == ["good"]
    assert cov.failed == [{"rule_id": "broken", "reason": "evaluation_error"}]
    assert cov.evaluated == 1


def test_unknown_action_is_reported_as_unsupported(df):
    rule = ConditionalRule(rule_id="va", rule_type="value_if", condition=Compare("gender", "eq", "Male"),
                           action="value_in", affected_fields=["pregnant"], description="d")
    _, cov = LogicValidator().validate_with_coverage([rule], df)
    assert cov.unsupported == [{"rule_id": "va", "reason": "unsupported_action"}]


def test_fully_supported_rules_report_complete_coverage(df):
    _, cov = LogicValidator().validate_with_coverage([_good_rule()], df)
    assert cov.complete and cov.evaluated == 1 and cov.summary() == ""


def test_validate_keeps_returning_a_plain_violation_list_and_exposes_coverage(df):
    validator = LogicValidator()
    bad = RuleExtractor()._parse_redcap_branching("pregnant", SECRET_TEXT)
    result = validator.validate([_good_rule(), bad], df)
    assert isinstance(result, list) and len(result) == 1
    assert validator.last_coverage.not_checked == 1


def test_unsupported_status_survives_a_save_and_reload_of_the_rule(df):
    bad = RuleExtractor()._parse_redcap_branching("pregnant", SECRET_TEXT)
    reloaded = ConditionalRule.from_dict(bad.to_dict())
    _, cov = LogicValidator().validate_with_coverage([reloaded], df)
    assert cov.unsupported and not cov.evaluated


def test_legacy_string_condition_is_reported_as_unsupported_and_never_executed(df):
    rule = ConditionalRule.from_dict({
        "rule_id": "legacy", "rule_type": "skip_if", "condition": "__import__('os').getcwd()",
        "action": "must_be_blank", "affected_fields": ["pregnant"], "description": "d"})
    violations, cov = LogicValidator().validate_with_coverage([rule], df)
    assert violations == [] and cov.unsupported[0]["rule_id"] == "legacy"


def test_a_genuine_constant_false_rule_is_not_flagged_unsupported(df):
    rule = ConditionalRule(rule_id="c", rule_type="skip_if", condition=Const(False),
                           action="must_be_blank", affected_fields=["pregnant"], description="d")
    _, cov = LogicValidator().validate_with_coverage([rule], df)
    assert cov.complete and cov.evaluated == 1


def test_coverage_dict_lists_counts_and_codes_only():
    bad = RuleExtractor()._parse_redcap_branching("pregnant", SECRET_TEXT)
    _, cov = LogicValidator().validate_with_coverage([bad], pd.DataFrame({"pregnant": ["1"]}))
    d = cov.to_dict()
    assert d["rules_total"] == 1 and d["rules_not_checked"] == 1 and d["complete"] is False
    assert "zz_secret_marker_field" not in str(d)


def test_logs_never_contain_the_raw_expression_text(df, caplog):
    caplog.set_level(logging.DEBUG)
    parse_redcap_expression(SECRET_TEXT)
    bad = RuleExtractor()._parse_redcap_branching("pregnant", SECRET_TEXT)
    broken = ConditionalRule(rule_id="broken", rule_type="skip_if", condition=_Boom(),
                             action="must_be_blank", affected_fields=["pregnant"], description="d")
    LogicValidator().validate_with_coverage([bad, broken], df)
    Condition.from_dict("legacy " + SECRET_TEXT)
    assert "zz_secret_marker_field" not in caplog.text


def test_one_bad_rule_does_not_hide_the_good_rules_violations_and_is_counted(df):
    # Strengthened version of test_one_bad_rule_does_not_abort_others (which only used a good rule).
    bad = RuleExtractor()._parse_redcap_branching("pregnant", SECRET_TEXT)
    violations, cov = LogicValidator().validate_with_coverage([bad, _good_rule()], df)
    assert len(violations) == 1 and violations[0].row_index == 0
    assert cov.not_checked == 1 and cov.evaluated == 1
