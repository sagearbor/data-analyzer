"""
Test LLM-based Conditional Logic Extraction

This test demonstrates the new capability to extract conditional rules
from ANY dictionary format using the LLM.
"""

import pytest
import pandas as pd
from src.logic_engine import RuleExtractor, ConditionalRule, Condition, Compare, Const


class TestLLMConditionalExtraction:
    """Test LLM-extracted conditional logic conversion.

    Option B: _convert_natural_language_condition returns a STRUCTURED Condition
    (data, not an executable Python string). Assertions verify the evaluated
    behavior rather than a generated code string.
    """

    def test_convert_natural_language_gender_male(self):
        """Test conversion of 'gender is male' pattern"""
        extractor = RuleExtractor()

        result = extractor._convert_natural_language_condition("gender is male")

        assert isinstance(result, Condition)
        df = pd.DataFrame({'gender': ['male', 'm', '1', 'female']})
        assert list(result.evaluate(df)) == [True, True, True, False]

    def test_convert_natural_language_gender_female(self):
        """Test conversion of 'gender is female' pattern"""
        extractor = RuleExtractor()

        result = extractor._convert_natural_language_condition("gender is female")

        df = pd.DataFrame({'gender': ['female', 'f', '2', 'male']})
        assert list(result.evaluate(df)) == [True, True, True, False]

    def test_convert_natural_language_age_greater_than(self):
        """Test conversion of 'age >= 18' pattern"""
        extractor = RuleExtractor()

        result = extractor._convert_natural_language_condition("age >= 18")

        df = pd.DataFrame({'age': [17, 18, 19]})
        assert list(result.evaluate(df)) == [False, True, True]

    def test_convert_natural_language_pregnant_yes(self):
        """Test conversion of 'pregnant is yes' pattern"""
        extractor = RuleExtractor()

        result = extractor._convert_natural_language_condition("pregnant is yes")

        df = pd.DataFrame({'pregnant': ['yes', 'y', '1', 'true', 'no']})
        assert list(result.evaluate(df)) == [True, True, True, True, False]

    def test_convert_natural_language_custom_value(self):
        """Test conversion of custom field/value pattern"""
        extractor = RuleExtractor()

        result = extractor._convert_natural_language_condition("treatment_arm is control")

        assert isinstance(result, Compare) and result.field == 'treatment_arm'
        df = pd.DataFrame({'treatment_arm': ['control', 'active']})
        assert list(result.evaluate(df)) == [True, False]

    def test_convert_natural_language_contains(self):
        """Test conversion of 'contains' pattern"""
        extractor = RuleExtractor()

        result = extractor._convert_natural_language_condition("diagnosis contains cancer")

        assert isinstance(result, Compare) and result.op == 'contains'
        df = pd.DataFrame({'diagnosis': ['lung cancer', 'healthy']})
        assert list(result.evaluate(df)) == [True, False]

    def test_convert_natural_language_invalid_returns_false(self):
        """Test that invalid patterns fail closed (Const(False)) for safety"""
        extractor = RuleExtractor()

        result = extractor._convert_natural_language_condition("some random text")

        assert isinstance(result, Const) and result.value is False

    def test_extract_llm_rules_from_field(self):
        """Test extraction of LLM rules from field definition"""
        extractor = RuleExtractor()

        field = {
            'field_name': 'pregnancy_status',
            'data_type': 'str',
            'conditional_rules': [
                {
                    'rule_type': 'skip_if',
                    'condition_text': 'gender is male',
                    'action': 'must_be_blank',
                    'affected_fields': ['pregnancy_status']
                }
            ]
        }

        rules = extractor._extract_llm_rules(field)

        assert len(rules) == 1
        assert isinstance(rules[0], ConditionalRule)
        assert rules[0].rule_type == 'skip_if'
        assert rules[0].action == 'must_be_blank'
        assert 'pregnancy_status' in rules[0].affected_fields
        assert 'gender' in str(rules[0].condition).lower()

    def test_extract_llm_rules_multiple_rules(self):
        """Test extraction of multiple LLM rules from single field"""
        extractor = RuleExtractor()

        field = {
            'field_name': 'alcohol_consumption',
            'data_type': 'int',
            'conditional_rules': [
                {
                    'rule_type': 'show_if',
                    'condition_text': 'age >= 18',
                    'action': 'must_be_filled',
                    'affected_fields': ['alcohol_consumption']
                },
                {
                    'rule_type': 'skip_if',
                    'condition_text': 'treatment_arm is control',
                    'action': 'must_be_blank',
                    'affected_fields': ['alcohol_consumption']
                }
            ]
        }

        rules = extractor._extract_llm_rules(field)

        assert len(rules) == 2
        assert all(isinstance(r, ConditionalRule) for r in rules)
        assert rules[0].rule_type == 'show_if'
        assert rules[1].rule_type == 'skip_if'

    def test_extract_rules_prioritizes_llm_over_format(self):
        """Test that LLM rules are extracted first"""
        extractor = RuleExtractor()

        fields = [
            {
                'field_name': 'pregnancy_test',
                'data_type': 'str',
                'conditional_rules': [
                    {
                        'rule_type': 'skip_if',
                        'condition_text': 'gender is male',
                        'action': 'must_be_blank',
                        'affected_fields': ['pregnancy_test']
                    }
                ],
                'business_rules': ['If male, skip this field']
            }
        ]

        # Extract rules - should use LLM rule AND business rule
        rules = extractor.extract_rules_from_fields(fields, format_type="Custom")

        # Should have at least 1 rule (LLM-extracted)
        assert len(rules) >= 1

        # First rule should be LLM-extracted
        llm_rule = rules[0]
        assert 'llm' in llm_rule.rule_id
        assert llm_rule.confidence == 0.85  # LLM confidence

    def test_extract_llm_rules_no_conditional_rules(self):
        """Test that fields without conditional_rules return empty list"""
        extractor = RuleExtractor()

        field = {
            'field_name': 'participant_id',
            'data_type': 'int'
        }

        rules = extractor._extract_llm_rules(field)

        assert rules == []

    def test_extract_llm_rules_handles_malformed_data(self):
        """Test that malformed conditional_rules are handled gracefully"""
        extractor = RuleExtractor()

        field = {
            'field_name': 'test_field',
            'data_type': 'str',
            'conditional_rules': 'not a list'  # Invalid format
        }

        rules = extractor._extract_llm_rules(field)

        assert rules == []

    def test_extract_llm_rules_skips_invalid_rule_objects(self):
        """Test that invalid rule objects in array are skipped"""
        extractor = RuleExtractor()

        field = {
            'field_name': 'test_field',
            'data_type': 'str',
            'conditional_rules': [
                "not a dict",
                {},  # Missing required fields
                {
                    'rule_type': 'skip_if',
                    'condition_text': '',  # Empty condition
                    'action': 'must_be_blank',
                    'affected_fields': ['test_field']
                }
            ]
        }

        rules = extractor._extract_llm_rules(field)

        # Should gracefully skip all invalid rules
        assert rules == []

    def test_comparison_operators(self):
        """Test various comparison operators evaluate correctly"""
        extractor = RuleExtractor()

        gt = extractor._convert_natural_language_condition("age > 65")
        assert list(gt.evaluate(pd.DataFrame({'age': [65, 66]}))) == [False, True]

        lt = extractor._convert_natural_language_condition("score < 50")
        assert list(lt.evaluate(pd.DataFrame({'score': [49, 50]}))) == [True, False]

        le = extractor._convert_natural_language_condition("count <= 10")
        assert list(le.evaluate(pd.DataFrame({'count': [10, 11]}))) == [True, False]

    def test_confidence_scores(self):
        """Test that LLM rules have appropriate confidence scores"""
        extractor = RuleExtractor()

        field = {
            'field_name': 'test_field',
            'data_type': 'str',
            'conditional_rules': [
                {
                    'rule_type': 'skip_if',
                    'condition_text': 'gender is male',
                    'action': 'must_be_blank',
                    'affected_fields': ['test_field']
                }
            ]
        }

        rules = extractor._extract_llm_rules(field)

        # LLM rules should have confidence of 0.85 (slightly lower than explicit dictionary rules)
        assert rules[0].confidence == 0.85

    def test_severity_levels(self):
        """Test that different rule types have appropriate severity levels"""
        extractor = RuleExtractor()

        field = {
            'field_name': 'test_field',
            'data_type': 'str',
            'conditional_rules': [
                {
                    'rule_type': 'skip_if',
                    'condition_text': 'gender is male',
                    'action': 'must_be_blank',
                    'affected_fields': ['test_field']
                },
                {
                    'rule_type': 'show_if',
                    'condition_text': 'age >= 18',
                    'action': 'must_be_filled',
                    'affected_fields': ['test_field']
                }
            ]
        }

        rules = extractor._extract_llm_rules(field)

        # skip_if should be error severity
        assert rules[0].severity == 'error'

        # show_if should be warning severity
        assert rules[1].severity == 'warning'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
