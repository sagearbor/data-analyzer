"""
Test LLM-based Conditional Logic Extraction

This test demonstrates the new capability to extract conditional rules
from ANY dictionary format using the LLM.
"""

import pytest
from src.logic_engine import RuleExtractor, ConditionalRule


class TestLLMConditionalExtraction:
    """Test LLM-extracted conditional logic conversion"""

    def test_convert_natural_language_gender_male(self):
        """Test conversion of 'gender is male' pattern"""
        extractor = RuleExtractor()

        result = extractor._convert_natural_language_condition("gender is male")

        assert "row.get('gender'" in result
        assert "['male', 'm', '1']" in result
        assert result == "str(row.get('gender', '')).lower() in ['male', 'm', '1']"

    def test_convert_natural_language_gender_female(self):
        """Test conversion of 'gender is female' pattern"""
        extractor = RuleExtractor()

        result = extractor._convert_natural_language_condition("gender is female")

        assert "row.get('gender'" in result
        assert "['female', 'f', '2']" in result

    def test_convert_natural_language_age_greater_than(self):
        """Test conversion of 'age >= 18' pattern"""
        extractor = RuleExtractor()

        result = extractor._convert_natural_language_condition("age >= 18")

        assert "int(row.get('age', 0)) >= 18" == result

    def test_convert_natural_language_pregnant_yes(self):
        """Test conversion of 'pregnant is yes' pattern"""
        extractor = RuleExtractor()

        result = extractor._convert_natural_language_condition("pregnant is yes")

        assert "row.get('pregnant'" in result
        assert "['yes', 'y', '1', 'true']" in result

    def test_convert_natural_language_custom_value(self):
        """Test conversion of custom field/value pattern"""
        extractor = RuleExtractor()

        result = extractor._convert_natural_language_condition("treatment_arm is control")

        assert "row.get('treatment_arm'" in result
        assert "'control'" in result

    def test_convert_natural_language_contains(self):
        """Test conversion of 'contains' pattern"""
        extractor = RuleExtractor()

        result = extractor._convert_natural_language_condition("diagnosis contains cancer")

        assert "'cancer'" in result
        assert "row.get('diagnosis'" in result
        assert "in str" in result

    def test_convert_natural_language_invalid_returns_false(self):
        """Test that invalid patterns return False for safety"""
        extractor = RuleExtractor()

        result = extractor._convert_natural_language_condition("some random text")

        assert result == "False"

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
        assert 'gender' in rules[0].condition

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
        """Test various comparison operators"""
        extractor = RuleExtractor()

        # Greater than
        assert "int(row.get('age', 0)) > 65" == extractor._convert_natural_language_condition("age > 65")

        # Less than
        assert "int(row.get('score', 0)) < 50" == extractor._convert_natural_language_condition("score < 50")

        # Less than or equal
        assert "int(row.get('count', 0)) <= 10" == extractor._convert_natural_language_condition("count <= 10")

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
