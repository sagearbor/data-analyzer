#!/usr/bin/env python3
"""
Demo: LLM-Based Conditional Logic Extraction

This script demonstrates how the enhanced Logic Validation Engine
can extract conditional rules from ANY dictionary format.

The LLM extracts conditional logic during dictionary parsing, then
the RuleExtractor converts it to executable ConditionalRule objects.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.logic_engine import RuleExtractor, ConditionalRule
import json


def demo_llm_extracted_rules():
    """Demonstrate LLM-extracted conditional rules"""

    print("=" * 80)
    print("DEMO: LLM-Based Conditional Logic Extraction")
    print("=" * 80)
    print()

    # Simulated output from LLM parser
    # This would normally come from LLMDictionaryParser.parse_dictionary()
    fields = [
        {
            "field_name": "pregnancy_status",
            "data_type": "str",
            "description": "Pregnancy status of participant",
            "conditional_rules": [
                {
                    "rule_type": "skip_if",
                    "condition_text": "gender is male",
                    "action": "must_be_blank",
                    "affected_fields": ["pregnancy_status"]
                }
            ]
        },
        {
            "field_name": "alcohol_consumption",
            "data_type": "int",
            "description": "Number of alcoholic drinks per week",
            "conditional_rules": [
                {
                    "rule_type": "show_if",
                    "condition_text": "age >= 18",
                    "action": "must_be_filled",
                    "affected_fields": ["alcohol_consumption"]
                }
            ]
        },
        {
            "field_name": "mammogram_date",
            "data_type": "date",
            "description": "Date of last mammogram",
            "conditional_rules": [
                {
                    "rule_type": "required_if",
                    "condition_text": "gender is female",
                    "action": "must_be_filled",
                    "affected_fields": ["mammogram_date"]
                },
                {
                    "rule_type": "show_if",
                    "condition_text": "age >= 40",
                    "action": "must_be_filled",
                    "affected_fields": ["mammogram_date"]
                }
            ]
        },
        {
            "field_name": "experimental_drug_dose",
            "data_type": "float",
            "description": "Dose of experimental drug in mg",
            "conditional_rules": [
                {
                    "rule_type": "skip_if",
                    "condition_text": "treatment_arm is control",
                    "action": "must_be_blank",
                    "affected_fields": ["experimental_drug_dose"]
                }
            ]
        }
    ]

    print("STEP 1: Simulated LLM Output")
    print("-" * 80)
    print(json.dumps(fields, indent=2))
    print()

    # Extract rules using RuleExtractor
    extractor = RuleExtractor()
    rules = extractor.extract_rules_from_fields(fields, format_type="Custom")

    print("STEP 2: Extracted ConditionalRule Objects")
    print("-" * 80)
    print(f"Total rules extracted: {len(rules)}")
    print()

    for i, rule in enumerate(rules, 1):
        print(f"Rule {i}:")
        print(f"  Rule ID: {rule.rule_id}")
        print(f"  Type: {rule.rule_type}")
        print(f"  Affected Fields: {rule.affected_fields}")
        print(f"  Description: {rule.description}")
        print(f"  Python Condition: {rule.condition}")
        print(f"  Action: {rule.action}")
        print(f"  Severity: {rule.severity}")
        print(f"  Confidence: {rule.confidence}")
        print()

    print("STEP 3: Natural Language → Python Examples")
    print("-" * 80)

    examples = [
        ("gender is male", "pregnancy_status"),
        ("age >= 18", "alcohol_consumption"),
        ("gender is female", "mammogram_date"),
        ("age >= 40", "mammogram_date"),
        ("treatment_arm is control", "experimental_drug_dose"),
        ("pregnant is yes", "test_field"),
        ("diagnosis contains cancer", "test_field"),
    ]

    for condition_text, field_name in examples:
        python_expr = extractor._convert_natural_language_condition(
            condition_text,
            field_name
        )
        print(f"  '{condition_text}'")
        print(f"    → {python_expr}")
        print()

    print("=" * 80)
    print("Demo complete!")
    print()
    print("Key Features:")
    print("  ✓ LLM extracts conditional logic from ANY dictionary format")
    print("  ✓ Natural language converted to executable Python expressions")
    print("  ✓ Backward compatible with REDCap/FHIR parsers")
    print("  ✓ Defensive coding handles unparseable conditions")
    print("=" * 80)


def demo_comparison_patterns():
    """Demonstrate various comparison pattern conversions"""

    print()
    print("=" * 80)
    print("BONUS: Comparison Pattern Examples")
    print("=" * 80)
    print()

    extractor = RuleExtractor()

    patterns = [
        # Equality
        ("gender is male", "Gender equality check"),
        ("status = active", "Status equality check"),

        # Special values
        ("gender is female", "Female check"),
        ("pregnant is yes", "Yes/true check"),
        ("smoking is no", "No/false check"),

        # Numeric comparisons
        ("age >= 18", "Age greater than or equal"),
        ("age > 65", "Age greater than"),
        ("score < 50", "Score less than"),
        ("count <= 10", "Count less than or equal"),

        # String operations
        ("diagnosis contains cancer", "Contains check"),
        ("name contains smith", "Name contains check"),

        # Invalid (should return False)
        ("random unparseable text", "Invalid pattern"),
    ]

    for condition_text, description in patterns:
        python_expr = extractor._convert_natural_language_condition(condition_text)
        print(f"{description}:")
        print(f"  Input:  '{condition_text}'")
        print(f"  Output: {python_expr}")
        print()


if __name__ == "__main__":
    demo_llm_extracted_rules()
    demo_comparison_patterns()
