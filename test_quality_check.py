#!/usr/bin/env python3
"""Test script to validate quality checking with data dictionary"""

import sys
from mcp_server import DataLoader, DataDictionaryParser, QualityChecker

def main():
    # Load the data file
    print("Loading data file...")
    data_df = DataLoader.load_csv("test_data_files/LTC-data-5rows01.csv")
    print(f"Loaded {len(data_df)} rows, {len(data_df.columns)} columns")

    # Load the data dictionary
    print("\nLoading data dictionary...")
    dict_df = DataLoader.load_csv("test_dictionaries/LTC-datadict-trunc01.csv")
    print(f"Loaded {len(dict_df)} dictionary entries")

    # Parse the dictionary
    print("\nParsing data dictionary...")
    schema, rules = DataDictionaryParser.parse_redcap_dictionary(dict_df)

    print(f"\nExtracted schema for {len(schema)} fields")
    print("Sample schema entries:")
    for field in ['consent_date', 'cand_dob', 'cand_sex', 'cand_eth', 'race_cand___1', 'race_cand___2']:
        if field in schema:
            print(f"  {field}: {schema[field]}")

    print(f"\nExtracted rules for {len(rules)} fields")
    print("Sample rules:")
    for field in ['cand_sex', 'cand_eth', 'race_cand___1', 'race_cand___2']:
        if field in rules:
            print(f"  {field}: {rules[field]}")

    # Run quality checks
    print("\n" + "="*60)
    print("RUNNING QUALITY CHECKS")
    print("="*60)

    checker = QualityChecker(data_df, schema=schema, rules=rules)

    # Check data types
    print("\n1. Data Type Validation:")
    type_result = checker.check_data_types()
    print(f"   Passed: {type_result['passed']}")
    print(f"   Total columns checked: {type_result['total_columns_checked']}")
    print(f"   Issues found: {type_result['issues_found']}")

    if type_result['issues']:
        print("\n   Type Issues:")
        for issue in type_result['issues']:
            print(f"\n   - Column: {issue['column']}")
            print(f"     Issue: {issue['issue']}")
            print(f"     Expected: {issue['expected_type']}")
            if 'invalid_values' in issue:
                print(f"     Invalid values: {issue['invalid_values']}")
            if 'description' in issue:
                print(f"     Description: {issue['description']}")

    # Check value ranges
    print("\n2. Value Range Validation:")
    range_result = checker.check_value_ranges()
    print(f"   Passed: {range_result['passed']}")
    print(f"   Issues found: {range_result['issues_found']}")

    if range_result['issues']:
        print("\n   Range Issues:")
        for issue in range_result['issues']:
            print(f"\n   - Column: {issue['column']}")
            print(f"     Rule: {issue['rule']}")
            if 'invalid_values' in issue:
                print(f"     Invalid values: {issue['invalid_values']}")
            print(f"     Violation count: {issue['violation_count']}")
            print(f"     Violating rows: {issue['violating_rows']}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    expected_issues = {
        'bad_dates': ['consent_date', 'cand_dob (row 3)', 'cand_dob (row 4)'],
        'bad_ranges': ['cand_sex=66', 'cand_eth=666', 'race_cand___1=666', 'race_cand___2=666']
    }

    print("\nExpected to find:")
    print("  - Bad date formats: consent_date=6/7oops/2024, cand_dob=6/8oops/2024, cand_dob=6/9oops/2024")
    print("  - Out of range values: cand_sex=66, cand_eth=666, race_cand___1=666, race_cand___2=666")

    total_issues = type_result['issues_found'] + range_result['issues_found']
    print(f"\nActual issues found: {total_issues}")

    if total_issues >= 5:  # At least the 3 date errors + some range errors
        print("\n✓ SUCCESS: Quality checker found the flawed data!")
        return 0
    else:
        print("\n✗ FAILURE: Quality checker did not find all expected issues")
        return 1

if __name__ == "__main__":
    sys.exit(main())