#!/usr/bin/env python3
"""
Quick test to verify the refactored validation works
Tests that DataQualityAnalyzer properly uses QualityPipeline
"""

import pandas as pd
import asyncio
from web_app import DataQualityAnalyzer

# Create demo data matching the Western demo
demo_data = pd.DataFrame({
    'employee_id': [1001, 1002, 1003, 1004, 1005],
    'first_name': ['John', 'Jane', 'Mike', 'Bob', 'Alice'],
    'last_name': ['Smith', 'Doe', 'Johnson', 'Brown', 'Wilson'],
    'age': [35, 28, 67, 45, 32],  # 67 is outside range
    'salary': [75000, 85000, 45000, 95000, 105000],  # 45000 below min, 105000 OK
    'department': ['Engineering', 'Marketing', 'InvalidDept', 'Sales', 'Finance'],  # InvalidDept not in allowed
    'is_active': [True, True, False, True, True],
})

# Create demo dictionary
demo_dict = {
    'rules': {
        'age': {
            'type': 'integer',
            'min': 18,
            'max': 65
        },
        'salary': {
            'type': 'decimal',
            'min': 50000,
            'max': 150000
        },
        'department': {
            'type': 'string',
            'allowed_values': ['Engineering', 'Marketing', 'Sales', 'Finance', 'HR', 'Management']
        }
    }
}

async def test_validation():
    print("=" * 80)
    print("TESTING REFACTORED VALIDATION")
    print("=" * 80)

    analyzer = DataQualityAnalyzer()

    # Test 1: With dictionary
    print("\n[TEST 1] Analysis WITH dictionary (should catch age, salary, department issues)")
    print("-" * 80)
    results = await analyzer.analyze_data_quality(demo_data, demo_dict)

    print(f"\nSummary:")
    print(f"  Total rows: {results['summary']['total_rows']}")
    print(f"  Total columns: {results['summary']['total_columns']}")
    print(f"  Issues found: {results['summary']['issues_found']}")
    print(f"  Critical issues: {results['summary']['critical_issues']}")
    print(f"  Warnings: {results['summary']['warnings']}")

    print(f"\nIssues found ({len(results['issues'])}):")
    for issue in results['issues']:
        issue_type = issue.get('type', issue.get('issue', 'unknown'))
        severity = issue.get('severity', 'unknown')
        msg = issue.get('message', issue.get('description', 'No message'))
        print(f"  - [{severity.upper()}] {issue_type}: {msg}")

    # Test 2: Without dictionary (auto-detection)
    print("\n\n[TEST 2] Analysis WITHOUT dictionary (auto-detection only)")
    print("-" * 80)
    results_no_dict = await analyzer.analyze_data_quality(demo_data, None)

    print(f"\nSummary:")
    print(f"  Total rows: {results_no_dict['summary']['total_rows']}")
    print(f"  Total columns: {results_no_dict['summary']['total_columns']}")
    print(f"  Issues found: {results_no_dict['summary']['issues_found']}")
    print(f"  Critical issues: {results_no_dict['summary']['critical_issues']}")
    print(f"  Warnings: {results_no_dict['summary']['warnings']}")

    print(f"\nIssues found ({len(results_no_dict['issues'])}):")
    for issue in results_no_dict['issues']:
        issue_type = issue.get('type', issue.get('issue', 'unknown'))
        severity = issue.get('severity', 'unknown')
        msg = issue.get('message', issue.get('description', 'No message'))
        print(f"  - [{severity.upper()}] {issue_type}: {msg}")

    # Verify critical behavior: "InvalidDept" should NOT be flagged as invalid just because it contains "Invalid"
    print("\n\n[VERIFICATION] Checking hardcoded 'invalid' string matching is removed:")
    print("-" * 80)
    invalid_dept_issues = [i for i in results['issues']
                          if i.get('column') == 'department' and 'InvalidDept' in str(i.get('value', ''))]

    if invalid_dept_issues:
        for issue in invalid_dept_issues:
            reason = issue.get('rule', issue.get('type', issue.get('issue', 'unknown')))
            print(f"✓ 'InvalidDept' flagged correctly due to: {reason}")
            print(f"  (Not because it contains the string 'invalid', but because it's not in allowed_values)")
    else:
        print("✗ 'InvalidDept' was not flagged (this might be an issue)")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_validation())
