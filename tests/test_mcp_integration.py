#!/usr/bin/env python
"""MCP-path integration tests: DataLoader -> QualityPipeline / QualityChecker.

ADP-89: these tests used to wrap everything in `try/except Exception` and
`return True/False`, which pytest ignores, so they could never fail. They now
assert real results.
"""

import json
import os
import sys

import pandas as pd
import pytest

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server import DataLoader, QualityChecker, QualityPipeline


@pytest.mark.asyncio
async def test_mcp_analyze_data():
    """Load a CSV and run all checks: the planted age of 150 must be the one issue."""
    csv_content = """id,name,age,salary,department
1,Alice,25,50000,HR
2,Bob,30,60000,IT
3,Charlie,35,70000,Finance
4,David,28,55000,HR
5,Eve,150,80000,IT"""  # Age 150 is intentionally invalid

    schema = {'id': 'int', 'name': 'str', 'age': 'int', 'salary': 'int', 'department': 'str'}
    rules = {
        'age': {'min': 18, 'max': 65},  # Age 150 violates this
        'salary': {'min': 40000, 'max': 100000},
        'department': {'allowed': ['HR', 'IT', 'Finance', 'Sales']},
    }

    df = DataLoader.load_csv(csv_content)
    results = QualityPipeline(df, schema=schema, rules=rules).run_all_checks(min_rows=3)

    assert results['overall_passed'] is False
    assert results['total_issues'] == 1
    assert results['checks']['row_count']['passed'] is True
    assert results['checks']['data_types']['passed'] is True
    assert results['checks']['value_ranges']['passed'] is False

    assert len(results['issues']) == 1
    issue = results['issues'][0]
    assert issue['column'] == 'age'
    assert issue['rule'] == 'max <= 65'
    assert issue['violating_rows'] == [4]  # Eve, the fifth row


@pytest.mark.asyncio
async def test_multi_format_mcp():
    """A JSON-loaded table and an Excel-style DataFrame both analyse cleanly."""
    json_data = json.dumps([
        {"id": 1, "name": "Alice", "score": 95},
        {"id": 2, "name": "Bob", "score": 87},
        {"id": 3, "name": "Charlie", "score": 92},
    ])
    json_df = DataLoader.load_json(json_data)
    assert list(json_df.columns) == ['id', 'name', 'score']
    assert len(json_df) == 3
    json_results = QualityPipeline(json_df).run_all_checks()
    assert json_results['overall_passed'] is True
    assert json_results['total_issues'] == 0
    assert json_results['summary_stats']['shape'] == {'rows': 3, 'columns': 3}

    excel_df = pd.DataFrame({
        'id': [1, 2, 3],
        'product': ['Widget', 'Gadget', 'Tool'],
        'price': [19.99, 29.99, 39.99],
    })
    excel_results = QualityPipeline(excel_df).run_all_checks()
    assert excel_results['overall_passed'] is True
    assert excel_results['total_issues'] == 0
    assert excel_results['summary_stats']['auto_detected_types']['price'] == 'float'


@pytest.mark.asyncio
async def test_data_quality_checks():
    """Each individual check reports the planted problems on a flawed table."""
    df = pd.DataFrame({
        'id': [1, 2, 2, 4, None],  # Duplicate and missing
        'email': ['alice@test.com', 'invalid-email', None, 'bob@test.com', 'eve@test.com'],
        'age': [25, 200, 35, -5, 40],  # Invalid ages
        'status': ['active', 'inactive', 'ACTIVE', 'pending', 'active'],  # Mixed case
    })
    schema = {'id': 'int', 'email': 'str', 'age': 'int', 'status': 'str'}
    rules = {
        'age': {'min': 0, 'max': 120},
        'status': {'allowed': ['active', 'inactive', 'pending']},
    }
    checker = QualityChecker(df, schema=schema, rules=rules)

    row_result = checker.check_row_count(min_rows=3)
    assert row_result['passed'] is True
    assert row_result['row_count'] == 5

    # Row count must also be able to fail
    assert checker.check_row_count(min_rows=10)['passed'] is False

    range_result = checker.check_value_ranges()
    assert range_result['passed'] is False
    assert range_result['issues_found'] == 3
    by_rule = {(i['column'], i['rule']): i for i in range_result['issues']}
    assert by_rule[('age', 'min >= 0')]['violating_rows'] == [3]       # age -5
    assert by_rule[('age', 'max <= 120')]['violating_rows'] == [1]     # age 200
    status_issue = by_rule[('status', "allowed_values: ['active', 'inactive', 'pending']")]
    assert status_issue['invalid_values'] == ['ACTIVE']
    assert status_issue['violating_rows'] == [2]

    stats = checker.get_summary_stats()
    assert stats['shape'] == {'rows': 5, 'columns': 4}
    assert stats['missing_values'] == {'id': 1, 'email': 1, 'age': 0, 'status': 0}


@pytest.mark.asyncio
async def test_auto_type_detection():
    """Integer, decimal, text, ISO date and true/false columns are recognised."""
    df = pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'str_col': ['a', 'b', 'c', 'd', 'e'],
        'date_col': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
        'bool_col': ['true', 'false', 'true', 'false', 'true'],
        'mixed_col': [1, '2', 3.0, '4', 5],
    })
    detected_types = QualityChecker(df)._auto_detect_types()

    assert detected_types['int_col'] == 'int'
    assert detected_types['float_col'] == 'float'
    assert detected_types['str_col'] == 'str'
    assert detected_types['date_col'] == 'datetime'
    assert detected_types['bool_col'] == 'bool'
