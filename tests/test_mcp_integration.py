#!/usr/bin/env python
"""Test MCP server integration and functionality"""

import asyncio
import json
import base64
import pandas as pd
import pytest
import sys
import os
from datetime import datetime

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server import DataLoader, QualityChecker, QualityPipeline

@pytest.mark.asyncio
async def test_mcp_analyze_data():
    """Test the analyze_data MCP tool function"""
    print("\n=== Testing MCP analyze_data Tool ===")

    # Create test CSV data
    csv_content = """id,name,age,salary,department
1,Alice,25,50000,HR
2,Bob,30,60000,IT
3,Charlie,35,70000,Finance
4,David,28,55000,HR
5,Eve,150,80000,IT"""  # Age 150 is intentionally invalid

    # Define schema with expected types
    schema = {
        'id': 'int',
        'name': 'str',
        'age': 'int',
        'salary': 'int',
        'department': 'str'
    }

    # Define validation rules
    rules = {
        'age': {'min': 18, 'max': 65},  # Age 150 will violate this
        'salary': {'min': 40000, 'max': 100000},
        'department': {'allowed': ['HR', 'IT', 'Finance', 'Sales']}
    }

    try:
        # Load and analyze data (simulating MCP tool call)
        df = DataLoader.load_csv(csv_content)
        pipeline = QualityPipeline(df, schema=schema, rules=rules)
        results = pipeline.run_all_checks(min_rows=3)

        print(f"✅ Analysis completed at: {results['timestamp']}")
        print(f"   Overall passed: {results['overall_passed']}")
        print(f"   Total issues: {results['total_issues']}")

        # Check specific results
        checks = results['checks']
        print("\n   Check Results:")
        print(f"   - Row count: {checks['row_count']['passed']}")
        print(f"   - Data types: {checks['data_types']['passed']}")
        print(f"   - Value ranges: {checks['value_ranges']['passed']}")

        # Verify that age violation was detected
        if not checks['value_ranges']['passed']:
            violations = checks['value_ranges'].get('range_violations', [])
            age_violations = [v for v in violations if v['column'] == 'age']
            if age_violations:
                print(f"\n   ✅ Correctly detected age violation: {age_violations[0]['issue']}")

        return results['overall_passed'] == False  # Should fail due to age violation

    except Exception as e:
        print(f"❌ MCP analyze_data failed: {e}")
        return False

@pytest.mark.asyncio
async def test_multi_format_mcp():
    """Test MCP with different data formats"""
    print("\n=== Testing MCP with Multiple Formats ===")

    formats_tested = []

    # Test JSON format through MCP
    json_data = json.dumps([
        {"id": 1, "name": "Alice", "score": 95},
        {"id": 2, "name": "Bob", "score": 87},
        {"id": 3, "name": "Charlie", "score": 92}
    ])

    try:
        df = DataLoader.load_json(json_data)
        pipeline = QualityPipeline(df)
        results = pipeline.run_all_checks()
        print(f"✅ JSON format: {results['overall_passed']}")
        formats_tested.append('JSON')
    except Exception as e:
        print(f"❌ JSON format failed: {e}")

    # Test Excel format (simulated)
    excel_df = pd.DataFrame({
        'id': [1, 2, 3],
        'product': ['Widget', 'Gadget', 'Tool'],
        'price': [19.99, 29.99, 39.99]
    })

    try:
        pipeline = QualityPipeline(excel_df)
        results = pipeline.run_all_checks()
        print(f"✅ Excel format (simulated): {results['overall_passed']}")
        formats_tested.append('Excel')
    except Exception as e:
        print(f"❌ Excel format failed: {e}")

    print(f"\nFormats tested through MCP: {formats_tested}")
    return len(formats_tested) == 2

@pytest.mark.asyncio
async def test_data_quality_checks():
    """Test comprehensive data quality check functionality"""
    print("\n=== Testing Data Quality Checks ===")

    # Create DataFrame with various data quality issues
    df = pd.DataFrame({
        'id': [1, 2, 2, 4, None],  # Duplicate and missing
        'email': ['alice@test.com', 'invalid-email', None, 'bob@test.com', 'eve@test.com'],
        'age': [25, 200, 35, -5, 40],  # Invalid ages
        'status': ['active', 'inactive', 'ACTIVE', 'pending', 'active']  # Mixed case
    })

    schema = {
        'id': 'int',
        'email': 'str',
        'age': 'int',
        'status': 'str'
    }

    rules = {
        'age': {'min': 0, 'max': 120},
        'status': {'allowed': ['active', 'inactive', 'pending']}
    }

    try:
        checker = QualityChecker(df, schema=schema, rules=rules)

        # Test row count
        row_result = checker.check_row_count(min_rows=3)
        print(f"✅ Row count check: {row_result['passed']} - {row_result['message']}")

        # Test data types
        type_result = checker.check_data_types()
        print(f"✅ Type check: {type_result['passed']} - Found {type_result['issues_found']} issues")

        # Test value ranges
        range_result = checker.check_value_ranges()
        print(f"✅ Range check: {range_result['passed']} - Found {range_result['issues_found']} issues")

        # Test summary stats
        stats = checker.get_summary_stats()
        print(f"✅ Summary stats generated:")
        print(f"   - Missing values: {stats['missing_values']}")
        print(f"   - Duplicate rows: {stats['duplicate_rows']}")

        return True

    except Exception as e:
        print(f"❌ Data quality checks failed: {e}")
        return False

@pytest.mark.asyncio
async def test_auto_type_detection():
    """Test automatic data type detection"""
    print("\n=== Testing Auto Type Detection ===")

    df = pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'str_col': ['a', 'b', 'c', 'd', 'e'],
        'date_col': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01'],
        'bool_col': ['true', 'false', 'true', 'false', 'true'],
        'mixed_col': [1, '2', 3.0, '4', 5]
    })

    try:
        checker = QualityChecker(df)
        detected_types = checker._auto_detect_types()

        print("✅ Auto-detected types:")
        for col, dtype in detected_types.items():
            print(f"   - {col}: {dtype}")

        # Verify correct detection
        assert detected_types['int_col'] == 'int'
        assert detected_types['float_col'] == 'float'
        assert detected_types['str_col'] == 'str'
        assert detected_types['date_col'] == 'datetime'
        assert detected_types['bool_col'] == 'bool'

        return True

    except Exception as e:
        print(f"❌ Auto type detection failed: {e}")
        return False

async def main():
    """Run all MCP integration tests"""
    print("=" * 60)
    print("MCP Server Integration Test Suite")
    print("=" * 60)

    results = {
        'MCP analyze_data': await test_mcp_analyze_data(),
        'Multi-format MCP': await test_multi_format_mcp(),
        'Data quality checks': await test_data_quality_checks(),
        'Auto type detection': await test_auto_type_detection()
    }

    print("\n" + "=" * 60)
    print("MCP Integration Test Results:")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20} : {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All MCP integration tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total_tests - total_passed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit(asyncio.run(main()))