#!/usr/bin/env python
"""Test script to verify multi-format data loading functionality"""

import json
import pandas as pd
import io
import sys
import os

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server import DataLoader

def test_csv_format():
    """Test CSV format loading"""
    print("\n=== Testing CSV Format ===")
    csv_data = """id,name,age,salary,department
1,Alice,25,50000,HR
2,Bob,30,60000,IT
3,Charlie,35,70000,Finance"""

    try:
        df = DataLoader.load_csv(csv_data)
        print(f"✅ CSV loaded successfully: {df.shape} shape")
        print(df.head())
        return True
    except Exception as e:
        print(f"❌ CSV failed: {e}")
        return False

def test_json_format():
    """Test JSON format loading"""
    print("\n=== Testing JSON Format ===")

    # Test 1: Array of objects
    json_data1 = json.dumps([
        {"id": 1, "name": "Alice", "age": 25},
        {"id": 2, "name": "Bob", "age": 30},
        {"id": 3, "name": "Charlie", "age": 35}
    ])

    try:
        df1 = DataLoader.load_json(json_data1)
        print(f"✅ JSON array loaded: {df1.shape} shape")
        print(df1.head())
    except Exception as e:
        print(f"❌ JSON array failed: {e}")
        return False

    # Test 2: Nested object with array
    json_data2 = json.dumps({
        "employees": [
            {"id": 1, "name": "Alice", "department": {"name": "HR", "floor": 3}},
            {"id": 2, "name": "Bob", "department": {"name": "IT", "floor": 5}}
        ],
        "company": "TechCorp"
    })

    try:
        df2 = DataLoader.load_json(json_data2)
        print(f"✅ Nested JSON loaded: {df2.shape} shape")
        print(df2.head())
        print(f"Columns: {list(df2.columns)}")
    except Exception as e:
        print(f"❌ Nested JSON failed: {e}")
        return False

    return True

def test_excel_format():
    """Test Excel format loading"""
    print("\n=== Testing Excel Format ===")

    # Create sample Excel file in memory
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'salary': [50000, 60000, 70000]
    })

    # Write to bytes buffer
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    excel_bytes = output.getvalue()

    try:
        df_loaded = DataLoader.load_excel(excel_bytes)
        print(f"✅ Excel loaded successfully: {df_loaded.shape} shape")
        print(df_loaded.head())
        return True
    except Exception as e:
        print(f"❌ Excel failed: {e}")
        return False

def test_parquet_format():
    """Test Parquet format loading"""
    print("\n=== Testing Parquet Format ===")

    # Create sample Parquet file in memory
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'salary': [50000.5, 60000.75, 70000.25]
    })

    # Write to bytes buffer
    output = io.BytesIO()
    df.to_parquet(output, index=False)
    parquet_bytes = output.getvalue()

    try:
        df_loaded = DataLoader.load_parquet(parquet_bytes)
        print(f"✅ Parquet loaded successfully: {df_loaded.shape} shape")
        print(df_loaded.head())
        return True
    except Exception as e:
        print(f"❌ Parquet failed: {e}")
        return False

def test_load_data_wrapper():
    """Test the generic load_data method"""
    print("\n=== Testing load_data Wrapper ===")

    formats_tested = []

    # Test CSV through wrapper
    csv_data = "id,name\n1,Alice\n2,Bob"
    try:
        df = DataLoader.load_data(csv_data, file_format='csv')
        print(f"✅ load_data(CSV): {df.shape}")
        formats_tested.append('CSV')
    except Exception as e:
        print(f"❌ load_data(CSV) failed: {e}")

    # Test JSON through wrapper
    json_data = json.dumps([{"id": 1, "name": "Alice"}])
    try:
        df = DataLoader.load_data(json_data, file_format='json')
        print(f"✅ load_data(JSON): {df.shape}")
        formats_tested.append('JSON')
    except Exception as e:
        print(f"❌ load_data(JSON) failed: {e}")

    print(f"\nFormats successfully tested through load_data: {formats_tested}")
    return len(formats_tested) >= 2

def main():
    """Run all format tests"""
    print("=" * 60)
    print("Multi-Format Data Loader Test Suite")
    print("=" * 60)

    results = {
        'CSV': test_csv_format(),
        'JSON': test_json_format(),
        'Excel': test_excel_format(),
        'Parquet': test_parquet_format(),
        'Wrapper': test_load_data_wrapper()
    }

    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print("=" * 60)

    for format_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{format_name:10} : {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All format tests passed successfully!")
        return 0
    else:
        print(f"\n⚠️ {total_tests - total_passed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit(main())