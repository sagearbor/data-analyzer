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
    csv_data = """id,name,age,salary,department
1,Alice,25,50000,HR
2,Bob,30,60000,IT
3,Charlie,35,70000,Finance"""

    df = DataLoader.load_csv(csv_data)
    assert df is not None, "CSV should load successfully"
    assert df.shape == (3, 5), f"Expected (3, 5) shape, got {df.shape}"

def test_json_format():
    """Test JSON format loading"""
    # Test 1: Array of objects
    json_data1 = json.dumps([
        {"id": 1, "name": "Alice", "age": 25},
        {"id": 2, "name": "Bob", "age": 30},
        {"id": 3, "name": "Charlie", "age": 35}
    ])

    df1 = DataLoader.load_json(json_data1)
    assert df1 is not None, "JSON array should load successfully"
    assert df1.shape == (3, 3), f"Expected (3, 3) shape, got {df1.shape}"

    # Test 2: Nested object with array
    json_data2 = json.dumps({
        "employees": [
            {"id": 1, "name": "Alice", "department": {"name": "HR", "floor": 3}},
            {"id": 2, "name": "Bob", "department": {"name": "IT", "floor": 5}}
        ],
        "company": "TechCorp"
    })

    df2 = DataLoader.load_json(json_data2)
    assert df2 is not None, "Nested JSON should load successfully"
    assert len(df2) == 2, f"Expected 2 rows, got {len(df2)}"

def test_excel_format():
    """Test Excel format loading"""
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

    df_loaded = DataLoader.load_excel(excel_bytes)
    assert df_loaded is not None, "Excel should load successfully"
    assert df_loaded.shape == (3, 4), f"Expected (3, 4) shape, got {df_loaded.shape}"

def test_parquet_format():
    """Test Parquet format loading"""
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

    df_loaded = DataLoader.load_parquet(parquet_bytes)
    assert df_loaded is not None, "Parquet should load successfully"
    assert df_loaded.shape == (3, 4), f"Expected (3, 4) shape, got {df_loaded.shape}"

def test_load_data_wrapper():
    """Test the generic load_data method"""
    # Test CSV through wrapper
    csv_data = "id,name\n1,Alice\n2,Bob"
    df_csv = DataLoader.load_data(csv_data, file_format='csv')
    assert df_csv is not None, "load_data should handle CSV"
    assert df_csv.shape == (2, 2), f"Expected (2, 2) shape for CSV, got {df_csv.shape}"

    # Test JSON through wrapper
    json_data = json.dumps([{"id": 1, "name": "Alice"}])
    df_json = DataLoader.load_data(json_data, file_format='json')
    assert df_json is not None, "load_data should handle JSON"
    assert df_json.shape == (1, 2), f"Expected (1, 2) shape for JSON, got {df_json.shape}"

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