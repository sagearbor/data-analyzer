"""
Test script for data analysis endpoints

This script tests:
1. POST /api/v1/analyze - with sample CSV data
2. POST /api/v1/analyze/with-program - using a cached program

Prerequisites:
- API server must be running: uvicorn api_server:app --reload
- Set DATA_ANALYZER_API_KEY environment variable
- Have a sample CSV file and optionally a data dictionary

Usage:
    python test_analyze_endpoints.py
"""

import requests
import os
import sys
from pathlib import Path
import tempfile

# Configuration
API_BASE_URL = "http://localhost:8000"
API_KEY = os.getenv("DATA_ANALYZER_API_KEY", "test-key-12345")

# Create sample CSV data for testing
SAMPLE_CSV_DATA = """id,name,age,email
1,Alice,25,alice@example.com
2,Bob,30,bob@example.com
3,Charlie,35,charlie@example.com
4,David,150,david@example.com
5,Eve,25,invalid-email
"""

SAMPLE_DICTIONARY_CSV = """Variable / Field Name,Field Type,Field Label,Choices, Calculations, OR Slider Labels,Text Validation Type OR Show Slider Number,Text Validation Min,Text Validation Max
id,text,Patient ID,,integer,,
name,text,Full Name,,,,
age,text,Age in years,,integer,0,120
email,text,Email Address,,email,,
"""


def test_health_check():
    """Test that the API is running"""
    print("\n=== Testing Health Check ===")
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health")
        if response.status_code == 200:
            data = response.json()
            print(f"Status: {data['status']}")
            print(f"Services: {data['services']}")
            return True
        else:
            print(f"Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error connecting to API: {e}")
        print(f"Make sure the API server is running at {API_BASE_URL}")
        return False


def test_analyze_basic():
    """Test POST /api/v1/analyze with CSV data only"""
    print("\n=== Test 1: Analyze CSV without dictionary ===")

    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(SAMPLE_CSV_DATA)
        csv_path = f.name

    try:
        # Prepare request
        files = {
            'data_file': ('test_data.csv', open(csv_path, 'rb'), 'text/csv')
        }
        data = {
            'data_format': 'csv',
            'validate_logic': 'false',  # No dictionary, so no logic validation
            'return_format': 'json'
        }
        headers = {
            'X-API-Key': API_KEY
        }

        # Make request
        response = requests.post(
            f"{API_BASE_URL}/api/v1/analyze",
            files=files,
            data=data,
            headers=headers
        )

        # Check response
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Analysis ID: {result['analysis_id']}")
            print(f"Summary:")
            print(f"  - Total rows: {result['summary']['total_rows']}")
            print(f"  - Total columns: {result['summary']['total_columns']}")
            print(f"  - Issues found: {result['summary']['issues_found']}")
            print(f"  - Logic violations: {result['summary']['logic_violations']}")
            print(f"  - Execution time: {result['summary']['execution_time_seconds']}s")

            if result['field_violations']:
                print(f"\nField Violations:")
                for violation in result['field_violations'][:3]:  # Show first 3
                    print(f"  - {violation['field_name']}: {violation['violation_type']}")

            if result['recommendations']:
                print(f"\nRecommendations:")
                for rec in result['recommendations']:
                    print(f"  - {rec}")

            print("\n✓ Test passed")
            return True
        else:
            print(f"Error: {response.text}")
            print("\n✗ Test failed")
            return False

    finally:
        # Cleanup
        os.unlink(csv_path)


def test_analyze_with_dictionary():
    """Test POST /api/v1/analyze with CSV data and dictionary"""
    print("\n=== Test 2: Analyze CSV with dictionary ===")

    # Create temporary files
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(SAMPLE_CSV_DATA)
        csv_path = f.name

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(SAMPLE_DICTIONARY_CSV)
        dict_path = f.name

    try:
        # Prepare request
        files = {
            'data_file': ('test_data.csv', open(csv_path, 'rb'), 'text/csv'),
            'dictionary_file': ('test_dictionary.csv', open(dict_path, 'rb'), 'text/csv')
        }
        data = {
            'data_format': 'csv',
            'validate_logic': 'true',
            'return_format': 'json'
        }
        headers = {
            'X-API-Key': API_KEY
        }

        # Make request
        response = requests.post(
            f"{API_BASE_URL}/api/v1/analyze",
            files=files,
            data=data,
            headers=headers
        )

        # Check response
        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Analysis ID: {result['analysis_id']}")
            print(f"Program Used: {result.get('program_used', 'None')}")
            print(f"Summary:")
            print(f"  - Total rows: {result['summary']['total_rows']}")
            print(f"  - Total columns: {result['summary']['total_columns']}")
            print(f"  - Issues found: {result['summary']['issues_found']}")
            print(f"  - Logic violations: {result['summary']['logic_violations']}")

            if result['field_violations']:
                print(f"\nField Violations ({len(result['field_violations'])} total):")
                for violation in result['field_violations'][:5]:  # Show first 5
                    print(f"  - Row {violation['row_index']}, Field '{violation['field_name']}': "
                          f"{violation['violation_type']}")

            print("\n✓ Test passed")
            return True
        else:
            print(f"Error: {response.text}")
            print("\n✗ Test failed")
            return False

    finally:
        # Cleanup
        os.unlink(csv_path)
        os.unlink(dict_path)


def test_analyze_with_program():
    """Test POST /api/v1/analyze/with-program"""
    print("\n=== Test 3: Analyze CSV with cached program ===")
    print("Note: This test requires a program to exist first.")
    print("Run test_analyze_with_dictionary() first to create a program.")

    # First, we need to get a list of available programs
    headers = {'X-API-Key': API_KEY}

    try:
        # Try to list programs (this endpoint may not exist yet)
        response = requests.get(
            f"{API_BASE_URL}/api/v1/programs",
            headers=headers
        )

        if response.status_code == 200:
            programs = response.json()
            if programs.get('programs'):
                program_name = programs['programs'][0]['name']
                print(f"Using program: {program_name}")

                # Create temporary CSV file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    f.write(SAMPLE_CSV_DATA)
                    csv_path = f.name

                try:
                    # Prepare request
                    files = {
                        'data_file': ('test_data.csv', open(csv_path, 'rb'), 'text/csv')
                    }
                    data = {
                        'program': program_name,
                        'data_format': 'csv',
                        'return_format': 'json'
                    }

                    # Make request
                    response = requests.post(
                        f"{API_BASE_URL}/api/v1/analyze/with-program",
                        files=files,
                        data=data,
                        headers=headers
                    )

                    print(f"Status Code: {response.status_code}")

                    if response.status_code == 200:
                        result = response.json()
                        print(f"Analysis ID: {result['analysis_id']}")
                        print(f"Program Used: {result['program_used']}")
                        print(f"Summary:")
                        print(f"  - Total rows: {result['summary']['total_rows']}")
                        print(f"  - Issues found: {result['summary']['issues_found']}")
                        print(f"  - Logic violations: {result['summary']['logic_violations']}")
                        print("\n✓ Test passed")
                        return True
                    else:
                        print(f"Error: {response.text}")
                        print("\n✗ Test failed")
                        return False

                finally:
                    os.unlink(csv_path)
            else:
                print("No programs available. Run test 2 first to create a program.")
                return False
        else:
            print("Program listing endpoint not available yet. Skipping this test.")
            return False

    except Exception as e:
        print(f"Error: {e}")
        print("\n⚠ Test skipped (expected if program management endpoints not implemented yet)")
        return False


def test_error_handling():
    """Test error handling"""
    print("\n=== Test 4: Error Handling ===")

    # Test 1: Missing API key
    print("\n4a. Test missing API key")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(SAMPLE_CSV_DATA)
        csv_path = f.name

    try:
        files = {'data_file': ('test.csv', open(csv_path, 'rb'), 'text/csv')}
        data = {'data_format': 'csv'}
        response = requests.post(f"{API_BASE_URL}/api/v1/analyze", files=files, data=data)

        if response.status_code in [401, 403]:
            print(f"  ✓ Correctly rejected (status {response.status_code})")
        else:
            print(f"  ✗ Unexpected status: {response.status_code}")

    finally:
        os.unlink(csv_path)

    # Test 2: Invalid file format
    print("\n4b. Test invalid file format")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("not valid data")
        txt_path = f.name

    try:
        files = {'data_file': ('test.txt', open(txt_path, 'rb'), 'text/plain')}
        data = {'data_format': 'invalid_format'}
        headers = {'X-API-Key': API_KEY}
        response = requests.post(
            f"{API_BASE_URL}/api/v1/analyze",
            files=files,
            data=data,
            headers=headers
        )

        if response.status_code == 400:
            print(f"  ✓ Correctly rejected invalid format (status 400)")
        else:
            print(f"  Status: {response.status_code}")

    finally:
        os.unlink(txt_path)

    # Test 3: Program not found
    print("\n4c. Test program not found")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(SAMPLE_CSV_DATA)
        csv_path = f.name

    try:
        files = {'data_file': ('test.csv', open(csv_path, 'rb'), 'text/csv')}
        data = {
            'program': 'nonexistent-program-12345',
            'data_format': 'csv'
        }
        headers = {'X-API-Key': API_KEY}
        response = requests.post(
            f"{API_BASE_URL}/api/v1/analyze/with-program",
            files=files,
            data=data,
            headers=headers
        )

        if response.status_code == 404:
            print(f"  ✓ Correctly returned 404 for missing program")
        else:
            print(f"  Status: {response.status_code}")

    finally:
        os.unlink(csv_path)

    print("\n✓ Error handling tests complete")
    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("Data Analysis Endpoints Test Suite")
    print("=" * 60)

    # Check if API is running
    if not test_health_check():
        print("\n✗ API server is not running. Exiting.")
        sys.exit(1)

    # Run tests
    results = []
    results.append(("Basic analysis", test_analyze_basic()))
    results.append(("Analysis with dictionary", test_analyze_with_dictionary()))
    results.append(("Analysis with program", test_analyze_with_program()))
    results.append(("Error handling", test_error_handling()))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")

    total_passed = sum(1 for _, passed in results if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")

    if total_passed == len(results):
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n⚠ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
