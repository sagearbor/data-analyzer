"""
Test script for dictionary management endpoints

This script tests:
1. POST /api/v1/dictionary/parse - Parse dictionary and save program
2. GET /api/v1/dictionary/{dict_id} - Retrieve saved program

Usage:
    python test_dictionary_endpoints.py
"""

import requests
import os
from pathlib import Path

# API configuration
BASE_URL = "http://localhost:8002"
API_KEY = os.getenv("DATA_ANALYZER_API_KEY", "test-key-12345")

# Test dictionary file
TEST_DICT_PATH = Path("tests/test_data/dictionaries/simple_csv_dict.csv")

def test_parse_dictionary():
    """Test POST /api/v1/dictionary/parse endpoint"""
    print("=" * 60)
    print("Testing POST /api/v1/dictionary/parse")
    print("=" * 60)

    if not TEST_DICT_PATH.exists():
        print(f"Error: Test dictionary not found at {TEST_DICT_PATH}")
        return None

    # Prepare multipart/form-data request
    with open(TEST_DICT_PATH, 'rb') as f:
        files = {
            'dictionary_file': (TEST_DICT_PATH.name, f, 'text/csv')
        }
        data = {
            'save_program': 'true',
            'program_name': 'TestProgram_API_Dict_Endpoints'
        }
        headers = {
            'X-API-Key': API_KEY
        }

        try:
            response = requests.post(
                f"{BASE_URL}/api/v1/dictionary/parse",
                files=files,
                data=data,
                headers=headers,
                timeout=120
            )

            print(f"Status Code: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                print(f"Success! Program created:")
                print(f"  Program ID: {result['program_id']}")
                print(f"  Program Name: {result['program_name']}")
                print(f"  Fields Extracted: {result['fields_extracted']}")
                print(f"  Rules Extracted: {result['rules_extracted']}")
                print(f"  Logic Rules: {result['logic_rules_extracted']}")
                print(f"  Dictionary Format: {result['dictionary_format']}")
                print(f"  Model Used: {result['model_used']}")
                print(f"  Generation Time: {result['generation_time_seconds']:.2f}s")
                return result['program_id']
            else:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None

        except Exception as e:
            print(f"Request failed: {e}")
            return None


def test_get_dictionary(program_id):
    """Test GET /api/v1/dictionary/{dict_id} endpoint"""
    print("\n" + "=" * 60)
    print(f"Testing GET /api/v1/dictionary/{program_id}")
    print("=" * 60)

    if not program_id:
        print("Skipping test - no program ID provided")
        return

    headers = {
        'X-API-Key': API_KEY
    }

    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/dictionary/{program_id}",
            headers=headers,
            timeout=30
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            print(f"Success! Retrieved program:")
            print(f"  Program ID: {result['program_id']}")
            print(f"  Program Name: {result['name']}")
            print(f"  Aliases: {result['aliases']}")
            print(f"  Created At: {result['created_at']}")
            print(f"  Created By: {result['created_by']}")
            print(f"  Dictionary Source: {result['dictionary_source']}")
            print(f"  Dictionary Format: {result['dictionary_format']}")
            print(f"  Fields: {result['num_fields']}")
            print(f"  Basic Rules: {result['num_basic_rules']}")
            print(f"  Logic Rules: {result['num_logic_rules']}")
            print(f"  Use Count: {result['use_count']}")
            print(f"  Status: {result['status']}")
            print(f"  Version: {result['version']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

    except Exception as e:
        print(f"Request failed: {e}")


def main():
    """Run all tests"""
    print("Starting Dictionary Management Endpoints Test")
    print(f"Base URL: {BASE_URL}")
    print(f"Test Dictionary: {TEST_DICT_PATH}")
    print()

    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health", timeout=5)
        if response.status_code != 200:
            print(f"Error: Server health check failed with status {response.status_code}")
            return
    except Exception as e:
        print(f"Error: Cannot connect to server at {BASE_URL}")
        print(f"Make sure the API server is running: uvicorn api_server:app --reload")
        print(f"Error details: {e}")
        return

    # Test 1: Parse dictionary
    program_id = test_parse_dictionary()

    # Test 2: Get dictionary (retrieve by ID)
    if program_id:
        test_get_dictionary(program_id)

        # Test 3: Get dictionary (retrieve by name)
        print("\n" + "=" * 60)
        print("Testing GET with program name")
        print("=" * 60)
        test_get_dictionary("TestProgram_API_Dict_Endpoints")

    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
