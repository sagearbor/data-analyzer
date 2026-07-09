"""
Test script for api_5 Program Management Endpoints

Tests all 5 endpoints:
1. GET /api/v1/programs - List programs
2. GET /api/v1/programs/{id_or_alias} - Get program details
3. POST /api/v1/programs/{id}/alias - Create alias
4. DELETE /api/v1/programs/{id} - Delete program (admin)
5. POST /api/v1/programs/{id}/restore - Restore program (admin)
"""

import requests
import json
import sys
import os
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8001"
API_KEY = os.getenv("DATA_ANALYZER_API_KEY", "test-api-key-12345")
ADMIN_PASSWORD = os.getenv("DATA_ANALYZER_ADMIN_PASSWORD", "admin123")

# Headers
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

ADMIN_HEADERS = {
    "X-API-Key": API_KEY,
    "X-Admin-Password": ADMIN_PASSWORD,
    "Content-Type": "application/json"
}


def print_section(title):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_test(test_name, passed, details=""):
    """Print test result"""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}: {test_name}")
    if details:
        print(f"  Details: {details}")


def test_health_check():
    """Test health endpoint first"""
    print_section("0. Health Check")

    try:
        response = requests.get(f"{BASE_URL}/api/v1/health")

        if response.status_code == 200:
            data = response.json()
            print_test("Health check", True, f"Status: {data.get('status')}")
            print(f"  Services: {json.dumps(data.get('services'), indent=2)}")
            return True
        else:
            print_test("Health check", False, f"Status code: {response.status_code}")
            return False
    except Exception as e:
        print_test("Health check", False, str(e))
        return False


def test_list_programs():
    """Test GET /api/v1/programs"""
    print_section("1. List Programs (GET /api/v1/programs)")

    tests_passed = 0
    tests_total = 5

    # Test 1: Basic list (no filters)
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/programs",
            headers=HEADERS
        )

        if response.status_code == 200:
            data = response.json()
            print_test("List all programs", True,
                      f"Total: {data['total']}, Returned: {len(data['programs'])}")
            tests_passed += 1
        else:
            print_test("List all programs", False, f"Status: {response.status_code}")
    except Exception as e:
        print_test("List all programs", False, str(e))

    # Test 2: With pagination
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/programs?limit=5&offset=0",
            headers=HEADERS
        )

        if response.status_code == 200:
            data = response.json()
            print_test("List with pagination", True,
                      f"Limit: 5, Offset: 0, Returned: {len(data['programs'])}")
            tests_passed += 1
        else:
            print_test("List with pagination", False, f"Status: {response.status_code}")
    except Exception as e:
        print_test("List with pagination", False, str(e))

    # Test 3: With search
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/programs?search=test",
            headers=HEADERS
        )

        if response.status_code == 200:
            data = response.json()
            print_test("List with search", True, f"Found: {data['total']} programs")
            tests_passed += 1
        else:
            print_test("List with search", False, f"Status: {response.status_code}")
    except Exception as e:
        print_test("List with search", False, str(e))

    # Test 4: Filter by status
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/programs?status=active",
            headers=HEADERS
        )

        if response.status_code == 200:
            data = response.json()
            print_test("Filter by status", True, f"Active programs: {data['total']}")
            tests_passed += 1
        else:
            print_test("Filter by status", False, f"Status: {response.status_code}")
    except Exception as e:
        print_test("Filter by status", False, str(e))

    # Test 5: Without API key (should fail)
    try:
        response = requests.get(f"{BASE_URL}/api/v1/programs")

        if response.status_code == 401:
            print_test("Authentication required", True, "Correctly rejected unauthenticated request")
            tests_passed += 1
        else:
            print_test("Authentication required", False,
                      f"Expected 401, got {response.status_code}")
    except Exception as e:
        print_test("Authentication required", False, str(e))

    print(f"\nList Programs Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_get_program(program_id=None):
    """Test GET /api/v1/programs/{id_or_alias}"""
    print_section("2. Get Program Details (GET /api/v1/programs/{id})")

    tests_passed = 0
    tests_total = 3

    # First, get a program ID from the list
    if not program_id:
        try:
            response = requests.get(
                f"{BASE_URL}/api/v1/programs?limit=1",
                headers=HEADERS
            )
            if response.status_code == 200:
                data = response.json()
                if data['programs']:
                    program_id = data['programs'][0]['program_id']
                    print(f"Using program ID: {program_id}")
                else:
                    print("No programs available for testing")
                    return False
            else:
                print(f"Failed to get program list: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error getting program list: {e}")
            return False

    # Test 1: Get by ID
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/programs/{program_id}",
            headers=HEADERS
        )

        if response.status_code == 200:
            data = response.json()
            print_test("Get program by ID", True,
                      f"Name: {data['name']}, Fields: {data['num_fields']}")
            tests_passed += 1
        else:
            print_test("Get program by ID", False, f"Status: {response.status_code}")
    except Exception as e:
        print_test("Get program by ID", False, str(e))

    # Test 2: Get non-existent program
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/programs/nonexistent-id-12345",
            headers=HEADERS
        )

        if response.status_code == 404:
            print_test("Get non-existent program", True, "Correctly returned 404")
            tests_passed += 1
        else:
            print_test("Get non-existent program", False,
                      f"Expected 404, got {response.status_code}")
    except Exception as e:
        print_test("Get non-existent program", False, str(e))

    # Test 3: Without API key (should fail)
    try:
        response = requests.get(f"{BASE_URL}/api/v1/programs/{program_id}")

        if response.status_code == 401:
            print_test("Authentication required", True, "Correctly rejected unauthenticated request")
            tests_passed += 1
        else:
            print_test("Authentication required", False,
                      f"Expected 401, got {response.status_code}")
    except Exception as e:
        print_test("Authentication required", False, str(e))

    print(f"\nGet Program Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_create_alias(program_id=None):
    """Test POST /api/v1/programs/{id}/alias"""
    print_section("3. Create Alias (POST /api/v1/programs/{id}/alias)")

    tests_passed = 0
    tests_total = 4

    # Get a program ID if not provided
    if not program_id:
        try:
            response = requests.get(
                f"{BASE_URL}/api/v1/programs?limit=1",
                headers=HEADERS
            )
            if response.status_code == 200:
                data = response.json()
                if data['programs']:
                    program_id = data['programs'][0]['program_id']
                    print(f"Using program ID: {program_id}")
                else:
                    print("No programs available for testing")
                    return False
        except Exception as e:
            print(f"Error getting program list: {e}")
            return False

    # Generate unique alias
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_alias = f"test_alias_{timestamp}"

    # Test 1: Create valid alias
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/programs/{program_id}/alias",
            headers=HEADERS,
            json={"alias": test_alias}
        )

        if response.status_code == 200:
            data = response.json()
            print_test("Create alias", True, f"Alias: {data['alias']}")
            tests_passed += 1
        else:
            print_test("Create alias", False,
                      f"Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print_test("Create alias", False, str(e))

    # Test 2: Create duplicate alias (should fail)
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/programs/{program_id}/alias",
            headers=HEADERS,
            json={"alias": test_alias}
        )

        if response.status_code == 409:
            print_test("Duplicate alias rejected", True, "Correctly returned 409 Conflict")
            tests_passed += 1
        else:
            print_test("Duplicate alias rejected", False,
                      f"Expected 409, got {response.status_code}")
    except Exception as e:
        print_test("Duplicate alias rejected", False, str(e))

    # Test 3: Invalid alias format
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/programs/{program_id}/alias",
            headers=HEADERS,
            json={"alias": "invalid alias with spaces!"}
        )

        if response.status_code == 422:  # Pydantic validation error
            print_test("Invalid alias format", True, "Correctly rejected invalid format")
            tests_passed += 1
        else:
            print_test("Invalid alias format", False,
                      f"Expected 422, got {response.status_code}")
    except Exception as e:
        print_test("Invalid alias format", False, str(e))

    # Test 4: Without API key
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/programs/{program_id}/alias",
            json={"alias": "another_alias"}
        )

        if response.status_code == 401:
            print_test("Authentication required", True, "Correctly rejected unauthenticated request")
            tests_passed += 1
        else:
            print_test("Authentication required", False,
                      f"Expected 401, got {response.status_code}")
    except Exception as e:
        print_test("Authentication required", False, str(e))

    print(f"\nCreate Alias Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def test_delete_program(program_id=None):
    """Test DELETE /api/v1/programs/{id}"""
    print_section("4. Delete Program (DELETE /api/v1/programs/{id})")

    tests_passed = 0
    tests_total = 4

    # Get a program ID if not provided
    if not program_id:
        try:
            response = requests.get(
                f"{BASE_URL}/api/v1/programs?limit=1",
                headers=HEADERS
            )
            if response.status_code == 200:
                data = response.json()
                if data['programs']:
                    program_id = data['programs'][0]['program_id']
                    print(f"Using program ID: {program_id}")
                else:
                    print("No programs available for testing")
                    return False, None
        except Exception as e:
            print(f"Error getting program list: {e}")
            return False, None

    # Test 1: Delete without admin password
    try:
        response = requests.delete(
            f"{BASE_URL}/api/v1/programs/{program_id}",
            headers=HEADERS,
            json={"reason": "Testing delete endpoint - this should fail"}
        )

        if response.status_code == 401:
            print_test("Admin password required", True, "Correctly rejected without admin password")
            tests_passed += 1
        else:
            print_test("Admin password required", False,
                      f"Expected 401, got {response.status_code}")
    except Exception as e:
        print_test("Admin password required", False, str(e))

    # Test 2: Delete with wrong admin password
    try:
        wrong_headers = HEADERS.copy()
        wrong_headers["X-Admin-Password"] = "wrong_password"
        response = requests.delete(
            f"{BASE_URL}/api/v1/programs/{program_id}",
            headers=wrong_headers,
            json={"reason": "Testing delete endpoint - this should fail"}
        )

        if response.status_code == 403:
            print_test("Invalid admin password", True, "Correctly rejected invalid password")
            tests_passed += 1
        else:
            print_test("Invalid admin password", False,
                      f"Expected 403, got {response.status_code}")
    except Exception as e:
        print_test("Invalid admin password", False, str(e))

    # Test 3: Delete with valid admin password
    try:
        response = requests.delete(
            f"{BASE_URL}/api/v1/programs/{program_id}",
            headers=ADMIN_HEADERS,
            json={"reason": "Testing delete endpoint - will be restored"}
        )

        if response.status_code == 200:
            data = response.json()
            print_test("Delete program", True, f"Deleted at: {data['deleted_at']}")
            tests_passed += 1
        else:
            print_test("Delete program", False,
                      f"Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print_test("Delete program", False, str(e))

    # Test 4: Verify program is deleted
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/programs/{program_id}",
            headers=HEADERS
        )

        if response.status_code == 404:
            print_test("Program marked as deleted", True, "Correctly returns 404 for deleted program")
            tests_passed += 1
        else:
            print_test("Program marked as deleted", False,
                      f"Expected 404, got {response.status_code}")
    except Exception as e:
        print_test("Program marked as deleted", False, str(e))

    print(f"\nDelete Program Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total, program_id


def test_restore_program(program_id):
    """Test POST /api/v1/programs/{id}/restore"""
    print_section("5. Restore Program (POST /api/v1/programs/{id}/restore)")

    tests_passed = 0
    tests_total = 3

    # Test 1: Restore without admin password
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/programs/{program_id}/restore",
            headers=HEADERS
        )

        if response.status_code == 401:
            print_test("Admin password required", True, "Correctly rejected without admin password")
            tests_passed += 1
        else:
            print_test("Admin password required", False,
                      f"Expected 401, got {response.status_code}")
    except Exception as e:
        print_test("Admin password required", False, str(e))

    # Test 2: Restore with valid admin password
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/programs/{program_id}/restore",
            headers=ADMIN_HEADERS
        )

        if response.status_code == 200:
            data = response.json()
            print_test("Restore program", True, f"Restored at: {data['restored_at']}")
            tests_passed += 1
        else:
            print_test("Restore program", False,
                      f"Status: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print_test("Restore program", False, str(e))

    # Test 3: Verify program is restored
    try:
        response = requests.get(
            f"{BASE_URL}/api/v1/programs/{program_id}",
            headers=HEADERS
        )

        if response.status_code == 200:
            data = response.json()
            if data['status'] == 'active':
                print_test("Program is active", True, "Program successfully restored to active status")
                tests_passed += 1
            else:
                print_test("Program is active", False, f"Status is {data['status']}, expected 'active'")
        else:
            print_test("Program is active", False,
                      f"Failed to get program: {response.status_code}")
    except Exception as e:
        print_test("Program is active", False, str(e))

    print(f"\nRestore Program Tests: {tests_passed}/{tests_total} passed")
    return tests_passed == tests_total


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("  API Program Management Endpoints Test Suite")
    print("  Testing api_5 implementation")
    print("=" * 60)
    print(f"\nBase URL: {BASE_URL}")
    print(f"API Key: {'*' * (len(API_KEY) - 4) + API_KEY[-4:]}")
    print(f"Admin Password: {'*' * len(ADMIN_PASSWORD)}")

    # Check if server is running
    if not test_health_check():
        print("\n✗ Server not running. Start with: uvicorn api_server:app --reload")
        return 1

    # Run all tests
    all_passed = True

    all_passed &= test_list_programs()
    all_passed &= test_get_program()
    all_passed &= test_create_alias()

    delete_passed, deleted_program_id = test_delete_program()
    all_passed &= delete_passed

    if deleted_program_id:
        all_passed &= test_restore_program(deleted_program_id)

    # Final summary
    print("\n" + "=" * 60)
    if all_passed:
        print("  ✓ ALL TESTS PASSED")
    else:
        print("  ✗ SOME TESTS FAILED")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
