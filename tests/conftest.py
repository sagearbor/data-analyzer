"""
Pytest configuration and shared fixtures for all tests
"""
import pytest
import pandas as pd
import json
import tempfile
import os
from pathlib import Path
import shutil

# Get the tests directory
TESTS_DIR = Path(__file__).parent
DATA_DIR = TESTS_DIR / "data"


@pytest.fixture(scope="session")
def test_data_dir():
    """Return the test data directory path"""
    return DATA_DIR


@pytest.fixture
def sample_csv_data():
    """Sample CSV data for testing"""
    return """id,name,age,department,salary,hire_date,is_active
1,Alice Johnson,25,Engineering,75000.50,2023-01-15,true
2,Bob Smith,30,Marketing,65000.00,2022-06-20,true
3,Charlie Brown,35,Engineering,85000.75,2021-03-10,false
4,Diana Prince,28,HR,70000.00,2022-11-05,true
5,Eve Adams,45,Engineering,95000.25,2019-08-22,true
6,,32,Marketing,68000.00,2023-02-28,true
7,Frank Miller,29,Sales,,2021-07-15,false
8,Grace Lee,31,Engineering,78000.50,2020-invalid-date,true
9,Henry Ford,27,Marketing,62000.00,2023-04-01,invalid
10,Iris West,33,Engineering,82000.00,2022-09-15,true"""


@pytest.fixture
def sample_json_data():
    """Sample nested JSON data simulating clinical trial structure"""
    return {
        "study_id": "TRIAL-001",
        "study_name": "Sample Clinical Trial",
        "sponsor": "Test Pharma Inc",
        "sites": [
            {
                "site_id": "SITE-01",
                "site_name": "Test Hospital A",
                "location": {"country": "USA", "city": "Boston", "state": "MA"},
                "subjects": [
                    {
                        "subject_id": "SUBJ-001",
                        "age": 45,
                        "gender": "M",
                        "enrollment_date": "2023-01-15",
                        "visits": [
                            {
                                "visit_id": "V1",
                                "visit_date": "2023-01-15",
                                "measurements": {
                                    "weight_kg": 75.5,
                                    "height_cm": 175,
                                    "blood_pressure": {"systolic": 120, "diastolic": 80}
                                }
                            },
                            {
                                "visit_id": "V2",
                                "visit_date": "2023-02-15",
                                "measurements": {
                                    "weight_kg": 74.8,
                                    "height_cm": 175,
                                    "blood_pressure": {"systolic": 118, "diastolic": 78}
                                }
                            }
                        ]
                    },
                    {
                        "subject_id": "SUBJ-002",
                        "age": 52,
                        "gender": "F",
                        "enrollment_date": "2023-01-20",
                        "visits": [
                            {
                                "visit_id": "V1",
                                "visit_date": "2023-01-20",
                                "measurements": {
                                    "weight_kg": 68.2,
                                    "height_cm": 162,
                                    "blood_pressure": {"systolic": 125, "diastolic": 82}
                                }
                            }
                        ]
                    }
                ]
            },
            {
                "site_id": "SITE-02",
                "site_name": "Test Hospital B",
                "location": {"country": "USA", "city": "New York", "state": "NY"},
                "subjects": [
                    {
                        "subject_id": "SUBJ-003",
                        "age": 38,
                        "gender": "M",
                        "enrollment_date": "2023-02-01",
                        "visits": []
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_excel_data():
    """Create a sample Excel file with multiple sheets"""
    data_sheet1 = {
        'product_id': [1, 2, 3, 4, 5],
        'product_name': ['Widget A', 'Widget B', 'Gadget C', 'Tool D', 'Device E'],
        'price': [19.99, 29.99, 39.99, 49.99, 59.99],
        'quantity': [100, 75, 50, 25, 10],
        'category': ['Hardware', 'Hardware', 'Software', 'Hardware', 'Software']
    }

    data_sheet2 = {
        'customer_id': [101, 102, 103, 104],
        'customer_name': ['Acme Corp', 'Beta LLC', 'Gamma Inc', 'Delta Co'],
        'country': ['USA', 'Canada', 'UK', 'Germany'],
        'credit_limit': [10000, 15000, 12000, 20000],
        'active': [True, True, False, True]
    }

    return {'products': pd.DataFrame(data_sheet1), 'customers': pd.DataFrame(data_sheet2)}


@pytest.fixture
def sample_parquet_data():
    """Sample data for Parquet format"""
    return pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
        'sensor_id': ['SENSOR-' + str(i % 5 + 1) for i in range(100)],
        'temperature': [20 + (i % 10) * 0.5 for i in range(100)],
        'humidity': [45 + (i % 20) * 1.5 for i in range(100)],
        'pressure': [1013.25 + (i % 15) * 0.1 for i in range(100)],
        'status': ['OK' if i % 7 != 0 else 'WARNING' for i in range(100)]
    })


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    shutil.rmtree(temp_dir)


@pytest.fixture
def create_test_csv(temp_dir, sample_csv_data):
    """Create a temporary CSV file for testing"""
    file_path = os.path.join(temp_dir, "test_data.csv")
    with open(file_path, 'w') as f:
        f.write(sample_csv_data)
    return file_path


@pytest.fixture
def create_test_json(temp_dir, sample_json_data):
    """Create a temporary JSON file for testing"""
    file_path = os.path.join(temp_dir, "test_data.json")
    with open(file_path, 'w') as f:
        json.dump(sample_json_data, f, indent=2)
    return file_path


@pytest.fixture
def create_test_excel(temp_dir, sample_excel_data):
    """Create a temporary Excel file for testing"""
    file_path = os.path.join(temp_dir, "test_data.xlsx")
    with pd.ExcelWriter(file_path) as writer:
        for sheet_name, df in sample_excel_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return file_path


@pytest.fixture
def create_test_parquet(temp_dir, sample_parquet_data):
    """Create a temporary Parquet file for testing"""
    file_path = os.path.join(temp_dir, "test_data.parquet")
    sample_parquet_data.to_parquet(file_path)
    return file_path


@pytest.fixture
def mock_schema():
    """Sample schema for validation testing"""
    return {
        'id': 'int',
        'name': 'str',
        'age': 'int',
        'department': 'str',
        'salary': 'float',
        'hire_date': 'datetime',
        'is_active': 'bool'
    }


@pytest.fixture
def mock_rules():
    """Sample validation rules for testing"""
    return {
        'age': {'min': 18, 'max': 65},
        'salary': {'min': 30000, 'max': 200000},
        'department': {'allowed': ['Engineering', 'Marketing', 'HR', 'Sales']}
    }


@pytest.fixture
def mock_mcp_request():
    """Mock MCP protocol request"""
    return {
        "jsonrpc": "2.0",
        "id": "test-1",
        "method": "tools/call",
        "params": {
            "name": "analyze_data",
            "arguments": {
                "file_format": "csv",
                "min_rows": 5
            }
        }
    }


# Async fixtures for MCP testing
@pytest.fixture
async def mock_mcp_server():
    """Mock MCP server for testing"""
    from mcp_server import create_server
    server = await create_server()
    yield server
    # Cleanup if needed


# Markers for test organization
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "mcp: mark test as MCP server test"
    )
    config.addinivalue_line(
        "markers", "api: mark test as API test"
    )