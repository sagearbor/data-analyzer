#!/usr/bin/env python3
"""
Test script for LLM dictionary parsing integration
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm_client import LLMDictionaryParser

def test_csv_dictionary():
    """Test parsing a CSV dictionary"""
    # Read the CSV dictionary
    csv_path = Path("tests/test_data/dictionaries/simple_csv_dict.csv")
    assert csv_path.exists(), f"Test file not found: {csv_path}"

    with open(csv_path, 'r') as f:
        csv_content = f.read()

    # Parse with LLM
    parser = LLMDictionaryParser()
    result = parser.parse_dictionary(csv_content)

    # Assertions instead of return
    assert 'fields' in result, "Result should contain 'fields' key"
    assert len(result['fields']) > 0, "Should extract at least one field"

def test_pdf_dictionary():
    """Test parsing a PDF dictionary (first page only for speed)"""
    import pytest

    pdf_path = Path("tests/test_data/dictionaries/LTCPROMISE8314QC.pdf")
    if not pdf_path.exists():
        pytest.skip(f"PDF test file not found: {pdf_path}")

    try:
        import pypdf  # Use pypdf instead of deprecated PyPDF2
    except ImportError:
        pytest.skip("pypdf not installed")

    with open(pdf_path, 'rb') as f:
        pdf_reader = pypdf.PdfReader(f)
        # Extract first 3 pages for testing
        text = ""
        for i in range(min(3, len(pdf_reader.pages))):
            text += pdf_reader.pages[i].extract_text() + "\n"

    # Parse with LLM
    parser = LLMDictionaryParser()
    result = parser.parse_dictionary(text)

    # Assertions instead of return
    assert 'fields' in result, "Result should contain 'fields' key"
    assert len(result['fields']) > 0, "Should extract at least one field from PDF"

def test_data_validation():
    """Test LLM-based data validation"""
    # Sample data with intentional issues
    sample_data = """employee_id,first_name,last_name,age,salary,department
999999999,John,Doe,150,25000,IT
1,Jane,,25,45000,Engineering
2,Bob,Smith,17,600000,Sales
3,Alice,Johnson,30,75000,Unknown Dept"""

    # Sample schema
    schema = {
        "employee_id": {"type": "int", "min": 1, "max": 999999},
        "first_name": {"type": "str", "required": True},
        "last_name": {"type": "str", "required": True},
        "age": {"type": "int", "min": 18, "max": 100},
        "salary": {"type": "float", "min": 30000, "max": 500000},
        "department": {"type": "str", "allowed_values": ["HR", "Engineering", "Sales", "Marketing"]}
    }

    # Validate with LLM
    parser = LLMDictionaryParser()
    validation_result = parser.validate_data_with_llm(sample_data, schema)

    # Assertions instead of return
    assert 'issues' in validation_result or 'summary' in validation_result, "Should return validation results"
    # The sample data has intentional issues, so we should find some
    assert len(validation_result.get('issues', [])) > 0, "Should detect issues in invalid data"

def main():
    """Run all tests"""
    print("\n🤖 LLM Dictionary Parser Integration Tests")
    print("Using Azure OpenAI GPT-4")

    # Test CSV dictionary parsing
    csv_result = test_csv_dictionary()

    # Test PDF dictionary parsing (optional)
    # pdf_result = test_pdf_dictionary()

    # Test data validation
    validation_result = test_data_validation()

    print("\n" + "="*60)
    print("✅ All tests completed!")
    print("="*60)

if __name__ == "__main__":
    main()
