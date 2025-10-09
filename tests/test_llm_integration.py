#!/usr/bin/env python3
"""
Test script for LLM dictionary parsing integration
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm_client import LLMDictionaryParser

def test_csv_dictionary():
    """Test parsing a CSV dictionary"""
    print("\n" + "="*60)
    print("Testing CSV Dictionary Parsing with LLM")
    print("="*60)

    # Read the CSV dictionary
    csv_path = Path("test_dictionaries/simple_csv_dict.csv")
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        return

    with open(csv_path, 'r') as f:
        csv_content = f.read()

    print(f"\nInput CSV Dictionary ({len(csv_content)} chars):")
    print("-" * 40)
    print(csv_content[:500])
    if len(csv_content) > 500:
        print("...")

    # Parse with LLM
    parser = LLMDictionaryParser()
    result = parser.parse_dictionary(csv_content)

    print(f"\n✅ Extracted {len(result['fields'])} field definitions")
    print("-" * 40)

    # Display fields
    for i, field in enumerate(result['fields'][:5], 1):
        print(f"\n{i}. {field['field_name']} ({field['data_type']})")
        if field.get('description'):
            print(f"   Description: {field['description']}")
        if field.get('required'):
            print(f"   Required: Yes")
        if field.get('min_value') or field.get('max_value'):
            print(f"   Range: {field.get('min_value', 'N/A')} to {field.get('max_value', 'N/A')}")
        if field.get('allowed_values'):
            print(f"   Allowed: {', '.join(field['allowed_values'][:5])}")

    if len(result['fields']) > 5:
        print(f"\n... and {len(result['fields']) - 5} more fields")

    return result

def test_pdf_dictionary():
    """Test parsing a PDF dictionary (first page only for speed)"""
    print("\n" + "="*60)
    print("Testing PDF Dictionary Parsing with LLM")
    print("="*60)

    pdf_path = Path("test_dictionaries/LTCPROMISE8314QC.pdf")
    if not pdf_path.exists():
        print(f"ERROR: {pdf_path} not found")
        return

    try:
        import PyPDF2

        with open(pdf_path, 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            num_pages = len(pdf_reader.pages)
            print(f"\nPDF has {num_pages} pages")

            # Extract first 3 pages for testing
            text = ""
            for i in range(min(3, num_pages)):
                text += pdf_reader.pages[i].extract_text() + "\n"

        print(f"\nExtracted text from first 3 pages ({len(text)} chars):")
        print("-" * 40)
        print(text[:500])
        if len(text) > 500:
            print("...")

        # Parse with LLM
        parser = LLMDictionaryParser()
        result = parser.parse_dictionary(text)

        print(f"\n✅ Extracted {len(result['fields'])} field definitions from PDF")
        print("-" * 40)

        # Display fields
        for i, field in enumerate(result['fields'][:5], 1):
            print(f"\n{i}. {field['field_name']} ({field['data_type']})")
            if field.get('description'):
                print(f"   Description: {field['description'][:100]}...")
            if field.get('required'):
                print(f"   Required: Yes")
            if field.get('min_value') or field.get('max_value'):
                print(f"   Range: {field.get('min_value', 'N/A')} to {field.get('max_value', 'N/A')}")

        if len(result['fields']) > 5:
            print(f"\n... and {len(result['fields']) - 5} more fields")

        return result

    except ImportError:
        print("PyPDF2 not installed. Skipping PDF test.")
        return None

def test_data_validation():
    """Test LLM-based data validation"""
    print("\n" + "="*60)
    print("Testing LLM-Based Data Validation")
    print("="*60)

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

    print("\nSample Data:")
    print("-" * 40)
    print(sample_data)

    print("\nSchema:")
    print("-" * 40)
    print(json.dumps(schema, indent=2))

    # Validate with LLM
    parser = LLMDictionaryParser()
    validation_result = parser.validate_data_with_llm(sample_data, schema)

    print("\n✅ LLM Validation Results:")
    print("-" * 40)

    if validation_result.get('issues'):
        print(f"\nFound {len(validation_result['issues'])} issues:")
        for issue in validation_result['issues']:
            print(f"  • {issue}")

    if validation_result.get('summary'):
        print(f"\nSummary: {validation_result['summary']}")

    if validation_result.get('recommendations'):
        print(f"\nRecommendations:")
        for rec in validation_result['recommendations']:
            print(f"  • {rec}")

    return validation_result

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