#!/usr/bin/env python3
"""
Test script for validating PDF dictionary extraction improvements
"""

import pytest
import os
import time
from pathlib import Path

from src.llm_client import LLMDictionaryParser
import pypdf


@pytest.fixture
def pdf_path(test_data_dir):
    """Fixture that returns the path to the test PDF file"""
    return test_data_dir / "dictionaries" / "LTCPROMISE8314QC.pdf"


def test_pdf_extraction(pdf_path):
    """Test PDF dictionary extraction with improved settings"""

    print(f"\n📄 Testing PDF extraction for: {pdf_path}")
    print("=" * 60)

    # Check file exists
    assert os.path.exists(pdf_path), f"File not found: {pdf_path}"

    # Read PDF content
    print("📖 Reading PDF content...")
    with open(pdf_path, 'rb') as file:
        pdf_reader = pypdf.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        print(f"  Pages: {num_pages}")

        # Extract text from all pages
        text_content = ""
        for page_num, page in enumerate(pdf_reader.pages):
            text_content += page.extract_text() + "\n"
            if page_num % 10 == 0:
                print(f"  Processed page {page_num + 1}/{num_pages}")

        print(f"  Total characters: {len(text_content):,}")

    assert len(text_content) > 0, "PDF extraction returned empty content"

    # Initialize LLM parser
    print("\n🤖 Initializing LLM parser...")
    parser = LLMDictionaryParser()
    print("✅ LLM parser initialized")

    # Parse dictionary
    print("\n📋 Parsing dictionary with LLM...")
    print(f"  Max fields: 500")
    print(f"  Chunk size: 4500 tokens")

    start_time = time.time()

    # Parse without truncation
    result = parser.parse_dictionary(text_content, max_fields=500)

    elapsed = time.time() - start_time

    print(f"\n✅ Parsing completed in {elapsed:.1f} seconds")
    print(f"  Fields extracted: {len(result.get('fields', []))}")
    print(f"  Chunks processed: {result.get('metadata', {}).get('chunks_processed', 0)}")

    # Show first 20 fields
    fields = result.get('fields', [])
    assert fields is not None, "No fields returned from parser"
    assert len(fields) > 0, "Parser returned empty fields list"

    if fields:
        print("\n📊 Sample of extracted fields:")
        print("-" * 40)
        for i, field in enumerate(fields[:20], 1):
            print(f"{i:3}. {field['field_name']:30} ({field['data_type']})")
            if field.get('required'):
                print(f"     [Required]")
            if field.get('description'):
                desc = field['description'][:60] + "..." if len(field['description']) > 60 else field['description']
                print(f"     {desc}")

        if len(fields) > 20:
            print(f"\n     ... and {len(fields) - 20} more fields")

    # Summary
    print("\n📈 Extraction Summary:")
    print(f"  Total fields: {len(fields)}")
    print(f"  Required fields: {sum(1 for f in fields if f.get('required'))}")
    print(f"  Fields with descriptions: {sum(1 for f in fields if f.get('description'))}")
    print(f"  Fields with allowed values: {sum(1 for f in fields if f.get('allowed_values'))}")

    # Assertions to validate results
    assert len(fields) > 0, "Should extract at least one field"
    assert result.get('metadata', {}).get('chunks_processed', 0) > 0, "Should process at least one chunk"


if __name__ == "__main__":
    # Allow running as script for manual testing
    pytest.main([__file__, "-v", "-s"])