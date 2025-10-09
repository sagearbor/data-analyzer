#!/usr/bin/env python3
"""
Test script for validating PDF dictionary extraction improvements
"""

import sys
import os
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.llm_client import LLMDictionaryParser
import PyPDF2

def test_pdf_extraction(pdf_path):
    """Test PDF dictionary extraction with improved settings"""

    print(f"\n📄 Testing PDF extraction for: {pdf_path}")
    print("=" * 60)

    # Check file exists
    if not os.path.exists(pdf_path):
        print(f"❌ File not found: {pdf_path}")
        return

    # Read PDF content
    print("📖 Reading PDF content...")
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        print(f"  Pages: {num_pages}")

        # Extract text from all pages
        text_content = ""
        for page_num, page in enumerate(pdf_reader.pages):
            text_content += page.extract_text() + "\n"
            if page_num % 10 == 0:
                print(f"  Processed page {page_num + 1}/{num_pages}")

        print(f"  Total characters: {len(text_content):,}")

    # Initialize LLM parser
    print("\n🤖 Initializing LLM parser...")
    try:
        parser = LLMDictionaryParser()
        print("✅ LLM parser initialized")
    except Exception as e:
        print(f"❌ Failed to initialize LLM parser: {e}")
        return

    # Parse dictionary
    print("\n📋 Parsing dictionary with LLM...")
    print(f"  Max fields: 500")
    print(f"  Chunk size: 4500 tokens")

    start_time = time.time()

    try:
        # Parse without truncation
        result = parser.parse_dictionary(text_content, max_fields=500)

        elapsed = time.time() - start_time

        print(f"\n✅ Parsing completed in {elapsed:.1f} seconds")
        print(f"  Fields extracted: {len(result.get('fields', []))}")
        print(f"  Chunks processed: {result.get('metadata', {}).get('chunks_processed', 0)}")

        # Show first 20 fields
        fields = result.get('fields', [])
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

        return result

    except Exception as e:
        print(f"\n❌ Error during parsing: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run test on the specific PDF"""

    # Test with the large LTCPROMISE PDF
    pdf_path = "test_dictionaries/LTCPROMISE8314QC.pdf"

    print("\n🚀 Starting PDF extraction test")
    print("This test validates improvements to handle large clinical data dictionaries")

    result = test_pdf_extraction(pdf_path)

    if result:
        print("\n✅ Test completed successfully!")
        print(f"Extracted {len(result.get('fields', []))} fields from {pdf_path}")
    else:
        print("\n❌ Test failed - check errors above")

    print("\n" + "=" * 60)
    print("Test complete!")

if __name__ == "__main__":
    main()