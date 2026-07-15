#!/usr/bin/env python3
"""
Test Dictionary Parser - Quick command-line tool to test dictionary parsing

Usage:
    python test_dictionary.py <dictionary_file>

Examples:
    python test_dictionary.py tests/test_data/dictionaries/synthetic/redcap_test-case-01-data-dictionary.csv
    python test_dictionary.py tests/test_data/dictionaries/synthetic/fhir_hl7_builder_example.json
    python test_dictionary.py tests/test_data/dictionaries/synthetic/CDISC_ODM_example_1.xml
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm_client import LLMDictionaryParser
from src.logic_engine import RuleExtractor


def read_dictionary_file(file_path: Path) -> str:
    """Read dictionary file content (handles JSON, CSV, XML, TXT, PDF)"""

    # Handle PDF files specially
    if file_path.suffix.lower() == '.pdf':
        try:
            import pypdf
        except ImportError:
            raise ImportError("pypdf library required for PDF files. Install with: pip install pypdf")

        print(f"  📄 Extracting text from PDF...")
        with open(file_path, 'rb') as f:
            pdf_reader = pypdf.PdfReader(f)
            total_pages = len(pdf_reader.pages)
            print(f"  📄 PDF has {total_pages} pages")

            # Extract all pages
            text = ""
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                text += page_text + "\n"
                if (i + 1) % 10 == 0:
                    print(f"  📄 Extracted {i + 1}/{total_pages} pages...")

            print(f"  ✅ Extracted {len(text):,} characters from {total_pages} pages")
            return text

    # Handle text-based files (JSON, CSV, XML, TXT)
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content


def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


def print_fields(fields: list):
    """Print extracted fields in a readable format"""
    print(f"Total fields extracted: {len(fields)}\n")

    for i, field in enumerate(fields, 1):
        print(f"Field {i}: {field.get('field_name', 'UNNAMED')}")
        print(f"  Type: {field.get('data_type', 'unknown')}")
        print(f"  Required: {field.get('required', False)}")

        if field.get('description'):
            desc = field['description'][:100]
            if len(field['description']) > 100:
                desc += "..."
            print(f"  Description: {desc}")

        if field.get('allowed_values'):
            values = field['allowed_values']
            if len(values) > 5:
                print(f"  Allowed Values: {values[:5]} ... ({len(values)} total)")
            else:
                print(f"  Allowed Values: {values}")

        if field.get('min_value') is not None:
            print(f"  Min Value: {field['min_value']}")

        if field.get('max_value') is not None:
            print(f"  Max Value: {field['max_value']}")

        # IMPORTANT: Show conditional rules extracted by LLM
        if field.get('conditional_rules'):
            print(f"  Conditional Rules: {len(field['conditional_rules'])} found")
            for rule in field['conditional_rules']:
                print(f"    - {rule.get('rule_type', 'unknown')}: {rule.get('condition_text', 'N/A')}")
                print(f"      Action: {rule.get('action', 'N/A')}")

        print()


def print_conditional_rules(rules: list):
    """Print ConditionalRule objects in readable format"""
    print(f"Total conditional rules extracted: {len(rules)}\n")

    for i, rule in enumerate(rules, 1):
        print(f"Rule {i}: {rule.rule_id}")
        print(f"  Type: {rule.rule_type}")
        print(f"  Description: {rule.description}")
        print(f"  Condition (natural language): {rule.source[:80]}")
        print(f"  Condition (Python): {rule.condition}")
        print(f"  Action: {rule.action}")
        print(f"  Affected Fields: {', '.join(rule.affected_fields)}")
        print(f"  Severity: {rule.severity}")
        print(f"  Confidence: {rule.confidence}")
        print()


def test_dictionary(file_path: str):
    """Test dictionary parsing with LLM"""

    # Validate file exists
    path = Path(file_path)
    if not path.exists():
        print(f"❌ Error: File not found: {file_path}")
        return 1

    print_section(f"Testing Dictionary Parser: {path.name}")

    print(f"📄 File: {path}")
    print(f"📏 Size: {path.stat().st_size:,} bytes")
    print(f"🔤 Format: {path.suffix.upper()}")

    try:
        # Read dictionary content
        print("\n⏳ Reading dictionary file...")
        content = read_dictionary_file(path)
        print(f"✅ Read {len(content):,} characters")

        # Parse with LLM
        print("\n⏳ Parsing dictionary with LLM (this may take 30-60 seconds)...")
        parser = LLMDictionaryParser()
        result = parser.parse_dictionary(content)

        # Show results
        print_section("Extracted Fields")
        print_fields(result.get('fields', []))

        # Show metadata
        print_section("Metadata")
        metadata = result.get('metadata', {})
        for key, value in metadata.items():
            print(f"  {key}: {value}")

        # Extract conditional rules using RuleExtractor
        print_section("Conditional Logic Extraction")
        print("⏳ Converting LLM-extracted rules to ConditionalRule objects...")

        extractor = RuleExtractor()

        # Try auto-detecting format
        format_type = "Custom"  # Default to custom (uses LLM extraction)
        if 'redcap' in path.name.lower():
            format_type = "REDCap"
        elif 'fhir' in path.name.lower():
            format_type = "FHIR"

        rules = extractor.extract_rules_from_fields(
            result.get('fields', []),
            format_type=format_type
        )

        if rules:
            print(f"✅ Successfully extracted {len(rules)} conditional rules\n")
            print_conditional_rules(rules)
        else:
            print("ℹ️  No conditional rules found in this dictionary")
            print("   (This is normal if the dictionary has no conditional logic)")

        # Success summary
        print_section("✅ Test Complete")
        print(f"Fields Extracted: {len(result.get('fields', []))}")
        print(f"Conditional Rules: {len(rules)}")
        print(f"Processing Time: {metadata.get('processing_time_seconds', 'N/A')} seconds")

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return 1


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print(__doc__)
        print("\n❌ Error: Please provide a dictionary file path")
        print("\nQuick test commands:")
        print("  python test_dictionary.py tests/test_data/dictionaries/synthetic/redcap_test-case-01-data-dictionary.csv")
        print("  python test_dictionary.py tests/test_data/dictionaries/synthetic/fhir_hl7_builder_example.json")
        print("  python test_dictionary.py tests/test_data/dictionaries/synthetic/CDISC_ODM_example_1.xml")
        return 1

    file_path = sys.argv[1]
    return test_dictionary(file_path)


if __name__ == "__main__":
    sys.exit(main())
