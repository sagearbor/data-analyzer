#!/usr/bin/env python3
"""
Test the DataDictionaryParser with various dictionary formats
"""

import asyncio
import json
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from core.data_dictionary_parser import DataDictionaryParser

async def test_parser():
    """Test the parser with different dictionary formats"""
    parser = DataDictionaryParser()

    # Test 1: Simple CSV dictionary
    print("\n" + "="*60)
    print("Testing CSV Dictionary")
    print("="*60)

    csv_dict = Path("../test_dictionaries/simple_csv_dict.csv").read_text()
    result = await parser.parse_dictionary(csv_dict, format_hint="csv", use_cache=False, debug=True)

    if result.get("success"):
        print(f"✅ SUCCESS - Confidence: {result.get('confidence_score', 0):.1%}")
        print(f"Schema columns: {len(result.get('schema', {}))}")
        print(f"Rules defined: {len(result.get('rules', {}))}")

        # Convert to schema and rules
        schema, rules = parser.convert_to_schema_and_rules(result)
        print(f"\nExtracted Schema:")
        for col, dtype in list(schema.items())[:5]:
            print(f"  {col}: {dtype}")
        print(f"\nExtracted Rules (sample):")
        for col, col_rules in list(rules.items())[:3]:
            print(f"  {col}: {col_rules}")
    else:
        print(f"❌ FAILED: {result.get('error')}")

    # Test 2: Clinical JSON dictionary
    print("\n" + "="*60)
    print("Testing Clinical Trial JSON Dictionary")
    print("="*60)

    json_dict = Path("../test_dictionaries/clinical_trial_dict.json").read_text()
    result = await parser.parse_dictionary(json_dict, format_hint="json", use_cache=False, debug=True)

    if result.get("success"):
        print(f"✅ SUCCESS - Confidence: {result.get('confidence_score', 0):.1%}")
        print(f"Schema columns: {len(result.get('schema', {}))}")
        print(f"Rules defined: {len(result.get('rules', {}))}")

        validation = result.get("validation", {})
        if validation.get("warnings"):
            print(f"⚠️ Warnings: {len(validation['warnings'])}")
    else:
        print(f"❌ FAILED: {result.get('error')}")

    # Test 3: REDCap style text dictionary
    print("\n" + "="*60)
    print("Testing REDCap Style Text Dictionary")
    print("="*60)

    redcap_dict = Path("../test_dictionaries/redcap_style_dict.txt").read_text()
    result = await parser.parse_dictionary(redcap_dict, format_hint="redcap", use_cache=False, debug=True)

    if result.get("success"):
        print(f"✅ SUCCESS - Confidence: {result.get('confidence_score', 0):.1%}")
        print(f"Schema columns: {len(result.get('schema', {}))}")

        # Show some clinical-specific fields
        metadata = result.get("metadata", {})
        columns = result.get("columns", {})
        clinical_fields = [k for k, v in columns.items() if v.get("clinical_type")]
        if clinical_fields:
            print(f"Clinical fields detected: {clinical_fields[:5]}")
    else:
        print(f"❌ FAILED: {result.get('error')}")

    # Test 4: Cache functionality
    print("\n" + "="*60)
    print("Testing Cache Functionality")
    print("="*60)

    # Parse once with caching
    result1 = await parser.parse_dictionary(csv_dict, format_hint="csv", use_cache=True, debug=False)
    print("First parse complete (cached)")

    # Parse again - should use cache
    result2 = await parser.parse_dictionary(csv_dict, format_hint="csv", use_cache=True, debug=False)
    print("Second parse complete (from cache)")

    # Check if results are identical
    if json.dumps(result1.get("schema"), sort_keys=True) == json.dumps(result2.get("schema"), sort_keys=True):
        print("✅ Cache working correctly - identical results")
    else:
        print("❌ Cache issue - results differ")

    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(test_parser())