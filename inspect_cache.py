#!/usr/bin/env python3
"""
Inspect cached dictionary to see what was extracted
"""
import pickle
from pathlib import Path
import json

cache_dir = Path.home() / '.data_analyzer_cache'

print("="*80)
print("INSPECTING CACHED DICTIONARIES")
print("="*80)

for cache_file in cache_dir.glob("*.pkl"):
    print(f"\n📦 Cache File: {cache_file.name}")
    print("-" * 80)

    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)

        print(f"Source: {data.get('source', 'Unknown')}")
        print(f"Filename: {data.get('filename', 'Unknown')}")

        # Check for rules
        rules = data.get('rules', {})
        print(f"\nRules count: {len(rules)}")

        if rules:
            print("\nFirst 5 rules:")
            for i, (field_name, rule) in enumerate(list(rules.items())[:5]):
                print(f"\n  {field_name}:")
                print(f"    {json.dumps(rule, indent=6, default=str)}")

        # Check for fields (LLM format)
        fields = data.get('fields', [])
        print(f"\nFields count: {len(fields)}")

        if fields:
            print("\nFirst 5 fields:")
            for i, field in enumerate(fields[:5]):
                print(f"\n  {i+1}. {field.get('field_name', 'Unknown')}:")
                print(f"     Type: {field.get('data_type', 'Unknown')}")
                print(f"     Required: {field.get('required', 'Unknown')}")
                if field.get('allowed_values'):
                    print(f"     Allowed: {field['allowed_values'][:5]}")
                if field.get('min_value') or field.get('max_value'):
                    print(f"     Range: {field.get('min_value')} - {field.get('max_value')}")

        # Check metadata
        metadata = data.get('metadata', {})
        if metadata:
            print(f"\nMetadata:")
            print(f"  Processing time: {metadata.get('processing_time_seconds', 'N/A')}s")
            print(f"  Chunks processed: {metadata.get('chunks_processed', 'N/A')}")

    except Exception as e:
        print(f"❌ Error reading cache: {e}")

print("\n" + "="*80)
