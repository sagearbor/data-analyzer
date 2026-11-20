#!/usr/bin/env python3
"""
Inspect the dictionary cache written by web_app.py (JSON files only).

The UI keeps one private folder per browser session under
`$DICT_CACHE_DIR` (default: <system temp>/data_analyzer_ui_cache_v2/<session>/).
This tool only reads `*.json` files below that root. It never unpickles
anything (loading an untrusted pickle can run code), so entries left behind by
very old versions in the old format are simply ignored.

Usage: python scripts/inspect_cache.py [cache_root]
"""
import json
import os
import sys
import tempfile
from pathlib import Path


def default_root() -> Path:
    override = (os.environ.get("DICT_CACHE_DIR") or "").strip()
    return Path(override) if override else Path(tempfile.gettempdir()) / "data_analyzer_ui_cache_v2"


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else default_root()
    print("=" * 80)
    print(f"INSPECTING CACHED DICTIONARIES under {root}")
    print("=" * 80)

    for cache_file in sorted(root.glob("**/*.json")):
        print(f"\nCache file: {cache_file.name}")
        print("-" * 80)
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                print("Not a dictionary entry; skipped.")
                continue

            print(f"Source: {data.get('source', 'Unknown')}")
            print(f"Filename: {data.get('filename', 'Unknown')}")

            rules = data.get("rules", {})
            print(f"\nRules count: {len(rules)}")
            if isinstance(rules, dict):
                print("\nFirst 5 rules:")
                for field_name, rule in list(rules.items())[:5]:
                    print(f"\n  {field_name}:")
                    print(f"    {json.dumps(rule, indent=6, default=str)}")

            fields = data.get("fields", [])
            print(f"\nFields count: {len(fields)}")
            if isinstance(fields, list):
                print("\nFirst 5 fields:")
                for i, field in enumerate(fields[:5]):
                    if not isinstance(field, dict):
                        continue
                    print(f"\n  {i + 1}. {field.get('field_name', 'Unknown')}:")
                    print(f"     Type: {field.get('data_type', 'Unknown')}")
                    print(f"     Required: {field.get('required', 'Unknown')}")
                    if field.get("allowed_values"):
                        print(f"     Allowed: {field['allowed_values'][:5]}")
                    if field.get("min_value") or field.get("max_value"):
                        print(f"     Range: {field.get('min_value')} - {field.get('max_value')}")

            metadata = data.get("metadata", {})
            if metadata:
                print("\nMetadata:")
                print(f"  Processing time: {metadata.get('processing_time_seconds', 'N/A')}s")
                print(f"  Chunks processed: {metadata.get('chunks_processed', 'N/A')}")
        except (OSError, ValueError) as e:
            print(f"Error reading cache: {type(e).__name__}")

    print("\n" + "=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
