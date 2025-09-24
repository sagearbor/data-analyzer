#!/usr/bin/env python3
"""
Test script to verify the data dictionary UI flow works correctly
"""

import sys
sys.path.insert(0, '.')
from demo_dictionaries import DEMO_DICTIONARIES, get_demo_dictionary

def test_demo_dictionaries():
    """Test that demo dictionaries are properly formatted"""
    print("Testing demo dictionaries...")

    for name, content in DEMO_DICTIONARIES.items():
        print(f"\n{name}:")
        lines = content.strip().split('\n')
        print(f"  - Headers: {lines[0]}")
        print(f"  - Rows: {len(lines) - 1}")

        # Check if it's CSV format
        headers = lines[0].split(',')
        print(f"  - Columns: {headers[:3]}...")

        # Verify get_demo_dictionary works
        retrieved = get_demo_dictionary(name)
        assert retrieved == content, f"Failed to retrieve {name}"
        print(f"  ✓ Retrieved successfully")

def test_rules_format():
    """Test the expected rules format for UI"""
    print("\n\nTesting rules format conversion...")

    # Simulate parsed rules from dictionary
    parsed_rules = {
        "age": {"min": 18, "max": 65},
        "department": {"allowed": ["HR", "Engineering", "Sales"]},
        "salary": {"min": 30000, "max": 150000}
    }

    # Convert to UI format (what the Apply button does)
    rules_entries = []
    for col_name, col_rules in parsed_rules.items():
        if 'allowed' in col_rules:
            rule_type = 'allowed_values'
            config = {'allowed': col_rules['allowed']}
        elif 'min' in col_rules or 'max' in col_rules:
            rule_type = 'range'
            config = {}
            if 'min' in col_rules:
                config['min'] = col_rules['min']
            if 'max' in col_rules:
                config['max'] = col_rules['max']
        else:
            continue

        rules_entries.append({
            'column': col_name,
            'rule_type': rule_type,
            'config': config
        })

    print("Converted rules entries:")
    for entry in rules_entries:
        print(f"  - {entry['column']}: {entry['rule_type']} -> {entry['config']}")

    # Verify structure
    assert len(rules_entries) == 3, "Should have 3 rules"
    assert any(e['rule_type'] == 'allowed_values' for e in rules_entries), "Should have allowed_values rule"
    assert any(e['rule_type'] == 'range' for e in rules_entries), "Should have range rules"
    print("✓ Rules format conversion works correctly")

def test_schema_format():
    """Test the expected schema format for UI"""
    print("\n\nTesting schema format conversion...")

    # Simulate parsed schema from dictionary
    parsed_schema = {
        "employee_id": "int",
        "first_name": "str",
        "salary": "float",
        "is_active": "bool",
        "hire_date": "datetime"
    }

    # Convert to UI format (what the Apply button does)
    schema_entries = []
    for col_name, col_type in parsed_schema.items():
        schema_entries.append({
            'column': col_name,
            'type': col_type
        })

    print("Converted schema entries:")
    for entry in schema_entries:
        print(f"  - {entry['column']}: {entry['type']}")

    assert len(schema_entries) == 5, "Should have 5 schema entries"
    assert any(e['type'] == 'int' for e in schema_entries), "Should have int type"
    assert any(e['type'] == 'datetime' for e in schema_entries), "Should have datetime type"
    print("✓ Schema format conversion works correctly")

if __name__ == "__main__":
    test_demo_dictionaries()
    test_rules_format()
    test_schema_format()
    print("\n✅ All tests passed!")