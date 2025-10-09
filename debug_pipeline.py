#!/usr/bin/env python3
"""
Debug script to see QualityPipeline output format
"""

import pandas as pd
from mcp_server import QualityPipeline
import json

# Create demo data
demo_data = pd.DataFrame({
    'employee_id': [1001, 1002, 1003],
    'age': [35, 28, 67],  # 67 is outside range
    'salary': [75000, 85000, 45000],  # 45000 below min
    'department': ['Engineering', 'Marketing', 'InvalidDept'],  # InvalidDept not in allowed
})

# Create schema and rules
schema = {
    'employee_id': 'int',
    'age': 'int',
    'salary': 'float',
    'department': 'str'
}

rules = {
    'age': {
        'min': 18,
        'max': 65
    },
    'salary': {
        'min': 50000,
        'max': 150000
    },
    'department': {
        'allowed': ['Engineering', 'Marketing', 'Sales', 'Finance', 'HR', 'Management']
    }
}

print("=" * 80)
print("DIRECT QUALITY PIPELINE TEST")
print("=" * 80)

pipeline = QualityPipeline(demo_data, schema=schema, rules=rules)
results = pipeline.run_all_checks(min_rows=1)

print("\nFull results structure:")
print(json.dumps(results, indent=2, default=str))

print("\n" + "=" * 80)
print(f"Total issues: {len(results.get('issues', []))}")
print("=" * 80)

for i, issue in enumerate(results.get('issues', []), 1):
    print(f"\nIssue {i}:")
    print(json.dumps(issue, indent=2, default=str))
