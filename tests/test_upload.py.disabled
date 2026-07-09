#!/usr/bin/env python3
"""
Test script to verify the Data Quality Analyzer with LTCPROMISE test file
"""

import pandas as pd
import asyncio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the MCPClient class from web_app
from web_app import MCPClient

async def test_ltcpromise_file():
    """Test the LTCPROMISE file with errors"""

    # Load the test file
    test_file = "/home/scb2/PROJECTS/gitRepos/data-analyzer/test_data_files/LTCPROMISE8314QC_DATA_2025-09-24_0657-2rowErrors.csv"
    df = pd.read_csv(test_file)

    print(f"Loaded file with shape: {df.shape}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")

    # Create MCP client and analyze
    client = MCPClient()
    results = await client.analyze_data_quality(df, None)

    # Print summary
    print("\n=== Analysis Summary ===")
    summary = results['summary']
    print(f"Total Rows: {summary['total_rows']}")
    print(f"Total Columns: {summary['total_columns']}")
    print(f"Issues Found: {summary['issues_found']}")
    print(f"Critical Issues: {summary['critical_issues']}")
    print(f"Warnings: {summary['warnings']}")

    # Look for specific errors
    print("\n=== Looking for Known Errors ===")
    issues = results['issues']

    # Find issues related to the errors we introduced
    date_errors = [i for i in issues if 'oops' in str(i.get('value', '')).lower()]
    value_666_errors = [i for i in issues if '666' in str(i.get('value', ''))]

    print(f"Found {len(date_errors)} date errors with 'oops'")
    for err in date_errors[:3]:  # Show first 3
        print(f"  - {err['message']}")

    print(f"Found {len(value_666_errors)} errors/warnings with '666'")
    for err in value_666_errors[:3]:  # Show first 3
        print(f"  - {err['message']}")

    # Check issue types
    print("\n=== Issue Types ===")
    issue_types = {}
    for issue in issues:
        issue_type = issue['type']
        issue_types[issue_type] = issue_types.get(issue_type, 0) + 1

    for issue_type, count in sorted(issue_types.items()):
        print(f"{issue_type}: {count}")

    return results

if __name__ == "__main__":
    results = asyncio.run(test_ltcpromise_file())
    print(f"\n✅ Test complete - found {results['summary']['issues_found']} total issues")