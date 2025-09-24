#!/usr/bin/env python3
"""
Test script to verify navigation structure is correct
"""

import ast

def check_navigation_structure():
    """Check that the navigation functions are properly defined"""
    with open('../web_app.py', 'r') as f:
        content = f.read()

    # Check for key functions
    functions = [
        'render_about_page',
        'render_home_page',
        'main'
    ]

    print("Checking navigation structure...")
    for func_name in functions:
        if f'def {func_name}(' in content:
            print(f"✓ Found function: {func_name}")
        else:
            print(f"✗ Missing function: {func_name}")

    # Check for navigation buttons
    nav_elements = [
        '🏠 Home',
        'ℹ️ About',
        'st.session_state.page'
    ]

    print("\nChecking navigation elements...")
    for element in nav_elements:
        if element in content:
            print(f"✓ Found element: {element}")
        else:
            print(f"✗ Missing element: {element}")

    # Check page routing logic
    if 'if st.session_state.page == "about"' in content:
        print("✓ Found page routing logic")
    else:
        print("✗ Missing page routing logic")

    print("\n✅ Navigation structure check complete!")

if __name__ == "__main__":
    check_navigation_structure()