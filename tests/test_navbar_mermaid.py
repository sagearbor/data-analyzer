#!/usr/bin/env python3
"""
Test the navbar and Mermaid diagram rendering
"""

import streamlit as st
import streamlit.components.v1 as components
from mermaid_renderer import render_mermaid

def test_mermaid():
    """Test Mermaid diagram rendering"""

    st.set_page_config(page_title="Mermaid Test", layout="wide")

    st.title("🧪 Testing Navbar and Mermaid Rendering")

    # Test navbar with option menu
    try:
        from streamlit_option_menu import option_menu

        selected = option_menu(
            menu_title=None,
            options=["Home", "About", "Test"],
            icons=["house", "info-circle", "gear"],
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#667eea", "font-size": "20px"},
                "nav-link": {
                    "font-size": "16px",
                    "text-align": "center",
                    "margin": "0px",
                },
                "nav-link-selected": {"background-color": "#667eea", "color": "white"},
            }
        )

        st.success(f"✅ Option menu working! Selected: {selected}")

    except ImportError:
        st.error("❌ streamlit-option-menu not installed")

    st.markdown("---")

    # Test Mermaid rendering
    st.subheader("Testing Mermaid Diagram Rendering")

    # Simple test diagram
    test_diagram = """
    graph LR
        A[User] --> B[Upload CSV]
        B --> C[Parse Dictionary]
        C --> D{Uses AI?}
        D -->|Yes| E[Azure OpenAI]
        D -->|No| F[Direct Parse]
        E --> G[Generate Code]
        F --> G
        G --> H[Validate Data]
        H --> I[Show Results]

        style E fill:#e1f5fe,stroke:#01579b,stroke-width:3px
        style A fill:#fff3e0,stroke:#e65100
        style I fill:#e8f5e9,stroke:#1b5e20
    """

    try:
        # Try our custom renderer
        render_mermaid(test_diagram, height=400)
        st.success("✅ Mermaid diagram rendered successfully!")

    except Exception as e:
        st.error(f"❌ Error rendering Mermaid: {str(e)}")

        # Fallback to showing code
        st.code(test_diagram, language='mermaid')

    # Test loading the actual diagram file
    st.subheader("Testing Actual Diagram File")

    try:
        with open('../assets/data_flow_diagram.mmd', 'r') as f:
            actual_diagram = f.read()

        st.info(f"📄 Loaded diagram file ({len(actual_diagram)} characters)")

        # Render it
        render_mermaid(actual_diagram, height=700)

    except FileNotFoundError:
        st.error("❌ Diagram file not found at ../assets/data_flow_diagram.mmd")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_mermaid()