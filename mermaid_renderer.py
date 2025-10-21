"""
Mermaid diagram renderer for Streamlit
Uses components.html to render Mermaid diagrams
"""

import streamlit as st
import streamlit.components.v1 as components

def render_mermaid(mermaid_code: str, height: int = 600) -> None:
    """
    Render a Mermaid diagram in Streamlit using HTML/JavaScript

    Args:
        mermaid_code: The Mermaid diagram code
        height: Height of the diagram container in pixels
    """

    import base64
    import urllib.parse

    # Try to use mermaid.ink API first (more reliable with Streamlit's iframe restrictions)
    # Fallback to embedded mermaid.js if needed
    try:
        # Encode mermaid code for URL
        mermaid_encoded = base64.b64encode(mermaid_code.encode('utf-8')).decode('utf-8')
        mermaid_url = f"https://mermaid.ink/img/{mermaid_encoded}"

        # Use an image-based approach which doesn't require JavaScript in iframe
        image_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    margin: 0;
                    padding: 20px;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    background: white;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <img src="{mermaid_url}" alt="Data Flow Diagram" />
        </body>
        </html>
        """
        components.html(image_html, height=height, scrolling=True)

    except Exception as e:
        # Fallback to embedded mermaid.js approach
        mermaid_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <script type="module">
                import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                mermaid.initialize({{
                    startOnLoad: true,
                    theme: 'default',
                    securityLevel: 'loose',
                    themeVariables: {{
                        primaryColor: '#667eea',
                        primaryTextColor: '#fff',
                        primaryBorderColor: '#4c5ed9',
                        lineColor: '#5a67d8',
                        secondaryColor: '#f7fafc',
                        tertiaryColor: '#e2e8f0'
                    }},
                    flowchart: {{
                        htmlLabels: true,
                        curve: 'basis'
                    }}
                }});
            </script>
            <style>
                body {{
                    margin: 0;
                    padding: 20px;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
                    background: white;
                }}
                .mermaid {{
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: {height - 40}px;
                }}
            </style>
        </head>
        <body>
            <div class="mermaid">
                {mermaid_code}
            </div>
        </body>
        </html>
        """
        # Note: Streamlit uses srcdoc for components.html, which can cause sandbox warnings
        # These warnings are expected and don't affect functionality
        components.html(mermaid_html, height=height, scrolling=True)