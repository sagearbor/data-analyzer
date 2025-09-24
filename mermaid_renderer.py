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

    # HTML template with Mermaid.js
    mermaid_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
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
            /* Custom styling for the diagram */
            .node rect {{
                rx: 5;
                ry: 5;
            }}
            .edgeLabel {{
                background-color: white;
                padding: 2px 4px;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="mermaid">
            {mermaid_code}
        </div>
        <script>
            mermaid.initialize({{
                startOnLoad: true,
                theme: 'default',
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
    </body>
    </html>
    """

    # Render the HTML
    components.html(mermaid_html, height=height, scrolling=True)