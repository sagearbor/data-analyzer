"""
Diagram display for the About page - fully local, no outside services.

Earlier versions loaded the diagram from a third-party image service and, as
a fallback, an unpinned script from a public CDN (B-15 / ADP-86). Both are
gone: the browser now fetches nothing from outside the app.

The diagrams are pre-rendered to PNG files that are committed next to their
Mermaid sources, and shown with st.image(). If a PNG is missing, the Mermaid
source is shown as plain text instead (st.text, so it never mixes with the
API example code blocks that the About page tests count).

To regenerate a PNG after editing a .mmd source (needs the Mermaid CLI `mmdc`
and a local Chrome; nothing is uploaded anywhere):
    mmdc -i assets/data_flow_simple.mmd -o assets/data_flow_simple.png -b white -s 2
"""

from pathlib import Path
from typing import Optional, Union

import streamlit as st


def render_mermaid(mermaid_code: str, height: int = 600,
                   image_path: Optional[Union[str, Path]] = None) -> None:
    """Show a data-flow diagram without calling any external service.

    Args:
        mermaid_code: The Mermaid diagram source (always available as text).
        height: Unused; kept so existing callers keep working.
        image_path: Pre-rendered PNG of the diagram. When it exists it is shown
            as an image, with the source in a collapsed expander.
    """
    if image_path and Path(image_path).is_file():
        st.image(str(image_path), caption="Data flow diagram")
        with st.expander("Diagram source (Mermaid text)"):
            st.text(mermaid_code)
        return
    st.caption("Diagram source (Mermaid text). The rendered picture is not available in this build.")
    st.text(mermaid_code)
