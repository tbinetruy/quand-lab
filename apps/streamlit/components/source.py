from __future__ import annotations

from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def show_source_file(relative_path: str) -> None:
    """Render a source file from the repository."""

    source_path = PROJECT_ROOT / relative_path
    st.code(source_path.read_text(encoding="utf-8"), language="python")

