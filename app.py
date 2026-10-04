"""ClaimGuard web app.  Run:  streamlit run app.py

This file only sets up the page and routes ?page=... to the right screen. All analysis is done by
core.pipeline.analyze_claim(); the ui/ package only collects input and displays the ClaimReport.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="ClaimGuard — fake claim detection", page_icon="🛡️", layout="wide",
                   initial_sidebar_state="expanded")

from ui import analyze, home, nav, pages_about, pages_resources, pages_solutions, pages_tech, theme  # noqa: E402

PAGES = {"home": home.render, "analyze": analyze.render,
         **pages_tech.PAGES, **pages_solutions.PAGES, **pages_resources.PAGES, **pages_about.PAGES}
# old links keep working
ALIASES = {"history": "analyze", "how-scoring": "tech-scoring", "method": "tech-scoring"}

theme.inject()
nav.render()
page = st.query_params.get("page", "home")
page = ALIASES.get(page, page)
PAGES.get(page, home.render)()
