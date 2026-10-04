"""Top bar with hover dropdowns. Every link is a query param (?page=...), read in app.py."""
from __future__ import annotations

import html

import streamlit as st

# page id -> (menu, title, one-line hint). Order here is the order in the dropdowns.
MENUS: dict[str, list[tuple[str, str, str]]] = {
    "Technology": [
        ("tech-how", "How it works", "The 6-step check, from upload to verdict"),
        ("tech-ai", "AI photo & deepfake selfie detection", "Is the photo real or AI-made?"),
        ("tech-docs", "Document tampering checks", "Edited bills, cover-ups, wrong totals"),
        ("tech-identity", "Identity check", "Licence photo vs selfie"),
        ("tech-scoring", "Explainable risk scoring", "How every point of the score is earned"),
    ],
    "Solutions": [
        ("sol-motor", "Motor claims", "Live"),
        ("sol-bills", "Bills & receipts", "Beta"),
        ("sol-health", "Health claims", "Coming next"),
        ("sol-property", "Property claims", "Coming next"),
    ],
    "Resources": [
        ("res-accuracy", "Accuracy report", "Our measured results, per dataset"),
        ("res-models", "Models & datasets", "What we use and where it comes from"),
        ("res-demos", "Try demo claims", "Six ready-made example claims"),
        ("res-faq", "FAQ", "Short answers to common questions"),
        ("res-github", "GitHub & docs", "Code and how to run it"),
    ],
    "About": [
        ("about-vision", "Our vision", "Why we built ClaimGuard"),
        ("about-problem", "The problem we solve", "The challenge in plain words"),
        ("about-team", "Team", "Who built what"),
        ("about-contact", "Contact", "Get in touch"),
    ],
}
PAGE_TITLES = {pid: title for items in MENUS.values() for pid, title, _ in items}
PAGE_TITLES.update({"home": "Home", "analyze": "Analyze a claim"})


def link(page: str, **extra: str) -> str:
    q = "&".join([f"page={page}"] + [f"{k}={v}" for k, v in extra.items()])
    return f"?{q}"


def render() -> None:
    menus = []
    for name, items in MENUS.items():
        links = "".join(f'<a href="{link(pid)}" target="_self">{html.escape(t)}<small>{html.escape(hint)}</small></a>'
                        for pid, t, hint in items)
        menus.append(f'<div class="cg-menu"><span>{name}</span><div class="cg-drop">{links}</div></div>')
    st.markdown(
        f'<div class="cg-topbar"><a class="cg-logo" href="{link("home")}" target="_self">🛡️ ClaimGuard</a>'
        f'<div class="cg-menus">{"".join(menus)}</div>'
        f'<a class="cg-cta" href="{link("analyze")}" target="_self">Analyze a claim</a></div>',
        unsafe_allow_html=True)
