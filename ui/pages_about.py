"""About pages."""
from __future__ import annotations

import streamlit as st

from .components import bullets, cards, e, md, page_footer_buttons, page_hero, section
from .pages_resources import REPO_URL

TEAM = [
    ("Shubham Raj", "Model / AI", [
        "Chose and integrated the AI-image detectors (CommunityForensics, SigLIP).",
        "Trained the calibrators on public datasets.",
        "Built the evaluation: cross-validation, unseen-generator test, per-dataset results."]),
    ("Khushboo", "Backend", [
        "Built the analysis pipeline and plug-in checks (photos, documents, identity, reuse).",
        "Built the explainable risk engine: log-odds scoring, reason codes, rules, INCONCLUSIVE logic.",
        "Made the rules config-driven."]),
    ("Aditi", "Frontend", [
        "Designed and built the Streamlit app: home page and menus.",
        "Built the step-by-step Analyze flow, verdict and evidence views.",
        "Added the case-report downloads."]),
    ("Vishesh", "Database", [
        "Designed the SQLite store: claim history and evidence storage.",
        "File, photo and face fingerprints for reuse detection.",
        "Privacy by design: faces stored as numbers, not photos."]),
]


def vision() -> None:
    page_hero("Our vision", "Trust the evidence again — without slowing honest customers down.")
    bullets(["AI tools now make fake damage photos and edited bills in minutes.",
             "Most fraud tools look at the claimant's behaviour, not at whether the evidence itself is real.",
             "We want every claim's photos, bills and IDs checked automatically before payment.",
             "Honest claims should move faster. Doubtful ones should reach a person with clear reasons.",
             "Every decision must be explainable, so investigators and customers can trust it."])
    page_footer_buttons()


def problem() -> None:
    page_hero("The problem we solve", "From the Adrosonic Build challenge, in plain words.")
    cards([("📷", "Fake photos", "Insurers get thousands of photos a day. AI can now create or change damage photos."),
           ("🧾", "Edited documents", "Bills and certificates can be edited to raise amounts or change details."),
           ("🪪", "Fake identities", "Selfies and ID photos can be swapped or generated."),
           ("⏱️", "Too much to check by hand", "Teams can't manually inspect every file of every claim.")], cols=4)
    section("What the challenge asked for — and what we built")
    bullets(["Detect AI-made or edited images, with a confidence score → <b>done</b>.",
             "Find tampering in documents, with reasons → <b>done</b>.",
             "Image score, document score and overall fraud likelihood, explained → <b>done</b>.",
             "A simple web app to upload files and see results → <b>done</b>.",
             "Bonus: compare faces across ID and selfie → <b>done</b> (face morphing not yet)."])
    page_footer_buttons()


def team() -> None:
    page_hero("Team", "Four people, four parts of one system.")
    cols = st.columns(4)
    for col, (name, role, points) in zip(cols, TEAM):
        initials = "".join(w[0] for w in name.split()[:2]).upper()
        with col:
            md(f'<div class="cg-card"><div class="cg-avatar">{initials}</div><h4 style="margin-top:12px">{e(name)}</h4>'
               f'<p style="font-weight:700;color:#2F5BEA;margin-bottom:8px">{e(role)}</p>'
               + "".join(f'<p style="margin-bottom:6px">• {e(p)}</p>' for p in points) + "</div>")


def contact() -> None:
    page_hero("Contact", "Questions, feedback or a pilot on your own claims.")
    bullets([f'Code and issues: <a href="{REPO_URL}" target="_blank">{REPO_URL}</a>',
             "Built for the Adrosonic Build hackathon."])
    page_footer_buttons()


PAGES = {"about-vision": vision, "about-problem": problem, "about-team": team, "about-contact": contact}
