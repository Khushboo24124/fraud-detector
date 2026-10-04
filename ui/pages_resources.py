"""Resources pages. The accuracy report is the backend's own reports/eval_report.md, shown as-is."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from core.config import ROOT, settings

from .components import bullets, button, md, page_footer_buttons, page_hero, section
from .home import faq as home_faq
from .facts import facts

REPO_URL = "https://github.com/Khushboo24124/fraud-detector"
SAMPLES = ROOT / "data" / "samples"

DEMO_TEXT = {
    "1_genuine_claim": ("Genuine claim", "Two real damage photos, a clean bill and RC.", "LOW"),
    "2_fraud_claim": ("Fraud claim", "AI-made photo, edited bill total, RC of another car, bill dated before the accident.", "HIGH"),
    "3_reused_photo": ("Reused photo", "Looks clean, but the photo is a cropped copy from claim 1. Run claim 1 first.", "HIGH"),
    "4_identity_match": ("Identity match", "Licence and selfie of the same person.", "LOW"),
    "5_identity_mismatch": ("Identity mismatch", "Same licence, but the selfie is someone else.", "HIGH"),
    "6_deepfake_selfie": ("Deepfake selfie", "One AI-made selfie, uploaded on its own.", "HIGH"),
}


def demo_folders() -> list:
    return sorted(p for p in SAMPLES.iterdir() if p.is_dir()) if SAMPLES.exists() else []


def accuracy() -> None:
    page_hero("Accuracy report", "What we measured, on which data — including what doesn't work yet.")
    rep = ROOT / "reports" / "eval_report.md"
    if rep.exists():
        st.markdown(rep.read_text(encoding="utf-8"))
    else:
        st.info("Run `python scripts/eval.py` to create the report.")
    page_footer_buttons()


def models() -> None:
    m = settings()["models"]
    page_hero("Models & datasets", "Free, open-source models running offline, plus small models we trained.")
    section("Models")
    st.dataframe(pd.DataFrame([
        {"What it does": "AI-image detection", "Model": "CommunityForensics ViT (CVPR 2025)",
         "Source": f"Hugging Face · {m['community_forensics']['repo']}", "Ours?": "pre-trained"},
        {"What it does": "AI-image detection (damage photos)", "Model": "SigLIP AI-vs-Human",
         "Source": f"Hugging Face · {m['siglip_ai_vs_human']['repo']}", "Ours?": "pre-trained"},
        {"What it does": "Combine detector scores", "Model": "Logistic-regression calibrators (photo, selfie)",
         "Source": "data/models/*_calibrator.joblib", "Ours?": "trained by us"},
        {"What it does": "Find faces", "Model": "OpenCV YuNet", "Source": "OpenCV Model Zoo", "Ours?": "pre-trained"},
        {"What it does": "Match faces", "Model": "OpenCV SFace", "Source": "OpenCV Model Zoo", "Ours?": "pre-trained"},
        {"What it does": "Read text (OCR)", "Model": "RapidOCR", "Source": "pip package", "Ours?": "pre-trained"},
        {"What it does": "Unusual invoice check", "Model": "Isolation Forest", "Source": "data/models/invoice_iforest.joblib",
         "Ours?": "trained by us"},
    ]), hide_index=True, width="stretch")
    st.caption("Model versions are pinned in config/settings.yaml, so scores never change silently. "
               "Run `python scripts/download_models.py` once; after that everything works offline.")
    section("Datasets")
    st.dataframe(pd.DataFrame([
        {"Dataset": "DrBimmer car damage", "Contains": "Real damaged-car photos", "Used for": "Training + testing (real photos)"},
        {"Dataset": "CarDD", "Contains": "Real damaged-car photos", "Used for": "Training + false-alarm test"},
        {"Dataset": "frontier-synthetic-images-2026", "Contains": "Images from ~20 newest AI tools", "Used for": "Training + testing (AI photos, AI faces)"},
        {"Dataset": "AI vs Human-Generated Images", "Contains": "Real stock photos + AI images", "Used for": "Training + testing"},
        {"Dataset": "face_recognition examples", "Contains": "17 real face photos", "Used for": "Real selfies, demo faces"},
        {"Dataset": "Selfies & ID Images", "Contains": "ID photos + selfies of 29 people", "Used for": "Face-match test, real selfies"},
        {"Dataset": "Find it again!", "Contains": "Real receipts, some forged by people", "Used for": "Testing only"},
        {"Dataset": "CASIA v2", "Contains": "Real and spliced photos with masks", "Used for": "Testing only (heatmap)"},
        {"Dataset": "Our synthetic invoices", "Contains": "Garage bills, genuine + 5 forgery types", "Used for": "Anomaly model training + testing"},
    ]), hide_index=True, width="stretch")
    page_footer_buttons()


def demos() -> None:
    page_hero("Try demo claims", "Six ready-made claims. Click one to run it on the Analyze page.")
    folders = demo_folders()
    if not folders:
        st.info("No demo claims found. Run `python scripts/make_demo_claims.py`.")
        return
    cols = st.columns(3)
    for k, folder in enumerate(folders):
        title, text, expect = DEMO_TEXT.get(folder.name, (folder.name.replace("_", " "), "", "—"))
        files = ", ".join(p.name for p in sorted(folder.iterdir()) if p.name != "info.json")
        with cols[k % 3]:
            md(f'<div class="cg-card"><h4>{k + 1}. {title}</h4><p>{text}</p>'
               f'<p style="font-size:.8rem;margin-top:8px"><b>Files:</b> {files}<br><b>Expected:</b> {expect}</p>'
               f'{button("Run this demo", "analyze", "grad", demo=folder.name)}</div>')
            st.write("")
    st.caption("Tip: before a presentation, clear the history on the Analyze page, then run demo 1 before demo 3.")


def faq() -> None:
    page_hero("FAQ", "Short answers to common questions.")
    home_faq(facts())
    page_footer_buttons()


def github() -> None:
    page_hero("GitHub & docs", "The full code, and how to run it on your own machine.")
    md(f'<p><a class="cg-btn cg-btn-grad" href="{REPO_URL}" target="_blank">Open the GitHub repository</a></p>')
    section("Run it locally (Windows)")
    st.code("py -3.11 -m venv .venv\n.venv\\Scripts\\activate\n"
            "pip install torch --index-url https://download.pytorch.org/whl/cpu\npip install -r requirements.txt\n"
            "python scripts/download_models.py\nstreamlit run app.py", language="bash")
    section("Useful files")
    bullets(["<b>README.md</b>: what it does, results, limits.", "<b>DEMO.md</b>: the 10-minute demo script.",
             "<b>reports/eval_report.md</b>: full accuracy report.",
             "<b>config/settings.yaml</b>: thresholds and weights. <b>config/reason_codes.yaml</b>: every reason."])
    page_footer_buttons()


PAGES = {"res-accuracy": accuracy, "res-models": models, "res-demos": demos, "res-faq": faq, "res-github": github}
