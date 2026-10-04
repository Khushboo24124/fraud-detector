"""Technology pages. Thresholds, weights and reason codes are read live from config/, so they always match the backend."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.config import domain_rules, reason_catalog, settings

from .components import bullets, cards, md, page_footer_buttons, page_hero, section
from .facts import facts, pct

GROUP_LABELS = {"ai_detector": "AI-image detectors", "gen_metadata": "Generator metadata",
                "local_anomaly": "Localized region", "compression": "Compression (ELA)",
                "exif_software": "Editing software", "exif_timeline": "Capture timeline",
                "doc_font": "Font consistency", "doc_structure": "PDF structure", "doc_editor": "Editing tool",
                "doc_overlay": "Cover-up / overlay", "doc_math": "Arithmetic", "doc_anomaly": "Anomaly model",
                "doc_image_edit": "Document image edit", "cross_document": "Cross-document",
                "identity": "Face match", "reuse": "Cross-claim reuse"}


def how() -> None:
    page_hero("How it works", "Six steps from upload to a verdict you can explain.")
    cards([
        ("📥", "1. Intake and quality check", "We check each file is really a photo or PDF, store a fingerprint of it, "
         "and rate its quality. Blurry or tiny files count less."),
        ("🗂️", "2. Identify each file", "Damage photo, bill, RC, licence or selfie? Found from the file name, the text "
         "on it and whether it shows a face. You can correct it."),
        ("🔬", "3. Check each file", "AI-photo detectors, metadata checks and document checks run on every file."),
        ("🔗", "4. Cross-check the claim", "Names, vehicle numbers and dates are compared across files. Faces are "
         "matched. Photos and faces are compared with past claims."),
        ("⚖️", "5. Risk scoring", "Every finding becomes points on one scale. The points add up to the final score, "
         "so the chart is the real calculation."),
        ("📋", "6. Verdict and report", "LOW, MEDIUM, HIGH or INCONCLUSIVE, with reasons, marked evidence and a "
         "downloadable case report."),
    ], cols=3)
    section("Built to be trusted")
    bullets(["<b>Explainable:</b> every point of the score comes from a named reason.",
             "<b>Safe with bad files:</b> poor-quality evidence moves the score less.",
             "<b>Never guesses:</b> weak or mixed evidence gives INCONCLUSIVE, not a made-up answer.",
             "<b>Keeps working:</b> if one check fails, the claim still completes and the gap is shown.",
             "<b>Private:</b> everything runs on your machine."])
    page_footer_buttons()


def ai() -> None:
    f, th = facts(), settings()["thresholds"]
    page_hero("AI photo & deepfake selfie detection", "Is this photo real, or made by an AI tool?")
    section("How we check")
    bullets([
        "<b>Two pre-trained detectors</b>: CommunityForensics (trained on thousands of AI generators) and a SigLIP "
        "AI-vs-human model.",
        "<b>A calibrator we trained</b> turns their raw scores into one fair probability.",
        f"<b>One clear line:</b> {th['ai_flag']:.0%} or more = flagged. Between {th['ai_borderline']:.0%} and "
        f"{th['ai_flag']:.0%} = INCONCLUSIVE (a person checks it). Below = looks real.",
        "<b>Selfies are checked too</b>, using only CommunityForensics, because SigLIP called many real faces AI.",
        "<b>Metadata:</b> AI-tool tags (C2PA, Stable Diffusion, ComfyUI), editing software and capture date.",
    ])
    section("Measured results")
    cards([
        ("🚗", pct(f["cardd_fp"]) + " false alarms", f"on {f['cardd_n']} real car-damage photos (CarDD)."),
        ("🤖", pct(f["aivh_tp"]) + " caught", "of AI images in a public AI-vs-Human dataset."),
        ("🆕", pct(f["frontier_tp"]) + " caught", "of images from the very newest AI tools. These are hard for every "
         "detector today."),
        ("🤳", pct((f["selfie_tp"] or 0) + (f["selfie_inc"] or 0)) + " reach a person",
         f"of AI selfies are flagged or marked inconclusive; {pct(f['selfie_fp'])} of real selfies are flagged."),
    ], cols=4)
    section("Limits")
    bullets(["A small AI patch inside a real photo is not reliably caught yet.",
             "The heatmap (compression view) is shown to investigators but never counted in the score."])
    page_footer_buttons()


def docs() -> None:
    markers = ", ".join(domain_rules().get("domain_markers", []))
    f = facts()
    page_hero("Document tampering checks", "Was the bill, RC or claim form edited after it was made?")
    cards([
        ("📄", "PDF structure", "Cover-up boxes with new text on top, hidden text under visible text, many saved "
         "versions, edit dates and editing tools named in the file."),
        ("🔤", "Fonts and text size", "A number in a different font or size from the rest of its line is a classic "
         "sign of an edit. For scans we measure character height."),
        ("➕", "Arithmetic", "Items → subtotal → tax → total must add up. A changed total usually breaks the sum."),
        ("📈", "Anomaly model", "Learns what normal garage bills look like and flags unusual ones."),
        ("🔗", "Cross-document", "Vehicle number and owner name must match across RC, bill and claim. The bill "
         "cannot be dated before the accident."),
    ], cols=3)
    section("Honest notes")
    bullets([f"Arithmetic and anomaly checks run only on Indian GST bills (markers: {markers}). On other "
             "documents they are skipped and a note says so.",
             f"Our test invoices: {f['doc_caught']} forged caught, {f['doc_false']} genuine flagged.",
             f"Real receipts edited by hand in Paint/GIMP: only {pct(f['receipt_caught'])} caught. This is our "
             "biggest open gap."])
    page_footer_buttons()


def identity() -> None:
    th, f = settings()["thresholds"], facts()
    page_hero("Identity check", "Is the person in the selfie the same person as on the photo ID?")
    section("How it works")
    bullets(["Works with any photo ID: driving licence, Aadhaar, PAN, voter ID or passport.",
             "We find the face on the ID and in the selfie, then compare them.",
             f"Similarity {th['face_match']:.2f} or more = match. Below {th['face_mismatch']:.2f} = different person "
             "(HIGH risk). In between = 'manual check needed'.",
             "The same face showing up in an earlier claim under another name is also flagged.",
             "Faces are stored as numbers only, never as photos.",
             "<b>Tip:</b> when uploading, set the ID to its type (or 'ID card') and the photo to 'Claimant selfie'. "
             "If no type is set, any document photo with a face is used as the ID."])
    section("Measured on real ID + selfie pairs")
    cards([("✅", pct(f["face_same_match"]) + " matched", f"Same person: {pct(f['face_same_manual'])} more went to a "
            "manual check."),
           ("🚫", f"{f['face_diff_match_n']} in {f['face_pairs']:,}" if f["face_pairs"] else "—",
            "pairs of different people were wrongly matched.")], cols=2)
    section("Limits")
    bullets(["A fully fake ID card made in one go has no edit marks, so document checks cannot see it. "
             "The selfie match catches it if the face is different.",
             "In production, the licence number would also be verified with the government database "
             "(e.g. Parivahan / DigiLocker)."])
    page_footer_buttons()


def scoring() -> None:
    cfg = settings()["risk"]
    page_hero("Explainable risk scoring", "Every point of the score comes from a named reason you can check.")
    section("How the score is built")
    bullets([
        "Each check gives a finding with a probability and a reliability (file quality).",
        "Each finding becomes points (log-odds) × its weight × reliability. Points add up, so the chart is the maths.",
        "Related findings share a group, and a group counts once. A real photo cannot cancel a fake one.",
        f"Tiers: LOW below {cfg['tiers']['medium']:.0%}, MEDIUM from {cfg['tiers']['medium']:.0%}, "
        f"HIGH from {cfg['tiers']['high']:.0%}. Some findings set a minimum tier by rule.",
        "INCONCLUSIVE when the evidence is borderline or all files are poor quality.",
    ])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Weight of each check")
        st.dataframe(pd.DataFrame([{"Check": GROUP_LABELS.get(g, g), "Weight": w}
                                   for g, w in cfg["group_weights"].items()]), hide_index=True, width="stretch")
    with c2:
        st.subheader("Rules that set a minimum tier")
        cat = reason_catalog()
        st.dataframe(pd.DataFrame([{"Finding": cat.get(c, {}).get("title", c), "Code": c, "Minimum tier": t}
                                   for c, t in cfg["overrides"].items()]), hide_index=True, width="stretch")
    st.subheader("Every reason code")
    st.dataframe(pd.DataFrame([{"Code": c, "Title": m["title"], "Counts in score": "yes" if m.get("group") else "note only",
                                "Group": GROUP_LABELS.get(m.get("group"), "—")}
                               for c, m in reason_catalog().items()]), hide_index=True, width="stretch")
    page_footer_buttons()


PAGES = {"tech-how": how, "tech-ai": ai, "tech-docs": docs, "tech-identity": identity, "tech-scoring": scoring}
