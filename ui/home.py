"""Home page."""
from __future__ import annotations

import streamlit as st

from .components import (HERO_IMAGE, INSPECTION_IMAGE, button, cards, cta_band, e, img_uri, md, section)
from .facts import facts, pct

INSPECTION_CHECKS = [
    # (title, text, dot position on the photo as left%, top%)
    ("AI-generated photo check", "Two detectors trained on thousands of AI generators.", 38, 30),
    ("Metadata check", "Editing software, AI tags, capture date vs accident date.", 86, 14),
    ("Reuse check", "Same photo, file or face seen in an earlier claim?", 64, 78),
    ("Document check", "Cover-up boxes, font changes, totals that don't add up.", 18, 70),
    ("Identity check", "Photo ID (licence, Aadhaar, PAN…) vs selfie, with a 'manual check' band.", 70, 40),
]


def inspection_photo() -> str:
    dots = "".join(f'<div class="cg-dot" style="left:{x}%;top:{y}%">{k}<div class="cg-tip"><b>{e(t)}</b><br>{e(d)}</div></div>'
                   for k, (t, d, x, y) in enumerate(INSPECTION_CHECKS, 1))
    callout = lambda k: (f'<div class="cg-callout"><b><span class="cg-num">{k}</span>{e(INSPECTION_CHECKS[k-1][0])}</b>'  # noqa: E731
                         f'<p>{e(INSPECTION_CHECKS[k-1][1])}</p></div>')
    return (f'<div class="cg-inspect"><div class="cg-inspect-col">{callout(1)}{callout(2)}{callout(3)}</div>'
            f'<div class="cg-inspect-img"><img src="{img_uri(INSPECTION_IMAGE)}" alt="Damage photo">{dots}</div>'
            f'<div class="cg-inspect-col">{callout(4)}{callout(5)}</div></div>')


def render() -> None:
    f = facts()
    # 1. hero
    md(f'<div class="cg-hero"><div class="cg-hero-text"><h1>Catch fake claims before you pay them.</h1>'
       f'<p>ClaimGuard checks every claim photo, bill and ID for AI fakes and edits — and explains every decision.</p>'
       f'{button("Analyze a claim", "analyze", "white")}{button("Try a demo", "res-demos", "ghost")}</div>'
       f'<div class="cg-hero-img"><img src="{img_uri(HERO_IMAGE)}" alt="ClaimGuard screen"></div></div>')

    # 2. stats strip (live numbers from the evaluation)
    stats = [
        (pct(f["cardd_fp"]), f"false alarms on {f['cardd_n'] or ''} real car photos (CarDD)"),
        (pct(f["aivh_tp"]), "of AI images caught (public dataset)"),
        (f"{f['face_diff_match_n']} in {f['face_pairs']:,}" if f["face_pairs"] else "—",
         "wrong face matches between different people"),
        ("100%", "offline — no data leaves the machine"),
    ]
    md('<div style="height:14px"></div>')
    cols = st.columns(4)
    for col, (v, t) in zip(cols, stats):
        col.markdown(f'<div class="cg-stat"><b>{v}</b><span>{e(t)}</span></div>', unsafe_allow_html=True)

    # 3. vision
    section("One verdict. Every check. One screen.",
            "Generative AI makes fake damage photos and edited bills cheap to produce. ClaimGuard checks the "
            "evidence itself, not just the claimant's behaviour. Every flag comes with a reason an investigator "
            "can verify.")

    # 4. inspection photo
    md(inspection_photo())
    md('<p style="font-size:.8rem;color:#94A3B8;margin-top:8px">Hover the numbered dots. Example photo — '
       'in a real analysis, flags appear only where a check actually finds something.</p>')

    # 5. where it helps
    section("Where ClaimGuard helps", "Built first for motor claims, where fake photos and edited bills are most common.")
    cards([
        ("🚗", "Motor claims", "Fake or AI-made damage photos, photos reused from old claims, edited garage bills, "
         "and RC or licence details that don't match.", "Live"),
        ("🧾", "Bills & receipts", "Edited totals and cover-up boxes on PDF bills. Photographed receipts edited by "
         "hand are still hard to catch.", "Beta"),
        ("🏥", "Health claims", "Edited hospital bills and AI-made injury photos. Same engine, new rules file.",
         "Coming next"),
        ("🏠", "Property claims", "Exaggerated damage photos and images reused from the internet or old claims.",
         "Coming next"),
    ], cols=4)

    # 6. how it works
    section("How it works", "Four steps, about a second per photo on a normal laptop.")
    steps = [("1", "Upload", "Add every file from the claim: photos, bills, RC, licence, selfie."),
             ("2", "Check every file", "AI-photo detectors, metadata and document checks run on each file."),
             ("3", "Cross-check the claim", "Names, vehicle numbers, dates and faces are compared across files and past claims."),
             ("4", "Clear verdict", "LOW, MEDIUM or HIGH, with the exact reasons and the evidence behind them.")]
    for col, (n, t, d) in zip(st.columns(4), steps):
        col.markdown(f'<div class="cg-card cg-step"><div class="n">{n}</div><h4>{e(t)}</h4><p>{e(d)}</p></div>',
                     unsafe_allow_html=True)

    # 7. who it's for
    section("Who it's for")
    cards([("🧑‍💼", "Claim handlers", "Genuine claims are fast-tracked, so customers get paid sooner."),
           ("🕵️", "Fraud investigators", "See exactly why a claim was flagged, with the evidence marked."),
           ("🖥️", "Insurer IT teams", "Runs on your own servers, no GPU, one simple function to call.")], cols=3)

    # 8. FAQ
    section("Questions people ask")
    faq(f)

    # 9. CTA
    cta_band("Try it on your own claim", "Upload a few photos and a bill. You get a verdict with reasons in seconds.")


def faq(f: dict) -> None:
    sec = f"~{f['photo_s']:.1f} s" if f.get("photo_s") else "about a second"
    doc = f"~{f['doc_ms']} ms" if f.get("doc_ms") else "well under a second"
    qa = [
        ("Does ClaimGuard reject claims by itself?",
         "No. It gives a screening signal with reasons. A person always makes the final decision."),
        ("Does any claim data leave our machine?",
         "No. All models run locally. Nothing is sent to the internet or any outside service."),
        ("How fast is it?", f"{sec} per photo and {doc} per PDF document, on a normal CPU."),
        ("What can't it catch yet?",
         f"The very newest AI image tools are caught about {pct(f['frontier_tp'])} of the time. Small hand edits "
         f"on photographed receipts are caught only {pct(f['receipt_caught'])} of the time. A small AI patch inside "
         "a real photo is not reliably caught. Doubtful cases are marked INCONCLUSIVE and go to a person."),
        ("Does it need a GPU?", "No. It runs on an ordinary laptop or server CPU."),
        ("Can it work for health or property insurance?",
         "Yes. The engine stays the same; a new rules file describes the documents and checks for that type."),
    ]
    for q, a in qa:
        with st.expander(q):
            st.write(a)
