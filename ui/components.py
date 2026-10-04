"""Small reusable UI pieces (pure presentation: they only display what the backend returns)."""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import streamlit as st
from PIL import Image

from core.config import ROOT
from core.schemas import ClaimReport, Reason, Tier

from .nav import link
from .theme import TIER_STYLE

# Placeholder images: replace these two files to change the pictures on the website.
ASSETS = ROOT / "assets"
HERO_IMAGE = ASSETS / "hero_placeholder.png"
INSPECTION_IMAGE = ASSETS / "inspection_placeholder.jpg"

e = html.escape


def md(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def img_uri(path: Path | str, max_side: int = 1400) -> str:
    """Inline an image as a data URI (keeps the app fully offline)."""
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        return ""
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def button(label: str, page: str, style: str = "grad", **extra: str) -> str:
    return f'<a class="cg-btn cg-btn-{style}" href="{link(page, **extra)}" target="_self">{e(label)}</a>'


def section(title: str, sub: str = "") -> None:
    md(f'<div class="cg-section-title">{e(title)}</div>' + (f'<div class="cg-section-sub">{e(sub)}</div>' if sub else ""))


def page_hero(title: str, sub: str) -> None:
    md(f'<div class="cg-page-hero"><h1>{e(title)}</h1><p>{e(sub)}</p></div>')


BADGES = {"Live": "cg-live", "Beta": "cg-beta", "Coming next": "cg-soon"}


def card(icon: str, title: str, text: str, badge: str | None = None) -> str:
    b = f'<span class="cg-badge {BADGES.get(badge, "cg-soon")}">{e(badge)}</span>' if badge else ""
    return f'<div class="cg-card"><div class="cg-icon">{icon}</div><h4>{e(title)}{b}</h4><p>{text}</p></div>'


def cards(items: list[tuple], cols: int | None = None) -> None:
    n = cols or len(items)
    for start in range(0, len(items), n):
        row = st.columns(n)
        for col, it in zip(row, items[start:start + n]):
            with col:
                md(card(*it))


def bullets(points: list[str]) -> None:
    md("<ul>" + "".join(f"<li>{p}</li>" for p in points) + "</ul>")


def cta_band(title: str, text: str, label: str = "Analyze a claim", page: str = "analyze") -> None:
    md(f'<div class="cg-cta-band"><h2>{e(title)}</h2><p>{e(text)}</p>{button(label, page, "white")}</div>')


def page_footer_buttons() -> None:
    md('<div style="margin-top:22px">' + button("Analyze a claim", "analyze") +
       button("Try a demo", "res-demos", "grad") + "</div>")


# ---------------------------------------------------------------- analyze-page pieces
TIER_TITLE = {
    Tier.HIGH: "HIGH risk — hold payment",
    Tier.MEDIUM: "MEDIUM risk — review before paying",
    Tier.LOW: "LOW risk — approve normally",
    Tier.INCONCLUSIVE: "INCONCLUSIVE — ask for better evidence",
}
NEXT_STEP = {
    Tier.HIGH: "Send this claim to the fraud team. Do not pay yet.",
    Tier.MEDIUM: "Check the flagged items below before approving.",
    Tier.LOW: "No strong warning signs. Process the claim normally.",
    Tier.INCONCLUSIVE: "The evidence is too weak or unclear. Ask the claimant for better photos or documents.",
}


def stepper(current: int, done_upto: int) -> None:
    names = ["1 · Claim details", "2 · Upload evidence", "3 · Run analysis", "4 · Verdict", "5 · Details"]
    parts = []
    for k, n in enumerate(names, 1):
        cls = "now" if k == current else "done" if k <= done_upto else ""
        parts.append(f'<div class="{cls}">{"✓ " if k <= done_upto and k != current else ""}{n}</div>')
    md('<div class="cg-stepper">' + "".join(parts) + "</div>")


def verdict_card(r: ClaimReport) -> None:
    colour, bg, icon = TIER_STYLE[r.tier]
    when = r.created_at.replace("T", " ")
    md(f'<div class="cg-verdict" style="background:{bg};border-color:{colour}">'
       f'<div class="ic">{icon}</div><div style="flex:1">'
       f'<h2 style="color:{colour}">{TIER_TITLE[r.tier]}</h2>'
       f'<div style="color:#0F1B3D">{e(r.summary)}</div>'
       f'<div class="meta">Claim {e(r.claim_id)} · {e(when)} · overall fraud likelihood {r.overall_risk:.0%}</div>'
       f'<div class="next"><b>What to do next:</b> {NEXT_STEP[r.tier]}</div></div></div>')


def score_tile(label: str, value: str, hint: str) -> str:
    return f'<div class="cg-tile" title="{e(hint)}"><div class="v">{value}</div><div class="l">{e(label)}</div><div class="h">{e(hint)}</div></div>'


def reason_card(r: Reason) -> None:
    if r.contribution > 0.05:
        colour, bg = "#DC2626", "#FEF2F2"
    elif r.contribution < -0.05:
        colour, bg = "#16A34A", "#F0FDF4"
    else:
        colour, bg = "#64748B", "#F8FAFC"
    where = f"<span class='cg-chip'>{e(r.filename)}</span>" if r.filename else "<span class='cg-chip'>whole claim</span>"
    impact = f"{r.contribution:+.2f}" if r.contribution else "note"
    md(f"<div class='cg-reason' style='border-color:{colour};background:{bg}'>"
       f"<b>{e(r.title)}</b>{where}<span class='cg-chip'>impact {impact}</span><br>{e(r.text)}<br>"
       f"<span class='code'>{e(r.code)}</span></div>")


def check_card(name: str, status: str, detail: str) -> str:
    """status: ok | flag | warn | na | note"""
    icon, cls = {"ok": ("✅", "ok"), "flag": ("⚠️", "flag"), "warn": ("🟡", "warn"),
                 "na": ("➖", "na"), "note": ("ℹ️", "na")}[status]
    label = {"ok": "passed", "flag": "flagged", "warn": "check manually", "na": "not checked", "note": "note"}[status]
    return f'<div class="cg-check {cls}"><b>{icon} {e(name)}</b> <span class="cg-chip">{label}</span><p>{e(detail)}</p></div>'
