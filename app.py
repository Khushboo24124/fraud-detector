"""ClaimGuard investigator dashboard.  Run:  streamlit run app.py"""
from __future__ import annotations

import io
import json
import math
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

from core import risk, store
from core.config import reason_catalog, settings
from core.pipeline import analyze_claim, claim_analyzers, item_analyzers
from core.report import build_html
from core.schemas import ClaimInfo, ClaimReport, Kind, Role, Tier

st.set_page_config(page_title="ClaimGuard", page_icon="🛡️", layout="wide")
SAMPLES = Path(__file__).parent / "data" / "samples"

TIER_STYLE = {
    Tier.LOW: ("#15803d", "#f0fdf4", "✅"), Tier.MEDIUM: ("#b45309", "#fffbeb", "⚠️"),
    Tier.HIGH: ("#b91c1c", "#fef2f2", "🚨"), Tier.INCONCLUSIVE: ("#475569", "#f1f5f9", "❔"),
}
ROLE_LABELS = {None: "Auto-detect", Role.DAMAGE_PHOTO: "Damage photo", Role.INVOICE: "Repair invoice",
               Role.RC: "Registration certificate (RC)", Role.LICENCE: "Driving licence",
               Role.SELFIE: "Claimant selfie", Role.CLAIM_FORM: "Claim form", Role.OTHER_DOC: "Other document"}
GROUP_LABELS = {"ai_detector": "AI-image detectors", "gen_metadata": "Generator metadata",
                "local_anomaly": "Localized region", "compression": "Compression (ELA)",
                "exif_software": "Editing software", "exif_timeline": "Capture timeline",
                "doc_font": "Font consistency", "doc_structure": "PDF structure", "doc_editor": "Editing tool", "doc_overlay": "Cover-up / overlay",
                "doc_math": "Arithmetic", "doc_anomaly": "Anomaly model", "doc_image_edit": "Document image edit",
                "cross_document": "Cross-document", "identity": "Face match", "reuse": "Cross-claim reuse"}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.verdict {border-radius: 10px; padding: 18px 22px; margin-bottom: 8px;}
.verdict h2 {margin: 0 0 6px 0; font-size: 1.7rem;}
.reason {border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; border-left: 6px solid;}
.reason .code {font-family: monospace; font-size: .78rem; opacity: .75;}
.chip {display:inline-block; font-size:.75rem; padding:1px 8px; border-radius:10px; background:#e2e8f0; color:#0f172a; margin-left:6px;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading detection models (first run only)…")
def warm_up() -> bool:
    _ = (item_analyzers(), claim_analyzers())  # assigned so Streamlit "magic" doesn't render it
    return True


# ------------------------------------------------------------------ charts
def gauge(value: float, tier: Tier) -> go.Figure:
    colour = TIER_STYLE[tier][0]
    t = settings()["risk"]["tiers"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(value * 100), number={"suffix": "%", "font": {"size": 40}},
        title={"text": "Overall fraud likelihood"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": colour, "thickness": 0.3},
               "steps": [{"range": [0, t["medium"] * 100], "color": "#dcfce7"},
                         {"range": [t["medium"] * 100, t["high"] * 100], "color": "#fef3c7"},
                         {"range": [t["high"] * 100, 100], "color": "#fee2e2"}]}))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def waterfall(report: ClaimReport) -> go.Figure:
    """How the score was built: base rate + each evidence group's log-odds = final likelihood."""
    prior = settings()["risk"]["prior_logit"]
    pooled = sorted(risk.pool_groups(report.signals).items(), key=lambda kv: -abs(kv[1]))
    names = ["Base rate"] + [GROUP_LABELS.get(g, g) for g, _ in pooled] + ["Final score"]
    vals = [prior] + [v for _, v in pooled] + [0]
    measures = ["absolute"] + ["relative"] * len(pooled) + ["total"]
    final = prior + sum(v for _, v in pooled)
    text = [f"{1 / (1 + math.exp(-prior)):.0%}"] + [f"{v:+.2f}" for _, v in pooled] + \
           [f"{1 / (1 + math.exp(-final)):.0%}"]
    fig = go.Figure(go.Waterfall(x=names, y=vals, measure=measures, text=text, textposition="outside",
                                 increasing={"marker": {"color": "#dc2626"}},
                                 decreasing={"marker": {"color": "#16a34a"}},
                                 totals={"marker": {"color": TIER_STYLE[report.tier][0]}}))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      yaxis_title="log-odds of fraud", title="Score breakdown (red = towards fraud, green = towards genuine)")
    return fig


# ------------------------------------------------------------------ evidence rendering
def draw_boxes(img: Image.Image, boxes: list[list[float]], scale: float = 1.0) -> Image.Image:
    img = img.copy()
    d = ImageDraw.Draw(img)
    for b in boxes:
        x0, y0, x1, y1 = (v * scale for v in b)
        d.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline=(220, 38, 38), width=max(3, img.width // 300))
    return img


def document_preview(item, report: ClaimReport) -> list[Image.Image]:
    sigs = [s for s in report.signals if s.item_id == item.id and "box" in s.evidence]
    if item.kind is Kind.IMAGE:
        return [draw_boxes(Image.open(item.path).convert("RGB"), [s.evidence["box"] for s in sigs])]
    import pymupdf
    dpi, pages = 110, []
    with pymupdf.open(item.path) as doc:
        for pno, page in enumerate(list(doc)[:3]):
            img = Image.open(io.BytesIO(page.get_pixmap(dpi=dpi).tobytes("png"))).convert("RGB")
            boxes = [s.evidence["box"] for s in sigs if s.evidence.get("page", 1) == pno + 1]
            pages.append(draw_boxes(img, boxes, dpi / 72))
    return pages


def reason_card(r) -> None:
    if r.contribution > 0.05:
        colour, bg = "#dc2626", "#fef2f2"
    elif r.contribution < -0.05:
        colour, bg = "#16a34a", "#f0fdf4"
    else:
        colour, bg = "#64748b", "#f8fafc"
    where = f"<span class='chip'>{r.filename}</span>" if r.filename else "<span class='chip'>whole claim</span>"
    impact = f"{r.contribution:+.2f}" if r.contribution else "info"
    st.markdown(f"<div class='reason' style='border-color:{colour};background:{bg};color:#0f172a'>"
                f"<b>{r.title}</b>{where}<span class='chip'>impact {impact}</span><br>{r.text}<br>"
                f"<span class='code'>{r.code}</span></div>", unsafe_allow_html=True)


def render_report(report: ClaimReport) -> None:
    colour, bg, icon = TIER_STYLE[report.tier]
    st.markdown(f"<div class='verdict' style='background:{bg};border-left:8px solid {colour};color:#0f172a'>"
                f"<h2 style='color:{colour}'>{icon} {report.tier.value} RISK · claim {report.claim_id}</h2>"
                f"{report.summary}</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.6])
    with c1:
        st.plotly_chart(gauge(report.overall_risk, report.tier), use_container_width=True)
        k = st.columns(3)
        for col, (name, label) in zip(k, [("image", "Image authenticity"), ("document", "Document authenticity"),
                                          ("identity", "Identity & reuse")]):
            d = report.domains[name]
            col.metric(label, f"{d.authenticity:.0%}" if d.evaluated else "—",
                       help="Not evaluated: no evidence of this type" if not d.evaluated else
                       f"Fraud risk from {name} evidence alone: {d.risk:.0%}")
    with c2:
        st.plotly_chart(waterfall(report), use_container_width=True)

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Why this score")
        scored = [r for r in report.reasons if r.contribution != 0]
        info = [r for r in report.reasons if r.contribution == 0]
        for r in scored:
            reason_card(r)
        if info:
            with st.expander(f"Notes ({len(info)})"):
                for r in info:
                    reason_card(r)
    with right:
        st.subheader("Evidence")
        tabs = st.tabs([f"{it.filename[:22]}" for it in report.items] or ["—"])
        for tab, it in zip(tabs, report.items):
            with tab:
                st.caption(f"{ROLE_LABELS.get(it.role, it.role.value)} · reliability {it.reliability:.0%}"
                           + (f" · {', '.join(it.quality.get('issues', []))}" if it.quality.get("issues") else ""))
                if it.role.is_document or it.kind is Kind.PDF:
                    for page in document_preview(it, report):
                        st.image(page, use_container_width=True)
                    if "heatmap" in it.artifacts:
                        with st.expander("Compression view (investigator aid, not scored)"):
                            st.image(it.artifacts["heatmap"], use_container_width=True)
                    if it.fields:
                        f = {k: v for k, v in it.fields.items() if k != "items" and v not in (None, [], "")}
                        st.json(f, expanded=False)
                        if it.fields.get("items"):
                            st.dataframe(pd.DataFrame(it.fields["items"]), hide_index=True, use_container_width=True)
                else:
                    st.image(it.path, caption="Submitted", use_container_width=True)
                    if "heatmap" in it.artifacts:
                        with st.expander("Compression & noise view (investigator aid, not scored)"):
                            st.caption("Highlights areas whose JPEG compression or sensor noise differs from the rest "
                                       "of the photo. It can point to crude local edits, but lights, dark sky, texture "
                                       "and blur also glow. It does **not** show where AI was detected: the AI verdict "
                                       "comes from the whole-image detectors and this view never changes the score.")
                            st.image(it.artifacts["heatmap"], use_container_width=True)
                    ai = next((s for s in report.signals if s.item_id == it.id and s.code.startswith("IMG-AI")), None)
                    if ai:
                        st.dataframe(pd.DataFrame([{"model": m, "P(AI-generated)": f"{p:.1%}"}
                                                   for m, p in ai.evidence["per_model"].items()]),
                                     hide_index=True, use_container_width=True)
                    if it.quality.get("exif"):
                        st.json(it.quality["exif"], expanded=False)
                if "face" in it.artifacts:
                    st.image(it.artifacts["face"], caption="Face used for identity match", width=140)

    st.download_button("⬇️ Download case report (HTML)", build_html(report), f"{report.claim_id}.html", "text/html")
    st.download_button("⬇️ Download raw result (JSON)", json.dumps(report.to_dict(), default=str, indent=2),
                       f"{report.claim_id}.json", "application/json")
    with st.expander("Processing details"):
        st.write({"timings_ms": report.timings_ms, "model_versions": report.model_versions})


# ------------------------------------------------------------------ pages
def page_analyze() -> None:
    st.title("🛡️ ClaimGuard — motor claim evidence check")
    st.caption("Upload everything submitted with one claim. Photos, invoices, RC, licence and selfie are analysed "
               "together as one case.")
    with st.form("claim"):
        a, b, c, d = st.columns(4)
        name = a.text_input("Claimant name")
        policy = b.text_input("Policy no.")
        plate = c.text_input("Vehicle no.", placeholder="MH 12 AB 1234")
        acc = d.date_input("Accident date", value=None, max_value=date.today())
        files = st.file_uploader("Claim files", type=["jpg", "jpeg", "png", "webp", "bmp", "pdf"],
                                 accept_multiple_files=True)
        roles = {}
        if files:
            st.caption("File types are auto-detected; override if needed.")
            cols = st.columns(min(4, len(files)))
            for k, f in enumerate(files):
                roles[f.name] = cols[k % len(cols)].selectbox(
                    f.name[:28], list(ROLE_LABELS), format_func=lambda r: ROLE_LABELS[r], key=f"role_{f.name}")
        submitted = st.form_submit_button("Analyze claim", type="primary", use_container_width=True)

    samples = sorted(p for p in SAMPLES.glob("*") if p.is_dir()) if SAMPLES.exists() else []
    if samples:
        a, b = st.columns([3, 1])
        demo = a.selectbox("…or load a prepared demo claim", samples, format_func=lambda p: p.name.replace("_", " "))
        b.write(""); b.write("")
        if b.button("Run demo claim", use_container_width=True):
            info = json.loads((demo / "info.json").read_text()) if (demo / "info.json").exists() else {}
            files_ = [(p.name, p.read_bytes(), None) for p in sorted(demo.iterdir()) if p.name != "info.json"]
            run(files_, ClaimInfo(**{k: info.get(k, "") for k in ("claimant_name", "policy_no",
                                                                   "vehicle_no", "accident_date")}))

    if submitted:
        if not files:
            st.warning("Add at least one file.")
            return
        info = ClaimInfo(name.strip(), policy.strip(), plate.strip(), acc.isoformat() if acc else "")
        run([(f.name, f.getvalue(), roles.get(f.name)) for f in files], info)
    if "report" in st.session_state:
        render_report(st.session_state["report"])


def run(files: list[tuple], info: ClaimInfo) -> None:
    st.session_state.pop("report", None)
    warm_up()
    bar = st.progress(0.0, "Starting…")
    st.session_state["report"] = analyze_claim(files, info, progress=lambda p, m: bar.progress(min(p, 1.0), m))
    bar.empty()


def page_history() -> None:
    st.title("Claim history")
    rows = store.list_claims()
    if not rows:
        st.info("No claims analysed yet.")
        return
    df = pd.DataFrame([dict(r) for r in rows])
    df["overall_risk"] = (df["overall_risk"] * 100).round().astype(int).astype(str) + "%"
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption("Every analysed claim feeds the photo and face index used for cross-claim reuse detection.")
    if st.button("Clear history (demo reset)"):
        store.clear_all()
        st.rerun()


def page_method() -> None:
    st.title("How the score works")
    cfg = settings()["risk"]
    st.markdown(f"""
**Pipeline:** intake → quality gate → per-file analyzers → claim-level checks → risk engine → explanation.

* Every analyzer emits **signals** (probability + reliability + evidence). The engine converts each to
  **log-odds × group weight × reliability**, so contributions **add up** and are shown in the waterfall.
* Correlated signals share a **group**; a group keeps its strongest fraud-side contribution, so the same
  evidence is never counted twice, and genuine photos can't cancel a fake one.
* **Reliability** comes from the quality gate: blurry, tiny or heavily recompressed inputs move the score less.
* **INCONCLUSIVE** is returned when detectors disagree or all evidence is low quality, instead of guessing.
* Tiers: LOW < {cfg['tiers']['medium']:.0%} ≤ MEDIUM < {cfg['tiers']['high']:.0%} ≤ HIGH. Hard rules can raise the
  floor (e.g. a photo reused from another claim is always HIGH).
""")
    st.subheader("Group weights")
    st.dataframe(pd.DataFrame([{"group": GROUP_LABELS.get(g, g), "weight": w} for g, w in cfg["group_weights"].items()]),
                 hide_index=True)
    st.subheader("Reason-code catalogue")
    st.dataframe(pd.DataFrame([{"code": c, "title": m["title"], "group": m.get("group") or "info"}
                               for c, m in reason_catalog().items()]), hide_index=True, use_container_width=True)
    rep = Path(__file__).parent / "reports" / "eval_report.md"
    if rep.exists():
        st.subheader("Measured accuracy")
        st.markdown(rep.read_text(encoding="utf-8"))


PAGES = {"Analyze claim": page_analyze, "Claim history": page_history, "How scoring works": page_method}
with st.sidebar:
    st.markdown("## 🛡️ ClaimGuard")
    choice = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("All analysis runs locally; no claim data leaves this machine.")
    st.caption("Screening signal, not proof. A human makes the final decision.")
PAGES[choice]()
