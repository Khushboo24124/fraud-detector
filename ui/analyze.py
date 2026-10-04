"""Analyze page: a guided, step-by-step flow. All results come from core.pipeline.analyze_claim();
this file only collects inputs and displays the ClaimReport — it never re-scores anything."""
from __future__ import annotations

import io
import json
import math
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

from core import risk, store
from core.config import BLOB_DIR, settings
from core.pipeline import analyze_claim, claim_analyzers, item_analyzers
from core.report import build_html
from core.schemas import ClaimInfo, ClaimReport, Kind, Role, Tier

from . import report_io
from .components import (check_card, md, reason_card, score_tile, stepper, verdict_card)
from .pages_resources import DEMO_TEXT, demo_folders
from .pages_tech import GROUP_LABELS
from .theme import TIER_STYLE

ROLE_LABELS = {None: "Auto-detect", Role.DAMAGE_PHOTO: "Damage photo", Role.INVOICE: "Repair invoice",
               Role.RC: "Registration certificate (RC)", Role.LICENCE: "Driving licence",
               Role.ID_CARD: "ID card (Aadhaar, PAN, Voter ID, passport)",
               Role.SELFIE: "Claimant selfie", Role.CLAIM_FORM: "Claim form", Role.OTHER_DOC: "Other document"}
# Live checklist: pipeline progress messages -> friendly names (analyzer names come from core/analyzers)
STAGES = [("classify", "Classifying files"), ("ai_ensemble", "AI photo check"), ("image_forensics", "Metadata check"),
          ("document_forensics", "Document checks"), ("cross_document", "Cross-checks between files"),
          ("identity", "Identity check"), ("cross_claim_reuse", "Reuse check (past claims)"), ("score", "Scoring")]
S = st.session_state


# ---------------------------------------------------------------- state
def reset() -> None:
    for k in [k for k in S.keys() if k.startswith("az_")]:
        del S[k]


def init() -> None:
    S.setdefault("az_step", 1)
    S.setdefault("az_info", {})
    S.setdefault("az_files", [])
    S.setdefault("az_up_key", 0)


@st.cache_resource(show_spinner="Loading detection models (first run only)…")
def warm_up() -> bool:
    _ = (item_analyzers(), claim_analyzers())
    return True


# ---------------------------------------------------------------- running
def run(files: list[tuple], info: ClaimInfo) -> None:
    """Runs the backend once and shows a live checklist while it works."""
    warm_up()
    box = st.empty()
    seen: list[str] = []

    def draw(current: str | None, finished: bool = False) -> None:
        rows = []
        for key, label in STAGES:
            if finished:
                mark = "✅" if key in seen else "➖"
                tail = "" if key in seen else " <span style='color:#94A3B8'>(not needed for these files)</span>"
            elif key == current:
                mark, tail = "⏳", " <span style='color:#2F5BEA'>running…</span>"
            elif key in seen:
                mark, tail = "✅", ""
            else:
                mark, tail = "○", ""
            rows.append(f"<div class='cg-live-row'>{mark} {label}{tail}</div>")
        box.markdown("<div class='cg-card'><h4>Checking your claim</h4>" + "".join(rows) + "</div>",
                     unsafe_allow_html=True)

    def progress(_: float, msg: str) -> None:
        key = ("classify" if msg.startswith("Classifying") else msg.split(" → ")[0] if " → " in msg
               else msg.split(": ", 1)[1] if msg.startswith("Claim-level check") else
               "score" if msg.startswith("Scoring") else None)
        if key and key not in seen:
            seen.append(key)
        if key:
            draw(key)

    draw("classify")
    report = analyze_claim(files, info, progress=progress)
    draw(None, finished=True)
    S["az_report"] = report
    S["az_step"] = 4


def run_demo(name: str) -> None:
    folder = next((f for f in demo_folders() if f.name == name), None)
    if folder is None:
        st.error(f"Demo '{name}' not found.")
        return
    raw = json.loads((folder / "info.json").read_text()) if (folder / "info.json").exists() else {}
    info = {k: raw.get(k, "") for k in ("claimant_name", "policy_no", "vehicle_no", "accident_date")}
    files = [{"name": p.name, "data": p.read_bytes(), "role": None}
             for p in sorted(folder.iterdir()) if p.name != "info.json"]
    reset(); init()
    S["az_info"], S["az_files"], S["az_demo"] = info, files, name
    run([(f["name"], f["data"], None) for f in files], ClaimInfo(**info))


# ---------------------------------------------------------------- toolbar (always visible on the page)
def clear_everything(delete_files: bool) -> int:
    """Clears the SQLite history (claims + photo/face index); optionally deletes stored uploads too."""
    store.clear_all()
    removed = 0
    if delete_files and BLOB_DIR.exists():
        for f in BLOB_DIR.iterdir():
            if f.is_file():
                try:
                    f.unlink(); removed += 1
                except OSError:
                    pass
    return removed


def toolbar() -> None:
    rows = store.list_claims(20)
    c1, c2, c3, c4 = st.columns([1, 1.4, 1.2, 1.2])
    if c1.button("＋ New analysis", type="primary", width="stretch"):
        reset(); st.rerun()

    with c2.popover(f"📂 Recent analyses ({len(rows)})", width="stretch"):
        if not rows:
            st.caption("No claims analysed yet.")
        for r in rows:
            icon = TIER_STYLE[Tier(r["tier"])][2]
            who = r["claimant_name"] or "No name"
            label = f"{icon} {r['tier']} {r['overall_risk']:.0%} · {who} · {r['created_at'][5:16].replace('T', ' ')}"
            if st.button(label, key=f"hist_{r['id']}", width="stretch"):
                rep = report_io.load(r["id"])
                if rep:
                    reset(); init()
                    S["az_report"], S["az_step"] = rep, 4
                    S["az_info"] = {"claimant_name": rep.info.claimant_name, "policy_no": rep.info.policy_no,
                                    "vehicle_no": rep.info.vehicle_no, "accident_date": rep.info.accident_date}
                    st.rerun()

    with c3.popover("▶ Demo claims", width="stretch"):
        for folder in demo_folders():
            title = DEMO_TEXT.get(folder.name, (folder.name,))[0]
            if st.button(f"▶ {title}", key=f"demo_{folder.name}", width="stretch"):
                S["az_run_demo"] = folder.name
                st.rerun()

    with c4.popover("🗑 Clear history", width="stretch"):
        st.markdown("Deletes **all saved claims** and the photo/face fingerprints used for reuse checks. "
                    "After this, no old upload can be flagged as reused.")
        files_too = st.checkbox("Also delete the uploaded files (data/blobs)", value=True)
        if st.button("Yes, clear everything", type="primary", width="stretch"):
            n = clear_everything(files_too)
            reset()
            S["az_flash"] = "History cleared." + (f" {n} stored file(s) deleted." if files_too else "")
            st.rerun()


# ---------------------------------------------------------------- steps 1-3
def step_details() -> None:
    info = S["az_info"]
    if S["az_step"] > 1:
        filled = [f"{k.replace('_', ' ').title()}: {v}" for k, v in info.items() if v]
        with st.expander("① Claim details — " + (", ".join(filled) if filled else "skipped"), expanded=False):
            if st.button("Edit claim details"):
                S.pop("az_report", None); S["az_step"] = 1; st.rerun()
        return
    st.markdown("### ① Claim details")
    st.caption("Optional, but it lets us cross-check names, vehicle numbers and dates against the documents.")
    with st.container(border=True):
        a, b = st.columns(2)
        name = a.text_input("Claimant name", info.get("claimant_name", ""))
        policy = b.text_input("Policy no.", info.get("policy_no", ""))
        c, d = st.columns(2)
        plate = c.text_input("Vehicle no.", info.get("vehicle_no", ""), placeholder="MH 12 AB 1234")
        acc_val = date.fromisoformat(info["accident_date"]) if info.get("accident_date") else None
        acc = d.date_input("Accident date", value=acc_val, max_value=date.today())
        x, y, _ = st.columns([1, 1, 4])
        if x.button("Next →", type="primary"):
            S["az_info"] = {"claimant_name": name.strip(), "policy_no": policy.strip(),
                            "vehicle_no": plate.strip(), "accident_date": acc.isoformat() if acc else ""}
            S["az_step"] = 2; st.rerun()
        if y.button("Skip"):
            S["az_step"] = 2; st.rerun()


def _thumb(f: dict) -> None:
    if f["data"][:5] == b"%PDF-":
        md("<div style='height:120px;display:flex;align-items:center;justify-content:center;font-size:3rem;"
           "background:#F1F4FF;border-radius:10px'>📄</div>")
        return
    try:
        img = Image.open(io.BytesIO(f["data"]))
        img.thumbnail((360, 240))
        st.image(img, width="stretch")
    except Exception:
        md("<div style='height:120px;display:flex;align-items:center;justify-content:center;font-size:2rem'>❓</div>")


def step_upload() -> None:
    files = S["az_files"]
    if S["az_step"] > 2:
        report = S.get("az_report")
        names = [f["name"] for f in files] or ([it.filename for it in report.items] if report else [])
        with st.expander(f"② Evidence — {len(names)} file(s): " + ", ".join(names), expanded=False):
            if files and st.button("Change files"):
                S.pop("az_report", None); S["az_step"] = 2; st.rerun()
        return
    if S["az_step"] < 2:
        return
    st.markdown("### ② Upload evidence")
    st.caption("Add everything from one claim: damage photos, repair bill, RC, driving licence, selfie.")
    new = st.file_uploader("Drag and drop files here", type=["jpg", "jpeg", "png", "webp", "bmp", "pdf"],
                           accept_multiple_files=True, key=f"uploader_{S['az_up_key']}")
    if new:
        have = {(f["name"], len(f["data"])) for f in files}
        for u in new:
            data = u.getvalue()
            if (u.name, len(data)) not in have:
                files.append({"name": u.name, "data": data, "role": None})
        S["az_up_key"] += 1
        st.rerun()
    if files:
        st.info("Tip: for the face match, set the ID (driving licence, Aadhaar, PAN, voter ID…) to its type and the "
                "person's photo to **Claimant selfie**.")
        cols = st.columns(4)
        for k, f in enumerate(list(files)):
            with cols[k % 4]:
                with st.container(border=True):
                    _thumb(f)
                    st.caption(f"**{f['name'][:28]}** · {len(f['data']) / 1024:.0f} KB")
                    f["role"] = st.selectbox("File type", list(ROLE_LABELS), index=list(ROLE_LABELS).index(f["role"]),
                                             format_func=lambda r: ROLE_LABELS[r], key=f"role_{k}_{f['name']}",
                                             label_visibility="collapsed")
                    if st.button("✕ Remove", key=f"rm_{k}_{f['name']}", width="stretch"):
                        files.pop(k); st.rerun()
    a, b, _ = st.columns([1, 1.4, 3])
    if a.button("← Back"):
        S["az_step"] = 1; st.rerun()
    if b.button("Next: run analysis →", type="primary", disabled=not files):
        S["az_step"] = 3; st.rerun()


def step_run() -> None:
    if S["az_step"] != 3:
        return
    st.markdown("### ③ Run analysis")
    st.caption("Every file is checked, then the whole claim is cross-checked and scored. This takes a few seconds.")
    if st.button("🔍 Run analysis", type="primary", width="stretch"):
        info = ClaimInfo(**{k: S["az_info"].get(k, "") for k in ("claimant_name", "policy_no", "vehicle_no", "accident_date")})
        run([(f["name"], f["data"], f["role"]) for f in S["az_files"]], info)
        st.rerun()
    if st.button("← Back to files"):
        S["az_step"] = 2; st.rerun()


# ---------------------------------------------------------------- charts (moved unchanged from the old app)
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
    fig.update_layout(height=250, margin=dict(l=30, r=40, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def waterfall(report: ClaimReport) -> go.Figure:
    prior = settings()["risk"]["prior_logit"]
    pooled = sorted(risk.pool_groups(report.signals).items(), key=lambda kv: -abs(kv[1]))
    names = ["Base rate"] + [GROUP_LABELS.get(g, g) for g, _ in pooled] + ["Final score"]
    vals = [prior] + [v for _, v in pooled] + [0]
    final = prior + sum(v for _, v in pooled)
    text = [f"{1 / (1 + math.exp(-prior)):.0%}"] + [f"{v:+.2f}" for _, v in pooled] + [f"{1 / (1 + math.exp(-final)):.0%}"]
    fig = go.Figure(go.Waterfall(x=names, y=vals, measure=["absolute"] + ["relative"] * len(pooled) + ["total"],
                                 text=text, textposition="outside",
                                 increasing={"marker": {"color": "#dc2626"}}, decreasing={"marker": {"color": "#16a34a"}},
                                 totals={"marker": {"color": TIER_STYLE[report.tier][0]}}))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), showlegend=False,
                      yaxis_title="log-odds of fraud", paper_bgcolor="rgba(0,0,0,0)",
                      title="Red pushes towards fraud, green towards genuine")
    return fig


def draw_boxes(img: Image.Image, boxes: list, scale: float = 1.0) -> Image.Image:
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


# ---------------------------------------------------------------- per-file check cards
def file_checks(item, report: ClaimReport) -> list[str]:
    """The 5 checks for one file, with their real result for this file (from the report's signals)."""
    text = {r.code + str(r.item_id): r.text for r in report.reasons}
    mine = [s for s in report.signals if s.item_id == item.id]
    codes = {s.code: s for s in mine}
    t = lambda s: text.get(s.code + str(s.item_id), s.code)  # noqa: E731
    out = []

    # 1. AI-generated photo check
    ai = next((s for s in mine if s.code.startswith("IMG-AI")), None)
    if ai:
        status = {"IMG-AI-01": "flag", "IMG-AI-02": "warn", "IMG-AI-00": "ok"}[ai.code]
        out.append(check_card("AI-generated photo check", status, t(ai)))
    else:
        out.append(check_card("AI-generated photo check", "na", "Runs on damage photos and selfies only."))

    # 2. Metadata check
    meta = [s for s in mine if s.code.startswith("IMG-META") or s.code.startswith("DOC-META")]
    if meta:
        out.append(check_card("Metadata check", "flag", " ".join(t(s) for s in meta)))
    elif item.kind is Kind.IMAGE:
        detail = ("No AI tags, editing software or date problems found." if item.quality.get("exif")
                  else "No camera details in the file (common after WhatsApp). Nothing suspicious found.")
        out.append(check_card("Metadata check", "ok", detail))
    else:
        out.append(check_card("Metadata check", "ok", "No editing tool or suspicious edit history in the PDF."))

    # 3. Reuse check
    reuse = [s for s in mine if s.code in ("XCLM-FILE-REUSE", "XCLM-PHOTO-REUSE", "XCLM-FACE-REUSE")]
    if reuse:
        out.append(check_card("Reuse check", "flag", " ".join(t(s) for s in reuse)))
    elif item.role in (Role.DAMAGE_PHOTO, Role.SELFIE) or item.role.is_identity:
        out.append(check_card("Reuse check", "ok", "Not seen in any earlier claim."))
    else:
        out.append(check_card("Reuse check", "na", "Runs on photos, selfies and ID cards."))

    # 4. Document check
    if item.role.is_document or item.kind is Kind.PDF:
        bad = [s for s in mine if s.code.startswith("DOC-") and s.fraud_prob and s.fraud_prob > 0.5]
        if bad:
            out.append(check_card("Document check", "flag", " ".join(t(s) for s in bad[:3])))
        elif "DOC-Q-01" in codes:
            out.append(check_card("Document check", "warn", t(codes["DOC-Q-01"])))
        elif "DOC-OK-00" in codes:
            out.append(check_card("Document check", "ok", t(codes["DOC-OK-00"])))
        else:
            out.append(check_card("Document check", "ok", "No tampering traces found."))
        if "DOC-DOM-00" in codes:
            out.append(check_card("Bill rules", "note", t(codes["DOC-DOM-00"])))
    else:
        out.append(check_card("Document check", "na", "Not a document."))

    # 5. Identity check
    idsig = next((s for s in report.signals if s.code.startswith("ID-FACE")), None)
    used = idsig and item.id in (idsig.evidence.get("id_item"), idsig.evidence.get("selfie_item"))
    if item.role is Role.SELFIE or item.role.is_identity or used:
        if idsig:
            status = {"ID-FACE-00": "ok", "ID-FACE-01": "flag", "ID-FACE-02": "warn", "ID-FACE-03": "na"}[idsig.code]
            out.append(check_card("Identity check", status, t(idsig)))
        else:
            out.append(check_card("Identity check", "na", "Needs a selfie and a photo ID (licence, Aadhaar, PAN…) in the claim."))
    else:
        out.append(check_card("Identity check", "na", "Runs on a photo ID and a selfie."))

    # extra: cross-checks that point at this file
    xchk = [s for s in report.signals if s.code.startswith(("DOC-XCHK", "IMG-XCHK")) and s.item_id == item.id]
    if xchk:
        out.append(check_card("Cross-check with other files", "flag", " ".join(t(s) for s in xchk)))
    return out


def evidence(report: ClaimReport) -> None:
    faces = [it for it in report.items if "face" in it.artifacts]
    idsig = next((s for s in report.signals if s.code.startswith("ID-FACE") and s.code != "ID-FACE-03"), None)
    names = [it.filename[:22] for it in report.items] + (["🧑 Face match"] if len(faces) >= 2 and idsig else [])
    tabs = st.tabs(names or ["—"])
    for tab, it in zip(tabs, report.items):
        with tab:
            st.caption(f"{ROLE_LABELS.get(it.role, it.role.value)} · reliability {it.reliability:.0%}"
                       + (f" · {', '.join(it.quality.get('issues', []))}" if it.quality.get("issues") else ""))
            checks = file_checks(it, report)
            left, mid, right = st.columns([1, 1.5, 1])
            half = (len(checks) + 1) // 2
            with left:
                md("".join(checks[:half]))
            with mid:
                if it.role.is_document or it.kind is Kind.PDF:
                    for page in document_preview(it, report):
                        st.image(page, width="stretch")
                    st.caption("Red boxes = suspicious spots found by the document checks.")
                else:
                    st.image(it.path, width="stretch")
            with right:
                md("".join(checks[half:]))
            ai = next((s for s in report.signals if s.item_id == it.id and s.code.startswith("IMG-AI")), None)
            if ai:
                st.dataframe(pd.DataFrame([{"Detector": m, "P(AI-generated)": f"{p:.1%}"}
                                           for m, p in ai.evidence["per_model"].items()]
                                          + [{"Detector": "Combined (calibrated)", "P(AI-generated)": f"{ai.evidence['prob']:.1%}"}]),
                             hide_index=True, width="stretch")
            if it.fields:
                with st.expander("Details read from the document"):
                    f = {k: v for k, v in it.fields.items() if k != "items" and v not in (None, [], "")}
                    st.json(f, expanded=True)
                    if it.fields.get("items"):
                        st.dataframe(pd.DataFrame(it.fields["items"]), hide_index=True, width="stretch")
            if it.quality.get("exif"):
                with st.expander("Camera details (EXIF)"):
                    st.json(it.quality["exif"], expanded=False)
            if "heatmap" in it.artifacts:
                with st.expander("Compression & noise view (investigator aid, not scored)"):
                    st.caption("Highlights areas whose JPEG compression or sensor noise differs from the rest. Lights, "
                               "dark sky, texture and blur also glow. It does not show where AI was found and never "
                               "changes the score.")
                    st.image(it.artifacts["heatmap"], width="stretch")
    if len(tabs) > len(report.items):
        with tabs[-1]:
            lic = next((it for it in faces if it.id == idsig.evidence.get("id_item")), faces[0])
            sel = next((it for it in faces if it.id == idsig.evidence.get("selfie_item")), faces[-1])
            a, b, c = st.columns([1, 1, 1.4])
            a.image(lic.artifacts["face"], caption=f"ID: {lic.filename}", width=180)
            b.image(sel.artifacts["face"], caption=f"Selfie: {sel.filename}", width=180)
            th = settings()["thresholds"]
            status = {"ID-FACE-00": "ok", "ID-FACE-01": "flag", "ID-FACE-02": "warn"}[idsig.code]
            c.markdown(check_card("Face match", status,
                                  f"Similarity {idsig.evidence['similarity']:.2f}. Match at {th['face_match']:.2f} or "
                                  f"more; different person below {th['face_mismatch']:.2f}."), unsafe_allow_html=True)


# ---------------------------------------------------------------- results
def results(report: ClaimReport) -> None:
    st.markdown("### ④ Verdict")
    verdict_card(report)
    st.write("")
    st.markdown("### ⑤ Details")

    st.markdown("#### Scores")
    g, tiles = st.columns([1, 1.6])
    with g:
        st.plotly_chart(gauge(report.overall_risk, report.tier), width="stretch", config={"displayModeBar": False})
    with tiles:
        st.write("")
        cols = st.columns(3)
        for col, (name, label) in zip(cols, [("image", "Image authenticity"), ("document", "Document authenticity"),
                                             ("identity", "Identity & reuse")]):
            d = report.domains[name]
            hint = "not checked (no files of this type)" if not d.evaluated else "higher = more genuine"
            col.markdown(score_tile(label, f"{d.authenticity:.0%}" if d.evaluated else "—", hint), unsafe_allow_html=True)
        st.caption("Each score looks only at its own evidence and starts from the normal fraud base rate. "
                   "'—' means nothing of that type was uploaded.")

    st.markdown("#### Why this score")
    scored = [r for r in report.reasons if r.contribution != 0]
    notes = [r for r in report.reasons if r.contribution == 0]
    if not scored:
        st.caption("No finding moved the score.")
    for r in scored:
        reason_card(r)
    if notes:
        with st.expander(f"Notes that did not change the score ({len(notes)})"):
            for r in notes:
                reason_card(r)

    with st.expander("See how the score was calculated"):
        st.plotly_chart(waterfall(report), width="stretch", config={"displayModeBar": False})
        if report.overrides:
            st.caption("Rule applied: " + "; ".join(report.overrides))

    st.markdown("#### Evidence")
    evidence(report)

    st.markdown("#### Download")
    a, b, _ = st.columns([1, 1, 2])
    a.download_button("⬇️ Case report (HTML)", build_html(report), f"{report.claim_id}.html", "text/html", width="stretch")
    b.download_button("⬇️ Raw result (JSON)", json.dumps(report.to_dict(), default=str, indent=2),
                      f"{report.claim_id}.json", "application/json", width="stretch")
    with st.expander("Processing details"):
        st.write({"timings_ms": report.timings_ms, "model_versions": report.model_versions})


# ---------------------------------------------------------------- page
def render() -> None:
    init()
    st.markdown("<h1 style='margin-bottom:0'>Analyze a claim</h1>", unsafe_allow_html=True)
    st.caption("Follow the steps. Everything runs on this machine. A person makes the final decision.")
    toolbar()
    if S.get("az_flash"):
        st.success(S.pop("az_flash"))

    demo = S.pop("az_run_demo", None) or st.query_params.get("demo")
    if demo:
        if "demo" in st.query_params:
            del st.query_params["demo"]
        stepper(3, 2)
        st.markdown(f"### Running demo: {DEMO_TEXT.get(demo, (demo,))[0]}")
        run_demo(demo)
        st.rerun()

    report = S.get("az_report")
    stepper(5 if report else S["az_step"], 5 if report else S["az_step"] - 1)
    step_details()
    step_upload()
    step_run()
    if report:
        if S.get("az_demo"):
            st.caption(f"Demo claim: {DEMO_TEXT.get(S['az_demo'], (S['az_demo'],))[0]}")
        results(report)
